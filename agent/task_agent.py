"""The task agent: takes "bring me the pills", plans the steps and runs them, asking the user before anything moves.

    python3 -m agent.task_agent "bring me the pills"                  pretend table, pretend arm
    python3 -m agent.task_agent "bring me the snack" --photo desk.jpg the objects come from Nova (#5)

Built with LangChain deepagents:
    - a perception sub-agent that finds the objects (list_objects),
    - a caregiver sub-agent that writes and sends the message (notify),
    - a notes file the agent keeps between runs ("pills are usually next to the cup"),
    - the main agent writes a plan (write_todos), asks the user to confirm with their eyes, then picks and drops.

Safety lock: pick() refuses unless the user confirmed that exact object, whatever the model says.

The tools are pretend for now (no arm, camera or Photon needed); the app plugs in the real ones later.

Environment variables:
    AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY   the agent's model runs on Amazon Bedrock
    AWS_REGION                                 default us-east-1
    AGENT_MODEL_ID                             default us.amazon.nova-2-lite-v1:0
    EYEREACH_NOTES                             notes folder, default ~/.eyereach/agent
"""

import argparse
import os
import sys
from pathlib import Path

from langchain_core.tools import tool

NOTES_DIR = Path(os.environ.get("EYEREACH_NOTES", Path.home() / ".eyereach" / "agent"))
NOTES_FILE = "/notes.md"                                      # inside NOTES_DIR
MODEL = os.environ.get("AGENT_MODEL_ID", "us.amazon.nova-2-lite-v1:0")
REGION = os.environ.get("AWS_REGION", "us-east-1")

PRETEND_TABLE = [{"name": "cup", "u": 0.30, "v": 0.55}, {"name": "pill bottle", "u": 0.42, "v": 0.60},
                 {"name": "phone", "u": 0.75, "v": 0.70}]

SYSTEM_PROMPT = f"""You help a person who cannot move their arms. They control a robot arm with their eyes.
For each task, first write the plan with write_todos, then:
1. Read your notes in {NOTES_FILE}; they may say where things usually are.
2. Ask the perception agent (task tool) which objects are on the table and which one the person means.
3. If it is unsure, or two objects could match, ask the person a question instead of guessing.
4. Call ask_user_to_confirm with the object's name. Never pick before the person says yes.
   If they say no, stop and say so.
5. pick the object, then drop it next to the person.
6. Ask the caregiver agent (task tool) to tell the caregiver what was done.
7. After a successful pick, always add one line to {NOTES_FILE} (edit_file, or write_file if it does not
   exist yet) saying which object the person meant and what was next to it, for example
   "pills: the pill bottle, next to the cup". Use these notes on later tasks.
Keep replies short."""

PERCEPTION_PROMPT = """You find objects for a robot arm. Call list_objects, then answer with the object that
matches the request (its exact name from the list) and where it is. If none or several match, say so."""

CAREGIVER_PROMPT = """You keep the person's caregiver informed. Write one short, friendly sentence about what
happened and send it with notify. Do not add anything that did not happen."""


class Robot:
    """The pretend arm and table. Records every action in self.log; the safety lock lives in pick()."""

    def __init__(self, objects=None, confirm=None, notify=None):
        self.objects = list(PRETEND_TABLE if objects is None else objects)
        self.confirm = confirm or (lambda name: input(f"Is it the {name}? (look and hold) [y/n] ").lower() == "y")
        self.send = notify or (lambda text: print(f"[to caregiver] {text}"))
        self.confirmed = None                                 # the one object the user said yes to
        self.holding = None
        self.log = []

    @classmethod
    def from_photo(cls, path, **kwargs):
        """The table as Nova (#5) sees it in a photo: each object at its box centre, 0-1 across and down."""
        import cv2
        from eyereach import vision_nova
        image = cv2.imread(str(path))
        if image is None:
            sys.exit(f"could not read {path}")
        height, width = image.shape[:2]
        objects = [{"name": d.name, "u": round((d.box[0] + d.box[2]) / 2 / width, 2),
                    "v": round((d.box[1] + d.box[3]) / 2 / height, 2)} for d in vision_nova.detect_objects(image)]
        return cls(objects, **kwargs)

    def tools(self):
        @tool
        def list_objects() -> list:
            """The objects on the table: name, and u, v (0-1 across and down the camera view)."""
            self.log.append(("list_objects",))
            return self.objects

        @tool
        def ask_user_to_confirm(name: str) -> bool:
            """Ask the person to confirm with their eyes that `name` is the object they want. True means yes."""
            said_yes = bool(self.confirm(name))
            self.log.append(("confirm", name, said_yes))
            self.confirmed = name if said_yes else None
            return said_yes

        @tool
        def pick(name: str) -> str:
            """Move the arm to the object and grab it. Only works after the person confirmed this object."""
            if name != self.confirmed:
                self.log.append(("refused", name))
                return f"Refused: the person has not confirmed the {name}. Ask them first."
            if name not in {o["name"] for o in self.objects}:
                self.log.append(("refused", name))
                return f"Refused: there is no {name} on the table."
            self.log.append(("pick", name))
            self.holding, self.confirmed = name, None
            return f"Holding the {name}."

        @tool
        def drop() -> str:
            """Put down what the arm is holding, next to the person."""
            if not self.holding:
                return "Nothing to drop."
            self.log.append(("drop", self.holding))
            dropped, self.holding = self.holding, None
            return (f"Dropped the {dropped} next to the person. Now add one line about the {dropped} to "
                    f"{NOTES_FILE} (what the person called it, what it was next to), then tell the caregiver.")

        @tool
        def notify(text: str) -> str:
            """Send a short message to the caregiver."""
            self.log.append(("notify", text))
            self.send(text)
            return "Sent."

        return {"list_objects": list_objects, "ask_user_to_confirm": ask_user_to_confirm, "pick": pick,
                "drop": drop, "notify": notify}


def default_model():
    from langchain.chat_models import init_chat_model
    return init_chat_model(f"bedrock_converse:{MODEL}", region_name=REGION, temperature=0)


def build_agent(robot, model=None, notes_dir=NOTES_DIR):
    """The deep agent wired to the robot's tools, with its notes kept in notes_dir."""
    from deepagents import create_deep_agent
    from deepagents.backends import FilesystemBackend
    from langchain.agents.middleware import TodoListMiddleware

    notes_dir = Path(notes_dir)
    notes_dir.mkdir(parents=True, exist_ok=True)
    t = robot.tools()
    return create_deep_agent(
        model=model or default_model(),
        tools=[t["ask_user_to_confirm"], t["pick"], t["drop"]],
        system_prompt=SYSTEM_PROMPT,
        subagents=[
            {"name": "perception", "description": "Finds the objects on the table and says which one a request "
             "means. Give it the request.", "system_prompt": PERCEPTION_PROMPT, "tools": [t["list_objects"]]},
            {"name": "caregiver", "description": "Tells the caregiver what was done. Give it what happened.",
             "system_prompt": CAREGIVER_PROMPT, "tools": [t["notify"]]},
        ],
        backend=FilesystemBackend(root_dir=notes_dir, virtual_mode=True),
        memory=[NOTES_FILE],
        middleware=[TodoListMiddleware()],                    # write_todos: the agent writes down its plan
    )


def run(task, robot, model=None, notes_dir=NOTES_DIR):
    """Run one task -> the agent's last reply."""
    result = build_agent(robot, model, notes_dir).invoke({"messages": [{"role": "user", "content": task}]})
    return result["messages"][-1].content


def main():
    parser = argparse.ArgumentParser(description="Run the task agent with a pretend arm.")
    parser.add_argument("task", help='for example "bring me the pills"')
    parser.add_argument("--photo", help="take the objects from this photo with Nova instead of the pretend table")
    args = parser.parse_args()
    robot = Robot.from_photo(args.photo) if args.photo else Robot()
    print(run(args.task, robot))
    print("\nwhat happened:", *robot.log, sep="\n  ")


if __name__ == "__main__":
    main()
