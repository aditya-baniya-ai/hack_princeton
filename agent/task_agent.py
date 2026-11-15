"""The task agent: takes "bring me the pills", plans the steps and runs them, asking the user before anything moves.

    python3 -m agent.task_agent "bring me the pills"                  pretend table, pretend arm
    python3 -m agent.task_agent "bring me the snack" --photo desk.jpg the objects come from Nova (#5)
    python3 -m agent.task_agent "grab me a banana" --camera 2 --arm real   Nova + the real arm
    launch/eyereach.sh --agent "grab me a banana" --camera 2         the same, and it starts the arm first
    python3 -m agent.task_agent --serve --camera 2                    inside the eye app (it starts this itself)

Built with LangChain deepagents:
    - a perception sub-agent that finds the objects (list_objects),
    - a caregiver sub-agent that writes and sends the message (notify),
    - a notes file the agent keeps between runs ("pills are usually next to the cup"),
    - the main agent writes a plan (write_todos), asks the user to confirm with their eyes, then picks and drops.

Safety lock: pick() refuses unless the user confirmed that exact object, whatever the model says.

Without --arm the arm is pretend. With --arm, pick runs brain/fetch.py (in the arm's Python, which has ROS):
it glides over the object, grabs it, carries it next to the person and lets go. The table camera must be
calibrated once first (python3 brain/fetch.py calibrate --camera N). Asking the person is a y/n in the terminal.

Environment variables:
    AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY   the agent's model runs on Amazon Bedrock
    AWS_REGION                                 default us-east-1
    AGENT_MODEL_ID                             default us.amazon.nova-2-lite-v1:0
    EYEREACH_NOTES                             notes folder, default ~/.eyereach/agent
    EYEREACH_ARM_PYTHON                        the Python that runs brain/fetch.py, default ~/eyereach_venv
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from langchain_core.tools import tool

NOTES_DIR = Path(os.environ.get("EYEREACH_NOTES", Path.home() / ".eyereach" / "agent"))
NOTES_FILE = "/notes.md"                                      # inside NOTES_DIR
MODEL = os.environ.get("AGENT_MODEL_ID", "us.amazon.nova-2-lite-v1:0")
REGION = os.environ.get("AWS_REGION", "us-east-1")
ROOT = Path(__file__).resolve().parent.parent
ARM_PYTHON = os.environ.get("EYEREACH_ARM_PYTHON") or next(
    (str(p) for p in [Path.home() / "eyereach_venv" / "bin" / "python"] if p.exists()), "python3")

PRETEND_TABLE = [{"name": "cup", "u": 0.30, "v": 0.55}, {"name": "pill bottle", "u": 0.42, "v": 0.60},
                 {"name": "phone", "u": 0.75, "v": 0.70}]

from eyereach.project_context import PROJECT_CONTEXT  # noqa: E402  (what every Amazon model is told)

SYSTEM_PROMPT = PROJECT_CONTEXT + f"""

You are the planner that carries out the person's spoken requests ("find the cup", "bring me my pills").
For each task, first write the plan with write_todos, then:
1. Read your notes in {NOTES_FILE}; they may say where things usually are.
2. Ask the perception agent (task tool) which objects are on the table and which one the person means.
3. If it offers one likely look-alike (the camera may have misnamed it), go to step 4 with that name: the
   person's yes or no settles it. If two objects could match, or none, ask the person instead of guessing.
4. Call ask_user_to_confirm with the object's name. Never pick before the person says yes.
   If they say no, stop and say so.
5. pick the object, then drop it next to the person.
6. Ask the caregiver agent (task tool) to tell the caregiver what was done.
7. After a successful pick, always add one line to {NOTES_FILE} (edit_file, or write_file if it does not
   exist yet) saying which object the person meant and what was next to it, for example
   "pills: the pill bottle, next to the cup". Use these notes on later tasks.
Keep replies short."""

PERCEPTION_PROMPT = PROJECT_CONTEXT + """

You find objects for the robot arm. Call list_objects, then answer with the object that
matches the request (its exact name from the list) and where it is. The camera sometimes names a thing
wrongly (an orange called an apple): if nothing matches exactly but one object is a likely look-alike (same
kind of thing, similar size or colour), give that one and say it may be what was asked for. If none or
several match, say so."""

CAREGIVER_PROMPT = PROJECT_CONTEXT + """

You keep the person's caregiver informed. Write one short, friendly sentence about what
happened and send it with notify. Do not add anything that did not happen."""


class Robot:
    """The table and the arm. Records every action in self.log; the safety lock lives in pick().
    arm: None for the pretend arm, or fetch(object) -> None when it was brought, or why it wasn't."""

    def __init__(self, objects=None, confirm=None, notify=None, arm=None):
        self.objects = list(PRETEND_TABLE if objects is None else objects)
        self.arm = arm
        self.confirm = confirm or (lambda name: input(f"Is it the {name}? (look and hold) [y/n] ").lower() == "y")
        self.send = notify or (lambda text: print(f"[to caregiver] {text}"))
        self.confirmed = None                                 # the one object the user said yes to
        self.holding = None
        self.log = []

    @classmethod
    def from_photo(cls, path, **kwargs):
        """The table as Nova (#5) sees it in a photo: each object at its box centre, 0-1 across and down."""
        import cv2
        image = cv2.imread(str(path))
        if image is None:
            sys.exit(f"could not read {path}")
        return cls.from_image(image, **kwargs)

    @classmethod
    def from_camera(cls, camera, **kwargs):
        """The same, from one picture taken now by the table camera (/dev/video2 or 2), or from the picture file
        the eye app keeps fresh (~/.eyereach/table_camera.jpg) while it has the camera."""
        import cv2
        if str(camera).endswith(".jpg"):
            import time
            path = Path(camera).expanduser()
            for _ in range(30):                               # the app writes it every half second
                if path.exists() and time.time() - path.stat().st_mtime < 3:
                    image = cv2.imread(str(path))
                    if image is not None:
                        return cls.from_image(image, **kwargs)
                time.sleep(0.1)
            sys.exit(f"no fresh picture in {path} (is the eye app running with the table camera?)")
        capture = cv2.VideoCapture(int(camera) if str(camera).isdigit() else camera)
        for _ in range(5):                                    # the first frames are often dark
            ok, image = capture.read()
        capture.release()
        if not ok:
            sys.exit(f"no picture from camera {camera}")
        return cls.from_image(image, **kwargs)

    @classmethod
    def from_image(cls, image, **kwargs):
        from eyereach import vision_nova
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
            self.confirmed = None                             # a yes is for one try only
            if self.arm:
                spot = next(o for o in self.objects if o["name"] == name)
                why = self.arm(spot)
                if why:
                    self.log.append(("pick failed", name, why))
                    return f"The arm could not bring the {name}: {why}. Tell the person; do not retry on your own."
                self.log.append(("pick", name))
                self.holding = name
                return f"Brought the {name} next to the person and let go of it. Call drop to finish."
            self.log.append(("pick", name))
            self.holding = name
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


def fetch_with(arm, *extra):
    """fetch(object) for Robot(arm=...): runs brain/fetch.py go in the arm's Python."""
    def fetch(spot):
        flags = ["--virtual"] if arm == "virtual" else []
        done = subprocess.run([ARM_PYTHON, str(ROOT / "brain" / "fetch.py"), "go", str(spot["u"]), str(spot["v"]),
                               *flags, *extra], cwd=ROOT, stdout=subprocess.PIPE, text=True)
        lines = done.stdout.strip().splitlines()
        print(*lines, sep="\n")
        return None if done.returncode == 0 else (lines[-1] if lines else "the arm program failed")
    return fetch


def serve(camera=None, model=None, notes_dir=NOTES_DIR, lines=sys.stdin, out=None, objects=None):
    """Inside the eye app, which owns the arm and the screen. One JSON object per line both ways:
        app -> agent   {"task": "bring me the banana"}, then {"answer": true} and {"result": null or "why not"}
        agent -> app   {"status": "ready"}, {"confirm": "banana"}, {"pick": {"name", "u", "v"}},
                       {"caregiver": "..."}, {"reply": "..."} (the task is over)
    Each task takes a new picture with the table camera (objects: a fixed table instead, for tests)."""
    out = out or sys.stdout
    sys.stdout = sys.stderr                               # stray prints must not reach the app

    def send(**message):
        print(json.dumps(message), file=out, flush=True)

    def wait_for(key):
        for line in lines:
            try:
                message = json.loads(line)
            except ValueError:
                continue
            if key in message:
                return message[key]
        raise SystemExit(0)                               # the app closed

    def confirm(name):
        send(confirm=name)
        return bool(wait_for("answer"))

    def arm(spot):
        send(pick=spot)
        return wait_for("result")

    send(status="ready")
    for line in lines:
        try:
            task = json.loads(line).get("task")
        except (ValueError, AttributeError):
            continue
        if not task:
            continue
        try:
            kwargs = dict(confirm=confirm, notify=lambda text: send(caregiver=text), arm=arm)
            robot = Robot(objects, **kwargs) if objects is not None else Robot.from_camera(camera, **kwargs)
            send(reply=run(task, robot, model, notes_dir))
        except SystemExit as error:                       # no picture, say
            send(reply=f"I could not do that: {error}")
        except Exception as error:                        # Bedrock refused, the network is down, ...
            send(reply=f"I could not do that: {type(error).__name__}: {error}")


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
    parser = argparse.ArgumentParser(description="Run the task agent.")
    parser.add_argument("task", nargs="?", help='for example "bring me the pills"')
    parser.add_argument("--serve", action="store_true", help="inside the eye app: tasks come as JSON lines on stdin")
    parser.add_argument("--photo", help="take the objects from this photo with Nova instead of the pretend table")
    parser.add_argument("--camera", help="take a picture with this table camera (/dev/video2 or 2) for Nova")
    parser.add_argument("--arm", choices=["real", "virtual"], help="move an arm when picking (default: pretend)")
    parser.add_argument("--grab-mm", default="50", help="passed to brain/fetch.py")
    parser.add_argument("--travel-mm", default="120", help="passed to brain/fetch.py")
    args = parser.parse_args()
    if args.serve:
        if not args.camera:
            sys.exit("--serve needs --camera (the table camera)")
        return serve(args.camera)
    if not args.task:
        parser.error("give a task, for example \"bring me the pills\"")
    if args.arm == "real" and not (args.camera or args.photo):
        sys.exit("--arm real needs --camera (or --photo from the calibrated table camera)")
    arm = fetch_with(args.arm, "--grab-mm", args.grab_mm, "--travel-mm", args.travel_mm) if args.arm else None
    if args.camera:
        robot = Robot.from_camera(args.camera, arm=arm)
    elif args.photo:
        robot = Robot.from_photo(args.photo, arm=arm)
    else:
        robot = Robot(arm=arm)
    print(run(args.task, robot))
    print("\nwhat happened:", *robot.log, sep="\n  ")


if __name__ == "__main__":
    main()
