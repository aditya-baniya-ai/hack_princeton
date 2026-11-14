"""No network, no arm: a scripted model plays the AI, step by step, against the pretend robot.
    python3 -m unittest discover -s tests
"""

import sys
import tempfile
import unittest
from itertools import count
from pathlib import Path

from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage
from pydantic import Field

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from agent.task_agent import Robot, run  # noqa: E402

ids = count()


def call(tool, **args):
    """The model asks for one tool."""
    return AIMessage("", tool_calls=[{"name": tool, "args": args, "id": f"call{next(ids)}"}])


def say(text):
    """The model answers in words (ends the agent or sub-agent)."""
    return AIMessage(text)


class Scripted(GenericFakeChatModel):
    """Replies with the given messages in order, and remembers every prompt it was shown."""
    seen: list = Field(default_factory=list)

    def bind_tools(self, tools, **kwargs):
        return self

    def _generate(self, messages, *args, **kwargs):
        self.seen.append(messages)
        return super()._generate(messages, *args, **kwargs)


def model(*steps):
    return Scripted(messages=iter(steps))


FIND_PILLS = [call("task", subagent_type="perception", description="Which object is the pills?"),
              call("list_objects"),                                    # the perception sub-agent
              say("The pill bottle, at u=0.42 v=0.60.")]
TELL_CAREGIVER = [call("task", subagent_type="caregiver", description="Pills brought to the person."),
                  call("notify", text="I brought the pills to your person."),   # the caregiver sub-agent
                  say("Sent.")]


class AgentTest(unittest.TestCase):
    def setUp(self):
        self.notes = tempfile.TemporaryDirectory()
        self.answers = []

    def tearDown(self):
        self.notes.cleanup()

    def robot(self, says_yes):
        def confirm(name):
            self.answers.append(name)
            return says_yes
        return Robot(confirm=confirm, notify=lambda text: None)

    def test_asks_before_picking(self):
        robot = self.robot(says_yes=True)
        run("bring me the pills", robot, model(
            call("write_todos", todos=[{"content": "find, confirm, pick, drop, tell", "status": "in_progress"}]),
            *FIND_PILLS,
            call("ask_user_to_confirm", name="pill bottle"),
            call("pick", name="pill bottle"),
            call("drop"),
            *TELL_CAREGIVER,
            say("Done: the pills are next to you.")), self.notes.name)
        steps = [entry[0] for entry in robot.log]
        self.assertEqual(steps, ["list_objects", "confirm", "pick", "drop", "notify"])
        self.assertEqual(robot.log[1], ("confirm", "pill bottle", True))

    def test_refuses_when_the_user_says_no(self):
        robot = self.robot(says_yes=False)
        run("bring me the pills", robot, model(
            *FIND_PILLS,
            call("ask_user_to_confirm", name="pill bottle"),
            call("pick", name="pill bottle"),                                # the model ignores the "no"...
            say("Stopped.")), self.notes.name)
        self.assertEqual(robot.log[-1], ("refused", "pill bottle"))         # ...and the arm still does not move
        self.assertNotIn("pick", [entry[0] for entry in robot.log])

    def test_never_picks_without_asking(self):
        robot = self.robot(says_yes=True)
        run("bring me the pills", robot, model(call("pick", name="pill bottle"), say("Stopped.")), self.notes.name)
        self.assertEqual(robot.log, [("refused", "pill bottle")])
        self.assertEqual(self.answers, [])

    def test_a_yes_is_for_one_object_only(self):
        robot = self.robot(says_yes=True)
        run("bring me the pills", robot, model(
            call("ask_user_to_confirm", name="cup"),
            call("pick", name="pill bottle"),
            say("Stopped.")), self.notes.name)
        self.assertEqual(robot.log[-1], ("refused", "pill bottle"))

    def test_reuses_a_note_from_an_earlier_run(self):
        run("bring me the pills", self.robot(says_yes=True), model(
            call("write_file", file_path="/notes.md", content="The pills are usually next to the cup.\n"),
            say("Noted.")), self.notes.name)
        self.assertIn("next to the cup", (Path(self.notes.name) / "notes.md").read_text())

        second = model(say("Looking next to the cup first."))
        run("bring me the pills again", self.robot(says_yes=True), second, self.notes.name)
        shown = " ".join(str(m.content) for m in second.seen[0])
        self.assertIn("The pills are usually next to the cup.", shown)      # the note is in the new run's prompt


class RobotTest(unittest.TestCase):
    def test_pick_needs_an_object_on_the_table(self):
        robot = Robot(confirm=lambda name: True)
        tools = robot.tools()
        tools["ask_user_to_confirm"].invoke({"name": "banana"})
        self.assertIn("no banana", tools["pick"].invoke({"name": "banana"}))

    def test_drop_with_empty_hand(self):
        self.assertEqual(Robot().tools()["drop"].invoke({}), "Nothing to drop.")


if __name__ == "__main__":
    unittest.main()
