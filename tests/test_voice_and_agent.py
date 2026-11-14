"""Speaking stops the face; quiet, the face is back. And the task agent inside the app: ask, confirm, fetch.
    python3 -m unittest tests.test_voice_and_agent
"""

import sys
import time
import unittest
from pathlib import Path
from unittest.mock import Mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from eyereach import eye_control  # noqa: E402
from eyereach.agent_link import run_agent_messages  # noqa: E402
from eyereach.eye_control import VOICE_IDLE, GuidedControl, arm_link  # noqa: E402
from eyereach.voice_commands import SonicInput  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "brain"))
import fetch  # noqa: E402


def mock_arm():
    robot = Mock()
    robot.error, robot.hover, robot.closed = None, 120.0, False
    robot.busy.return_value = False
    robot.jog.return_value = True
    robot.gripper_area.return_value = (0.5, 0.5, 120.0)
    return robot


def auto(robot):
    ctl = GuidedControl(robot, movement="follow")
    ctl.voice_available, ctl.auto_mode = True, True
    return ctl


class AutoModeTest(unittest.TestCase):
    def test_speaking_stops_the_face_and_quiet_brings_it_back(self):
        robot = mock_arm()
        ctl = auto(robot)
        self.assertTrue(ctl.voice(("jog", "left", 20.0), 1.0))          # no "Robot use voice" needed
        self.assertEqual(ctl.control_mode, "voice")
        robot.jog.assert_called_once_with("left", 20.0)
        ctl.phase = "there"
        ctl.step(("map", (0.9, 0.2)), now=2.0)
        robot.follow.assert_not_called()                                 # the face doesn't move the arm now
        ctl.step(("map", (0.9, 0.2)), now=1.0 + VOICE_IDLE + 0.1)
        self.assertEqual(ctl.control_mode, "vision")                      # quiet: the face is back ...
        self.assertTrue(ctl.live_enabled)                                 # ... at once, no MOVE AGAIN
        ctl.step(("map", (0.9, 0.2)), now=1.0 + VOICE_IDLE + 0.2)
        robot.follow.assert_called()

    def test_stop_stays_stopped(self):
        ctl = auto(mock_arm())
        ctl.voice(("jog", "left", 20.0), 1.0)
        ctl.voice("stop", 1.5)
        ctl.step(None, now=1.0 + VOICE_IDLE + 5)
        self.assertEqual(ctl.control_mode, "voice")                       # STOP is never undone by waiting
        self.assertFalse(ctl.live_enabled)

    def test_off_by_default(self):
        ctl = GuidedControl(mock_arm(), movement="follow")
        ctl.voice_available = True
        ctl.voice(("jog", "left", 20.0), 1.0)
        self.assertEqual(ctl.control_mode, "vision")                      # the old way: switch first


class FakeAgent:
    def __init__(self, replies=()):
        self.sent, self.replies = [], list(replies)

    def send(self, **message):
        self.sent.append(message)
        return True

    def poll(self):
        replies, self.replies = self.replies, []
        return replies


def pretend_arm():
    robot = arm_link.VirtualArm()
    robot.hover = 120.0
    robot.start()
    return robot


class AgentInTheAppTest(unittest.TestCase):
    def setUp(self):
        self.robot = pretend_arm()
        self.ctl = auto(self.robot)
        self.to_job = lambda u, v: fetch.FetchJob(self.robot, *arm_link.area_to_point(u, v))

    def frames(self, agent, until, seconds=30):
        end = time.monotonic() + seconds
        while time.monotonic() < end and not until():
            run_agent_messages(agent, self.ctl, self.to_job, time.monotonic())
            self.ctl.step(None, time.monotonic())
            time.sleep(0.01)

    def test_bring_me_the_banana(self):
        ctl, agent = self.ctl, FakeAgent()
        self.assertTrue(ctl.voice(("task", "bring me the banana"), time.monotonic()))
        run_agent_messages(agent, ctl, self.to_job, time.monotonic())
        self.assertEqual(agent.sent, [{"task": "bring me the banana"}])
        self.assertEqual(ctl.control_mode, "voice")                       # the face waits while it works

        agent.replies = [{"confirm": "banana"}]
        run_agent_messages(agent, ctl, self.to_job, time.monotonic())
        self.assertIn("Is it the banana?", ctl.instruction())
        self.assertTrue(ctl.voice("yes", time.monotonic()))
        agent.replies = [{"pick": {"name": "banana", "u": 0.3, "v": 0.6}}]
        run_agent_messages(agent, ctl, self.to_job, time.monotonic())
        self.assertEqual(agent.sent[-1], {"answer": True})
        self.assertEqual(ctl.phase, "fetching")

        self.frames(agent, lambda: {"result": None} in agent.sent)
        self.assertIn({"result": None}, agent.sent)                       # brought: the agent hears so
        self.assertFalse(self.robot.closed)
        agent.replies = [{"reply": "Done: the banana is next to you."}]
        run_agent_messages(agent, ctl, self.to_job, time.monotonic())
        self.assertFalse(ctl.task_running)

    def test_stop_during_the_fetch(self):
        ctl, agent = self.ctl, FakeAgent()
        ctl.voice(("task", "bring me the cup"), time.monotonic())
        agent.replies = [{"pick": {"name": "cup", "u": 0.2, "v": 0.3}}]
        run_agent_messages(agent, ctl, self.to_job, time.monotonic())
        for _ in range(10):
            ctl.step(None, time.monotonic())
        ctl.voice("stop", time.monotonic())
        run_agent_messages(agent, ctl, self.to_job, time.monotonic())
        self.assertIn({"result": "the person said stop"}, agent.sent)
        ctl.step(None, time.monotonic())
        self.assertNotEqual(ctl.phase, "fetching")

    def test_stop_answers_no(self):
        ctl, agent = self.ctl, FakeAgent([{"confirm": "pills"}])
        ctl.voice(("task", "bring me my pills"), time.monotonic())
        run_agent_messages(agent, ctl, self.to_job, time.monotonic())
        ctl.voice("stop", time.monotonic())
        run_agent_messages(agent, ctl, self.to_job, time.monotonic())
        self.assertIn({"answer": False}, agent.sent)

    def test_yes_without_a_question_does_nothing(self):
        self.assertFalse(self.ctl.voice("yes", time.monotonic()))

    def test_out_of_reach_goes_back_to_the_agent(self):
        ctl, agent = self.ctl, FakeAgent()
        ctl.voice(("task", "bring me the cup"), time.monotonic())
        to_far = lambda u, v: fetch.FetchJob(self.robot, 400.0, 0.0)   # noqa: E731
        agent.replies = [{"pick": {"name": "cup", "u": 0.2, "v": 0.3}}]
        run_agent_messages(agent, ctl, to_far, time.monotonic())
        self.assertIn("outside", agent.sent[-1]["result"])


class SonicTaskTest(unittest.TestCase):
    def test_a_named_thing_becomes_a_task(self):
        sonic = SonicInput()
        sonic.read(['{"task": "bring me the banana", "heard": "can you grab me a banana"}\n'])
        (event,) = sonic.poll()
        self.assertEqual(event.action, ("task", "bring me the banana"))


if __name__ == "__main__":
    unittest.main()
