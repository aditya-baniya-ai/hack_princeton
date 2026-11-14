"""Runs without a robot or a HoloLens: the pretend arm from the driver stands in for the real one.
    python3 -m unittest discover -s tests
"""

import sys
import time
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from eyereach import eye_control  # noqa: E402

CENTER, DROP_SPOT = (0.5, 0.5), (0.1, 0.9)


def run(ctl, gaze, until, seconds=12):
    """Feed the same gaze until the phase is `until` (or time runs out)."""
    end = time.monotonic() + seconds
    while ctl.phase != until and time.monotonic() < end:
        ctl.step(gaze)
        time.sleep(0.02)
    return ctl.phase


def new_arm():
    robot = eye_control.arm_link.VirtualArm()
    robot.start()
    return robot, eye_control.EyeControl(robot)


class EyeControlTest(unittest.TestCase):
    def test_grab_carry_and_drop(self):
        robot, ctl = new_arm()
        robot.held = object()                     # the pretend table: something is in the claw
        self.assertEqual(run(ctl, CENTER, "holding"), "holding")
        self.assertTrue(robot.closed)
        self.assertEqual(run(ctl, DROP_SPOT, "open"), "open")
        self.assertFalse(robot.closed)

    def test_empty_grab_is_not_held(self):
        robot, ctl = new_arm()                    # nothing in the claw
        run(ctl, CENTER, "grab")
        self.assertEqual(run(ctl, CENTER, "open"), "open")

    def test_no_grab_before_the_dwell(self):
        robot, ctl = new_arm()
        for _ in range(10):
            ctl.step(CENTER)
            time.sleep(0.02)
        self.assertEqual(ctl.phase, "open")

    def test_lost_gaze_holds_still(self):
        robot, ctl = new_arm()
        moves = []
        robot.follow = lambda u, v: moves.append((u, v))
        for _ in range(10):
            ctl.step(None)
        self.assertEqual((moves, ctl.phase), ([], "open"))


if __name__ == "__main__":
    unittest.main()
