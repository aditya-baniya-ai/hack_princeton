"""brain/fetch.py with the pretend arm: glide, grab, carry, let go.
    python3 -m unittest tests.test_fetch
"""

import math
import sys
import time
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "brain"))
import arm  # noqa: E402
import arm_link  # noqa: E402
import fetch  # noqa: E402


def pretend_arm(holds=True):
    robot = arm_link.VirtualArm()
    robot.hover = 120.0
    robot.start()
    robot.held = True if holds else None
    return robot


class FetchTest(unittest.TestCase):
    def test_brings_it_to_the_person(self):
        robot = pretend_arm()
        x, y = arm_link.area_to_point(0.3, 0.6)
        self.assertIsNone(fetch.fetch(robot, x, y))
        gx, gy, gz, _ = arm.forward(robot.joints())
        dx, dy = arm_link.area_to_point(*fetch.DROP_AREA)
        self.assertLess(math.hypot(gx - dx, gy - dy), 3)        # over the drop spot
        self.assertFalse(robot.closed)                            # and let go
        self.assertAlmostEqual(gz, 120.0, delta=5)                # back up at the travel height

    def test_the_grip_guess_is_not_trusted(self):
        # on the real arm the gripper said "nothing" even when it held the object: carry it anyway
        robot = pretend_arm(holds=False)
        self.assertIsNone(fetch.fetch(robot, *arm_link.area_to_point(0.5, 0.5)))

    def test_stop_ends_it(self):
        robot = pretend_arm()
        job = fetch.FetchJob(robot, *arm_link.area_to_point(0.3, 0.6))
        for _ in range(400):                       # until it is down at the object
            robot.update()
            if job.step() or (job.stage and job.stage[1] == "grab" and job.started):
                break
            time.sleep(0.01)
        robot.halt()                                 # STOP
        while not job.step():
            robot.update()
        self.assertEqual(job.why, "stopped")
        self.assertFalse(robot.busy())

    def test_stop_while_gliding(self):
        robot = pretend_arm()
        job = fetch.FetchJob(robot, *arm_link.area_to_point(0.1, 0.1))
        for _ in range(5):
            robot.update()
            job.step()
        robot.halt()
        self.assertTrue(job.step())
        self.assertEqual(job.why, "stopped")

    def test_out_of_reach(self):
        self.assertIn("outside", fetch.fetch(pretend_arm(), 400.0, 0.0))

    def test_to_table(self):
        # a camera that sees x = 100 + 200u, y = 50 - 100v
        matrix = [[200, 0, 100], [0, -100, 50], [0, 0, 1]]
        self.assertEqual(fetch.to_table(matrix, 0.5, 0.5), (200.0, 0.0))


if __name__ == "__main__":
    unittest.main()
