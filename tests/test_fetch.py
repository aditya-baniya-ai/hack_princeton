"""brain/fetch.py with the pretend arm: glide, grab, carry, let go.
    python3 -m unittest tests.test_fetch
"""

import math
import sys
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

    def test_missed_grab_says_so(self):
        robot = pretend_arm(holds=False)
        self.assertIn("missed", fetch.fetch(robot, *arm_link.area_to_point(0.5, 0.5)))
        self.assertFalse(robot.closed)

    def test_out_of_reach(self):
        self.assertIn("outside", fetch.fetch(pretend_arm(), 400.0, 0.0))

    def test_to_table(self):
        # a camera that sees x = 100 + 200u, y = 50 - 100v
        matrix = [[200, 0, 100], [0, -100, 50], [0, 0, 1]]
        self.assertEqual(fetch.to_table(matrix, 0.5, 0.5), (200.0, 0.0))


if __name__ == "__main__":
    unittest.main()
