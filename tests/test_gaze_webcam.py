"""No camera needed: the gaze maths on made-up landmarks, and the screen drawing.
    python3 -m unittest discover -s tests
"""

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from eyereach import eye_control, run_webcam  # noqa: E402
from eyereach.gaze_webcam import LEFT, NOSE, RIGHT, GazeMap, Smooth, features_from_landmarks  # noqa: E402


def face(iris_shift=(0.0, 0.0), opening=14.0):
    """A flat made-up face, 640 x 480: eyes 60 px wide, iris shifted by iris_shift eye widths."""
    pts = np.zeros((478, 2))
    pts[RIGHT["outer"]], pts[RIGHT["inner"]] = (200, 200), (260, 200)
    pts[LEFT["inner"]], pts[LEFT["outer"]] = (340, 200), (400, 200)
    for eye, x in ((RIGHT, 230), (LEFT, 370)):
        pts[eye["upper"]], pts[eye["lower"]] = (x, 200 - opening / 2), (x, 200 + opening / 2)
        pts[eye["iris"]] = (x + 60 * iris_shift[0], 200 + 60 * iris_shift[1])
    pts[NOSE] = (300, 250)
    return pts


class FeaturesTest(unittest.TestCase):
    def test_iris_offset_is_in_eye_widths(self):
        f = features_from_landmarks(face((0.1, 0.05)), (640, 480))
        self.assertAlmostEqual(f[0], 0.1, places=6)
        self.assertAlmostEqual(f[1], 0.05, places=6)

    def test_centered_iris_is_zero(self):
        self.assertTrue(np.allclose(features_from_landmarks(face(), (640, 480))[:2], 0))

    def test_blink_is_skipped(self):
        self.assertIsNone(features_from_landmarks(face(opening=3.0), (640, 480)))


class MapTest(unittest.TestCase):
    def data(self, n=300, seed=1):
        rng = np.random.default_rng(seed)
        f = np.column_stack([rng.uniform(-0.2, 0.2, n), rng.uniform(-0.1, 0.1, n), rng.uniform(0.2, 0.3, n),
                             rng.uniform(-0.1, 0.1, (n, 2)), rng.uniform(0.4, 0.6, (n, 2))])
        xy = np.column_stack([640 + 2800 * f[:, 0] + 3000 * f[:, 0] ** 2, 360 + 3600 * f[:, 1] + 300 * f[:, 3]])
        return f, xy

    def test_learns_a_known_mapping(self):
        f, xy = self.data()
        test_f, test_xy = self.data(100, seed=2)
        error = np.abs(GazeMap.fit(f, xy).predict(test_f) - test_xy).max()
        self.assertLess(error, 15)                    # pixels, on points it never saw

    def test_save_and_load(self):
        f, xy = self.data()
        gmap = GazeMap.fit(f, xy)
        with tempfile.TemporaryDirectory() as d:
            gmap.save(Path(d) / "c.json")
            self.assertTrue(np.allclose(GazeMap.load(Path(d) / "c.json").predict(f), gmap.predict(f)))


class SmoothTest(unittest.TestCase):
    def test_averages_noise_but_follows_a_jump(self):
        s = Smooth(frames=4, jump=100)
        for p in ((100, 100), (104, 96), (96, 104)):
            out = s(p)
        self.assertLess(np.hypot(out[0] - 100, out[1] - 100), 5)
        self.assertEqual(s((600, 600)), (600, 600))
        self.assertIsNone(s(None))


class DrawTest(unittest.TestCase):
    def test_draws_a_full_frame_with_the_pretend_arm(self):
        robot = eye_control.arm_link.VirtualArm()
        robot.start()
        image = run_webcam.draw(eye_control.EyeControl(robot), (640, 360), True, False, 30.0)
        self.assertEqual(image.shape, (run_webcam.H, run_webcam.W, 3))
        self.assertEqual(run_webcam.to_area((run_webcam.PLAY[0], run_webcam.PLAY[1])), (0.0, 0.0))
        self.assertIsNone(run_webcam.to_area((0, 0)))


if __name__ == "__main__":
    unittest.main()
