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


class HeadTest(unittest.TestCase):
    def face(self, yaw_deg=0.0, pitch_deg=0.0):
        """3D key points of a simple face (pixels, z smaller = nearer), turned by yaw then pitch."""
        from eyereach import gaze_webcam as g
        base = {g.LEFT["outer"]: (45, 0, 25), g.RIGHT["outer"]: (-45, 0, 25), g.FOREHEAD: (0, -70, 15),
                g.CHIN: (0, 75, 15), g.NOSE: (0, 30, 0)}
        y, p = np.radians(yaw_deg), np.radians(pitch_deg)
        turn = np.array([[np.cos(y), 0, -np.sin(y)], [0, 1, 0], [np.sin(y), 0, np.cos(y)]])
        nod = np.array([[1, 0, 0], [0, np.cos(p), np.sin(p)], [0, -np.sin(p), np.cos(p)]])
        pts = np.zeros((478, 3))
        for i, xyz in base.items():
            pts[i] = nod @ turn @ np.array(xyz, float) + (320, 240, 0)
        return pts

    def test_head_direction_follows_the_face(self):
        from eyereach.gaze_webcam import head_pose
        straight = head_pose(self.face(), (640, 480))
        self.assertAlmostEqual(np.degrees(straight[0]), 0, delta=1)
        self.assertAlmostEqual(np.degrees(straight[1]), 0, delta=1)
        right = head_pose(self.face(yaw_deg=20), (640, 480))
        self.assertAlmostEqual(np.degrees(right[0]), 20, delta=2)
        up = head_pose(self.face(pitch_deg=15), (640, 480))
        self.assertAlmostEqual(np.degrees(up[1]), 15, delta=2)

    def test_one_euro_is_steady_when_still_and_follows_a_move(self):
        from eyereach.gaze_webcam import OneEuro
        f, rng, t = OneEuro(), np.random.default_rng(0), 0.0
        outs = []
        for _ in range(60):                                    # holding still, with 15 px of jitter
            t += 0.05
            outs.append(f((500 + rng.normal(0, 15), 400 + rng.normal(0, 15)), t))
        self.assertLess(np.std(np.array(outs[20:]), axis=0).max(), 6)
        for _ in range(30):                                    # a deliberate move 600 px away
            t += 0.05
            out = f((1100, 400), t)
        self.assertLess(abs(out[0] - 1100), 30)


class DrawTest(unittest.TestCase):
    def test_draws_a_full_frame_with_the_pretend_arm(self):
        robot = eye_control.arm_link.VirtualArm()
        robot.start()
        ctl = eye_control.GuidedControl(robot)
        for tracking, paused in ((True, False), (False, False), (True, True)):
            face = (np.zeros((120, 160, 3), np.uint8), (80, 60))
            image = run_webcam.draw(ctl, (640, 360), tracking, paused, 30.0, face)
            self.assertEqual(image.shape, (run_webcam.H, run_webcam.W, 3))
        self.assertEqual(run_webcam.to_area((run_webcam.PLAY[0], run_webcam.PLAY[1])), (0.0, 0.0))
        self.assertIsNone(run_webcam.to_area((0, 0)))
        self.assertEqual(run_webcam.on_screen((-20, 800)), (0.0, run_webcam.H - 1.0))   # just outside: pinned
        self.assertIsNone(run_webcam.on_screen((3 * run_webcam.W, 300)))                # far outside: looking away
        self.assertEqual(run_webcam.on_screen((640, 360)), (640, 360))

    def test_buttons_and_map_are_found_where_they_are_drawn(self):
        robot = eye_control.arm_link.VirtualArm()
        robot.start()
        ctl = eye_control.GuidedControl(robot)
        for (name, _), (left, top, right, bottom) in zip(ctl.buttons(), run_webcam.button_rects(len(ctl.buttons()))):
            self.assertEqual(run_webcam.looking_at(ctl, ((left + right) / 2, (top + bottom) / 2)), ("button", name))
        left, top, right, bottom = run_webcam.PLAY
        self.assertEqual(run_webcam.looking_at(ctl, ((left + right) / 2, (top + bottom) / 2))[0], "map")
        self.assertIsNone(run_webcam.looking_at(ctl, (5, 5)))

    def test_calibration_dots_go_like_a_snake(self):
        spots = run_webcam.dot_path()
        self.assertEqual(len(spots), 12)
        steps = [abs(a[0] - b[0]) + abs(a[1] - b[1]) for a, b in zip(spots, spots[1:])]
        self.assertLess(max(steps), run_webcam.W * 0.45)   # never across the whole screen


if __name__ == "__main__":
    unittest.main()


class MouthLandmarksTest(unittest.TestCase):
    def test_opening_is_invariant_to_scale_translation_and_roll(self):
        from eyereach.gaze_webcam import mouth_opening
        pts = np.zeros((478, 2))
        pts[61], pts[291] = (-50, 0), (50, 0)
        pts[13], pts[14] = (0, -20), (0, 20)
        self.assertAlmostEqual(mouth_opening(pts), 0.4)
        angle = 0.7
        rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
        self.assertAlmostEqual(mouth_opening(pts @ rotation * 3 + (300, 200)), 0.4)
        pts[14] = pts[13]
        self.assertEqual(mouth_opening(pts), 0)
        pts[291] = pts[61]
        self.assertIsNone(mouth_opening(pts))
