"""No camera: the test dots, the error maths, the CSV and the grab-trial timer.
    python3 -m unittest discover -s tests
"""

import csv
import sys
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from eyereach.run_webcam import COLUMNS, ROWS  # noqa: E402
from tools import measure  # noqa: E402


class DotsTest(unittest.TestCase):
    def test_nine_dots_inside_the_screen(self):
        spots = measure.test_spots(1280, 720)
        self.assertEqual(len(spots), 9)
        self.assertEqual(spots[0], (256, 180))
        self.assertEqual(spots[4], (640, 360))
        self.assertTrue(all(0 < x < 1280 and 0 < y < 720 for x, y in spots))

    def test_no_test_dot_is_a_calibration_dot(self):
        calibration = {(c, r) for c in COLUMNS for r in ROWS}
        tests = {(c, r) for c in measure.TEST_COLUMNS for r in measure.TEST_ROWS}
        self.assertFalse(calibration & tests)


class ErrorTest(unittest.TestCase):
    def test_dot_error_uses_the_median(self):
        points = [(103, 104)] * 9 + [(900, 900)]          # one wild guess (a half blink) does not count
        self.assertAlmostEqual(measure.dot_error(points, (100, 100)), 5.0)

    def test_too_few_frames_is_not_a_measurement(self):
        self.assertIsNone(measure.dot_error([(100, 100)] * 4, (100, 100)))

    def test_summary(self):
        result = measure.summarize([10.0, 30.0, None, 20.0], 1280, screen_cm=29, distance_cm=55)
        self.assertEqual(result["dots_seen"], 3)
        self.assertEqual(result["mean_error_px"], 20.0)
        self.assertEqual(result["worst_error_px"], 30.0)
        self.assertEqual(result["mean_error_pct"], 1.56)
        self.assertAlmostEqual(result["mean_error_deg"], 0.47, places=2)   # 20 px = 0.45 cm at 55 cm

    def test_no_degrees_without_the_measurements(self):
        self.assertIsNone(measure.summarize([10.0], 1280)["mean_error_deg"])

    def test_nothing_seen(self):
        self.assertIsNone(measure.summarize([None, None], 1280))


class CsvTest(unittest.TestCase):
    def test_header_once_then_rows(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "results" / "m.csv"
            measure.append_rows([{"person": "a", "kind": "grab", "trial": 1}], path)
            measure.append_rows([{"person": "b", "kind": "grab", "trial": 1}], path)
            rows = list(csv.DictReader(path.read_text().splitlines()))
        self.assertEqual([r["person"] for r in rows], ["a", "b"])
        self.assertEqual(list(rows[0]), measure.FIELDS)


class TrialsTest(unittest.TestCase):
    def test_times_each_trial_and_saves_it(self):
        answers = iter(["", "maybe", "y", "", "n", "q"])          # "maybe" is asked again
        ticks = iter([0.0, 12.34, 20.0, 27.5])
        with tempfile.TemporaryDirectory() as d:
            out = Path(d) / "m.csv"
            rows = measure.trials(Namespace(person="jackfin", out=out), ask=lambda prompt: next(answers),
                                  clock=lambda: next(ticks))
            saved = list(csv.DictReader(out.read_text().splitlines()))
        self.assertEqual([(r["trial"], r["seconds"], r["success"]) for r in rows], [(1, 12.3, True), (2, 7.5, False)])
        self.assertEqual([(r["kind"], r["seconds"], r["success"]) for r in saved],
                         [("grab", "12.3", "True"), ("grab", "7.5", "False")])


if __name__ == "__main__":
    unittest.main()
