"""Nearby table-camera sampling and fitting, without a camera or moving hardware."""
import json
import math
import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import Mock, patch

import cv2
import numpy as np

from tests.test_fetch import fetch, arm_link


class TableCalibrationTest(unittest.TestCase):
    def test_ten_unique_nearby_reachable_points(self):
        spots = fetch.CALIBRATION_SPOTS
        self.assertEqual(len(spots), 10)
        self.assertEqual(len(set(spots)), 10)
        xy = np.array([arm_link.area_to_point(*s) for s in spots])
        self.assertLess(np.linalg.norm(np.ptp(xy, axis=0)), 65)  # whole patch diagonal < 6.5 cm
        self.assertTrue(all(math.dist(a, b) < 35 for a, b in zip(xy, xy[1:])))
        self.assertEqual(np.linalg.matrix_rank(xy - xy.mean(axis=0)), 2)
        self.assertTrue(all(0 <= u <= 1 and 0 <= v <= 1 for u, v in spots))

    def calibrate(self, directory, stop=False, duplicate=False):
        table = np.array([arm_link.area_to_point(*s) for s in fetch.CALIBRATION_SPOTS])
        # Synthetic camera: x=100+200u, y=-100+200v.
        pixels = np.rint((table - (100, -100)) / 200 * (640, 480)).astype(int)
        capture = Mock()
        capture.read.return_value = (True, np.zeros((480, 640, 3), np.uint8))
        robot, handler = Mock(), {}
        i = 0

        def wait_key(_):
            nonlocal i
            if stop:
                return 27
            x, y = pixels[0 if duplicate else i]
            handler['click'](cv2.EVENT_LBUTTONDOWN, int(x), int(y), 0, None)
            i += 1
            return -1

        path = Path(directory) / 'table_camera.json'
        with ExitStack() as stack:
            stack.enter_context(patch.object(fetch, 'CALIBRATION', path))
            glide = stack.enter_context(patch.object(fetch, 'glide', return_value=None))
            stack.enter_context(patch.object(fetch.arm, 'forward', side_effect=[(*p, 120, 0) for p in table]))
            stack.enter_context(patch.object(cv2, 'VideoCapture', return_value=capture))
            stack.enter_context(patch.object(cv2, 'namedWindow'))
            stack.enter_context(patch.object(cv2, 'setMouseCallback', side_effect=lambda _, cb: handler.update(click=cb)))
            stack.enter_context(patch.object(cv2, 'imshow'))
            stack.enter_context(patch.object(cv2, 'waitKey', side_effect=wait_key))
            cleanup = stack.enter_context(patch.object(cv2, 'destroyAllWindows'))
            why = fetch.calibrate(robot, '/dev/video0')
        capture.release.assert_called_once()
        cleanup.assert_called_once()
        return why, path, table, pixels, glide

    def test_all_ten_clicks_fit_and_save_a_usable_mapping(self):
        with tempfile.TemporaryDirectory() as directory:
            why, path, table, pixels, glide = self.calibrate(directory)
            self.assertIsNone(why)
            self.assertEqual(glide.call_count, 10)
            saved = json.loads(path.read_text())
            self.assertEqual(saved['camera'], '/dev/video0')
            for pixel, point in zip(pixels, table):
                self.assertLess(math.dist(fetch.to_table(saved['matrix'], *(pixel / (640, 480))), point), 1)

    def test_cancel_and_degenerate_clicks_preserve_previous_calibration(self):
        for options in ({'stop': True}, {'duplicate': True}):
            with self.subTest(options=options), tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / 'table_camera.json'
                path.write_text('previous calibration')
                why, *_ = self.calibrate(directory, **options)
                self.assertIsNotNone(why)
                self.assertEqual(path.read_text(), 'previous calibration')
