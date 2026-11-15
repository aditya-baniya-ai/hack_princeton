import sys
import unittest
from pathlib import Path

import numpy

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from eyereach import table_view as tv  # noqa: E402
from eyereach.vision_nova import Detection  # noqa: E402

# a made-up calibration: the picture (0-1) spans x 60-220 mm and y -150..150 mm
MATRIX = numpy.array([[0.0, 160.0, 60.0], [300.0, 0.0, -150.0], [0.0, 0.0, 1.0]])


class TableViewTest(unittest.TestCase):
    def test_every_point_of_the_playground_is_within_reach_with_room_to_spare(self):
        cx, cy, r = tv.PLAYGROUND_MM
        self.assertGreaterEqual(r, 50)                                   # at least 10 cm across
        for angle in numpy.linspace(0, 2 * numpy.pi, 72):
            for d in numpy.linspace(0, r + tv.MARGIN_MM - 1, 8):         # check the remaining margin beyond the blue edge
                x, y = cx + d * numpy.cos(angle), cy + d * numpy.sin(angle)
                self.assertTrue(tv.reachable(x, y), (x, y))

    def test_to_picture_undoes_to_table(self):
        u, v = tv.to_picture(MATRIX, *tv.to_table(MATRIX, 0.3, 0.7))
        self.assertAlmostEqual(u, 0.3)
        self.assertAlmostEqual(v, 0.7)

    def test_colours_say_where_a_thing_is(self):
        self.assertEqual(tv.verdict(MATRIX, 0.5, 0.65)[1], "can grab")     # (164, 0) mm: the middle
        self.assertEqual(tv.verdict(MATRIX, 0.5, 0.25)[1], "can grab")     # (100, 0): inside the enlarged edge
        self.assertEqual(tv.verdict(MATRIX, 0.5, 0.2125)[1], "reachable") # (94, 0): just outside it
        self.assertEqual(tv.verdict(MATRIX, 0.5, 0.1)[1], "out of reach")  # (76, 0): too near the base
        self.assertEqual(tv.verdict(MATRIX, 0.0, 1.0)[1], "out of reach")  # (220, -150): too far

    def test_draws_without_calibration_or_nova(self):
        frame = numpy.zeros((480, 640, 3), numpy.uint8)
        self.assertEqual(tv.draw_table(frame, None).shape, frame.shape)
        self.assertIsNone(tv.draw_table(None, MATRIX))

    def test_draws_nova_boxes_without_touching_the_camera_picture(self):
        frame = numpy.zeros((480, 640, 3), numpy.uint8)

        class Boxes:
            found, size, status = [Detection("orange", (300, 200, 340, 240))], (480, 640), "Nova: 1 found"
        drawn = tv.draw_table(frame, MATRIX, Boxes())
        self.assertGreater(drawn.sum(), 0)
        self.assertEqual(frame.sum(), 0)

    def test_the_watcher_asks_nova_in_the_background(self):
        import time

        class Cam:
            frame = numpy.zeros((480, 640, 3), numpy.uint8)
        boxes = tv.TableBoxes(Cam(), detect=lambda f: [Detection("cup", (1, 2, 3, 4))], every=0.1)
        deadline = time.monotonic() + 3
        while not boxes.found and time.monotonic() < deadline:
            time.sleep(0.05)
        boxes.close()
        self.assertEqual(boxes.found[0].name, "cup")
        self.assertEqual(boxes.status, "Nova: 1 found")


if __name__ == "__main__":
    unittest.main()
