"""No network: Nova's answers come from saved responses in tests/fixtures/.
    python3 -m unittest discover -s tests
"""

import json
import sys
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from eyereach import vision_nova  # noqa: E402
from eyereach.vision_nova import Detection, parse_detections, parse_match, to_pixels  # noqa: E402

FIXTURES = Path(__file__).resolve().parent / "fixtures"
IMAGE = np.zeros((480, 640, 3), np.uint8)       # 640 wide, 480 high


def fixture(name):
    return json.loads((FIXTURES / name).read_text())


def nova_says(response=None, error=None):
    """A pretend Bedrock client whose converse() returns response, or raises error."""
    client = mock.Mock()
    client.converse.side_effect = error
    client.converse.return_value = response
    return mock.patch.object(vision_nova, "_client", return_value=client), client


class ScaleTest(unittest.TestCase):
    def test_0_1000_scale_to_pixels(self):
        self.assertEqual(to_pixels([0, 0, 1000, 1000], 640, 480), (0, 0, 640, 480))
        self.assertEqual(to_pixels([500, 250, 750, 500], 640, 480), (320, 120, 480, 240))

    def test_out_of_range_and_swapped_corners(self):
        self.assertEqual(to_pixels([1100, 600, -20, 300], 640, 480), (0, 144, 640, 288))


class ParseTest(unittest.TestCase):
    def test_detect_fixture(self):
        text = vision_nova.response_text(fixture("nova_detect.json"))
        found = parse_detections(text, 640, 480)
        self.assertEqual([d.name for d in found], ["cup", "pill bottle", "phone"])
        self.assertEqual(found[0].box, (77, 192, 198, 346))
        self.assertEqual(found[2].box, (448, 293, 608, 480))
        self.assertIsNone(found[0].confidence)

    def test_pick_fixture(self):
        text = vision_nova.response_text(fixture("nova_pick.json"))
        match, reason = parse_match(text, 640, 480)
        self.assertEqual(match, Detection("pill bottle", (333, 218, 390, 331)))
        self.assertIn("pills", reason)

    def test_bad_entries_are_skipped(self):
        text = '[{"name": "cup", "box": [0, 0, 500, 500]}, {"name": "x"}, {"box": [1, 2, 3]}, "junk"]'
        self.assertEqual([d.name for d in parse_detections(text, 640, 480)], ["cup"])

    def test_not_json(self):
        self.assertEqual(parse_detections("I see a cup.", 640, 480), [])
        self.assertEqual(parse_match("I see a cup.", 640, 480)[0], None)

    def test_unsure_match(self):
        match, reason = parse_match('{"name": null, "box": null, "reason": "two bottles"}', 640, 480)
        self.assertIsNone(match)
        self.assertEqual(reason, "two bottles")


class CallTest(unittest.TestCase):
    def test_detect_objects(self):
        patch, client = nova_says(fixture("nova_detect.json"))
        with patch:
            found = vision_nova.detect_objects(IMAGE, names=["cup", "pills"])
        self.assertEqual(len(found), 3)
        sent = client.converse.call_args.kwargs
        self.assertEqual(client.converse.call_count, 1)                  # one call, no retries
        self.assertIn("only these: cup, pills", sent["messages"][0]["content"][1]["text"])
        self.assertEqual(sent["messages"][0]["content"][0]["image"]["format"], "jpeg")

    def test_pick_match(self):
        patch, _ = nova_says(fixture("nova_pick.json"))
        with patch:
            match, _ = vision_nova.pick_match(IMAGE, "the pills")
        self.assertEqual(match.name, "pill bottle")

    def test_failure_does_not_crash(self):
        patch, _ = nova_says(error=TimeoutError("read timed out"))
        with patch, self.assertLogs(vision_nova.log, "WARNING"):
            self.assertEqual(vision_nova.detect_objects(IMAGE), [])
        patch, _ = nova_says(error=RuntimeError("no credentials"))
        with patch, self.assertLogs(vision_nova.log, "WARNING"):
            self.assertEqual(vision_nova.pick_match(IMAGE, "the cup"), (None, "Nova could not be reached"))


if __name__ == "__main__":
    unittest.main()
