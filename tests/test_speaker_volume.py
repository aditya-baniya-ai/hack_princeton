"""Speaker volume: the helper (with a pretend pactl) and the QUIETER / LOUDER buttons."""

import sys
import time
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from eyereach import eye_control, speaker_volume  # noqa: E402


class FakePactl:
    def __init__(self, volume):
        self.volume, self.muted = volume, True

    def __call__(self, *args):
        if args[0] == "get-sink-volume":
            return f"Volume: front-left: 1 / {self.volume}% / -1 dB,   front-right: 1 / {self.volume}% / -1 dB"
        if args[0] == "set-sink-volume":
            self.volume = int(args[2].rstrip("%"))
        if args[0] == "set-sink-mute":
            self.muted = args[2] == "1"
        return ""


class SpeakerVolumeTest(unittest.TestCase):
    def test_change_stays_between_0_and_100_and_unmutes(self):
        fake = FakePactl(95)
        with patch.object(speaker_volume, "_pactl", fake):
            self.assertEqual(speaker_volume.change(10), 100)
            self.assertFalse(fake.muted)
            self.assertEqual(speaker_volume.set_volume(-20), 0)

    def test_no_pactl_means_none(self):
        with patch.object(speaker_volume, "_pactl", lambda *a: None):
            self.assertIsNone(speaker_volume.get_volume())
            self.assertIsNone(speaker_volume.change(10))


class VolumeButtonTest(unittest.TestCase):
    def test_dwell_steps_and_keeps_stepping_while_held(self):
        robot = eye_control.arm_link.VirtualArm()
        robot.start()
        ctl = eye_control.GuidedControl(robot, look=0.2, movement="follow")
        steps, t = [], 0.0
        for _ in range(60):                           # look at LOUDER for 1.2 s
            t += 0.02
            ctl.step(("button", "volume up"), now=t)
            if ctl.wants_volume:
                steps.append(ctl.wants_volume)
                ctl.wants_volume = 0
        self.assertGreaterEqual(len(steps), 3)        # about one step per dwell, not just one
        self.assertTrue(all(s == eye_control.VOLUME_STEP for s in steps))

    def test_works_even_while_picking_up(self):
        robot = eye_control.arm_link.VirtualArm()
        robot.start()
        ctl = eye_control.GuidedControl(robot, movement="follow")
        ctl.phase = "picking"
        self.assertTrue(ctl.can("volume down"))
        ctl.press("volume down", time.monotonic(), how="click")
        self.assertEqual(ctl.wants_volume, -eye_control.VOLUME_STEP)


class VolumeScreenTest(unittest.TestCase):
    def test_buttons_are_found_where_they_are_drawn(self):
        from eyereach import run_webcam
        robot = eye_control.arm_link.VirtualArm()
        robot.start()
        ctl = eye_control.GuidedControl(robot, movement="follow")
        ctl.volume = 40
        for name, _, (left, top, right, bottom) in run_webcam.volume_rects():
            self.assertEqual(run_webcam.looking_at(ctl, ((left + right) / 2, (top + bottom) / 2)), ("button", name))
        image = run_webcam.draw(ctl, (100, 100), True, False, 20.0)
        self.assertEqual(image.shape, (run_webcam.H, run_webcam.W, 3))


if __name__ == "__main__":
    unittest.main()
