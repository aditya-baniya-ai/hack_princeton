"""Which microphone the voice programs pick (pretend device lists, no audio hardware)."""

import unittest
from unittest.mock import patch

from eyereach import audio_devices as ad

WEBCAM, SPEAKER = (24, "HD Pro Webcam C920: USB Audio (hw:2,0)"), (25, "USB PnP Audio Device: Audio (hw:3,0)")


def pick(mics, webcam_ok=True, opens=True, override=None):
    with patch.object(ad, "microphones", lambda: mics), patch.object(ad, "opens", lambda i, r=16000: opens), \
         patch.object(ad, "webcam_as_default", lambda: webcam_ok):
        return ad.pick_microphone(override=override, say=lambda s: None)


class PickMicrophoneTest(unittest.TestCase):
    def test_webcam_first(self):
        self.assertIsNone(pick([WEBCAM, SPEAKER]))                 # None = the system default, set to the webcam

    def test_speaker_when_there_is_no_webcam(self):
        self.assertEqual(pick([SPEAKER]), 25)
        self.assertEqual(pick([WEBCAM, SPEAKER], webcam_ok=False), 25)

    def test_nothing_usable_means_the_default(self):
        self.assertIsNone(pick([SPEAKER], opens=False))

    def test_override_picks_the_speaker(self):
        self.assertEqual(pick([WEBCAM, SPEAKER], override="PnP"), 25)


if __name__ == "__main__":
    unittest.main()
