"""Which microphone to listen on: the speaker's own microphone first, the webcam's as the fallback.

Used by both voice programs (the eye app's offline commands and launch/talk.sh with Nova Sonic).

    python3 -m eyereach.audio_devices        shows the microphones and which one would be used

The rule: a USB microphone that is not a camera (a speaker or speakerphone, like the "USB PnP Audio Device")
wins, if it opens at 16 kHz. Otherwise None: the system default, which is set to the webcam (pactl
set-default-source). PulseAudio may not list a speaker's microphone at all (its card only offers output
profiles), so this asks PortAudio, which sees the hardware directly.

EYEREACH_MIC overrides it: a device number or part of its name (EYEREACH_MIC=C920 forces the webcam).
"""

import os

CAMERA_WORDS = ("webcam", "camera", "c920", "c922", "c930", "brio", "uvc")
SKIP_WORDS = ("ape", "hda", "hdmi", "pulse", "default", "sysdefault", "dmix", "dsnoop")


def microphones():
    """[(index, name)] of every input device PortAudio sees."""
    import sounddevice as sd
    return [(i, d["name"]) for i, d in enumerate(sd.query_devices()) if d["max_input_channels"] > 0]


def opens(index, rate=16000):
    """Can this device record 16-bit mono at this rate?"""
    import sounddevice as sd
    try:
        sd.check_input_settings(device=index, samplerate=rate, channels=1, dtype="int16")
        return True
    except Exception:
        return False


def is_speaker_mic(name):
    """A USB microphone that isn't a camera's."""
    lowered = name.lower()
    return ("usb" in lowered and not any(w in lowered for w in CAMERA_WORDS)
            and not any(lowered.startswith(w) or f" {w}" in lowered for w in SKIP_WORDS))


def pick_microphone(override=None, rate=16000, say=print):
    """The device to open: an index, or None for the system default (the webcam). Says which and why."""
    override = override if override is not None else os.environ.get("EYEREACH_MIC")
    try:
        mics = microphones()
    except Exception as error:                         # no sounddevice / PortAudio: the default it is
        say(f"Microphone: system default ({error})")
        return None
    if override:
        for index, name in mics:
            if str(override) == str(index) or str(override).lower() in name.lower():
                if "c920" in name.lower() or "webcam" in name.lower():
                    say(f"Microphone: webcam (system default), as EYEREACH_MIC={override} asks")
                    return None                        # the webcam goes through PulseAudio
                say(f"Microphone: {name} (EYEREACH_MIC={override})")
                return index
        say(f"Microphone: EYEREACH_MIC={override} not found; using the system default")
        return None
    for index, name in mics:
        if is_speaker_mic(name) and opens(index, rate):
            say(f"Microphone: the speaker's ({name}), device {index}")
            return index
    say("Microphone: the webcam (system default); the speaker microphone is missing or busy")
    return None


if __name__ == "__main__":
    for index, name in microphones():
        print(f"  {index:3d}  {name}{'   <- speaker mic' if is_speaker_mic(name) else ''}")
    pick_microphone()
