"""Which microphone to listen on: the webcam's first, the speaker's as the fallback.

Used by both voice programs (the eye app's offline commands and launch/talk.sh with Nova Sonic).

    python3 -m eyereach.audio_devices        shows the microphones and which one would be used

Measured on the Jetson (the same speech played through the speaker, both microphones recording): the C920
webcam heard it twice as loud with word confidence 0.95, the USB speaker's microphone 0.87, so voice
commands were often dropped on the speaker's. So: the webcam when it is there (through PulseAudio, made
the default source here); otherwise a USB microphone that is not a camera (a speaker or speakerphone), if it
opens at 16 kHz. PulseAudio doesn't list the speaker's microphone (its card only offers output profiles), so
that one is opened directly through PortAudio.

EYEREACH_MIC overrides it: a device number or part of its name (EYEREACH_MIC=PnP forces the speaker's).
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
                if any(w in name.lower() for w in CAMERA_WORDS):
                    say(f"Microphone: webcam (system default), as EYEREACH_MIC={override} asks")
                    return None                        # the webcam goes through PulseAudio
                say(f"Microphone: {name} (EYEREACH_MIC={override})")
                return index
        say(f"Microphone: EYEREACH_MIC={override} not found; using the system default")
        return None
    if any(any(w in name.lower() for w in CAMERA_WORDS) for _, name in mics) and webcam_as_default():
        say("Microphone: the webcam (system default)")
        return None
    for index, name in mics:
        if is_speaker_mic(name) and opens(index, rate):
            say(f"Microphone: the speaker's ({name}), device {index}, as the webcam's isn't available")
            return index
    say("Microphone: system default; no webcam or speaker microphone found")
    return None


def webcam_as_default():
    """Make the webcam's microphone PulseAudio's default source (the default can drift back to the
    Jetson's empty headset input). True if the webcam is the default now."""
    import shutil
    import subprocess
    if shutil.which("pactl") is None:
        return False
    try:
        sources = subprocess.run(["pactl", "list", "short", "sources"], capture_output=True, text=True,
                                 timeout=3).stdout.splitlines()
        webcam = next((line.split("\t")[1] for line in sources if "\t" in line and ".monitor" not in line
                       and any(w in line.lower() for w in CAMERA_WORDS)), None)
        if webcam is None:
            return False
        subprocess.run(["pactl", "set-default-source", webcam], capture_output=True, timeout=3)
        return True
    except (OSError, subprocess.SubprocessError):
        return False


if __name__ == "__main__":
    for index, name in microphones():
        print(f"  {index:3d}  {name}{'   <- speaker mic' if is_speaker_mic(name) else ''}")
    pick_microphone()
