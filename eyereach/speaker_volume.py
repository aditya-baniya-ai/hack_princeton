"""The speaker's volume (the system's default output), through PulseAudio's pactl.

    python3 -m eyereach.speaker_volume          prints the volume
    python3 -m eyereach.speaker_volume 70       sets it to 70 %

The eye app's VOLUME - / VOLUME + buttons use change(); it is the same volume Nova Sonic
(launch/talk.sh) and the spoken prompts play at. Everything returns None when pactl is missing.
"""

import re
import shutil
import subprocess
import sys

STEP = 10            # % per press
SINK = "@DEFAULT_SINK@"


def _pactl(*args):
    if shutil.which("pactl") is None:
        return None
    try:
        return subprocess.run(["pactl", *args], capture_output=True, text=True, timeout=3).stdout
    except (OSError, subprocess.SubprocessError):
        return None


def get_volume():
    """The volume in % (0-100), or None if it can't be read."""
    out = _pactl("get-sink-volume", SINK)
    found = re.search(r"(\d+)%", out or "")
    return int(found.group(1)) if found else None


def set_volume(percent):
    """Set the volume (kept within 0-100; unmutes when above 0). Returns the new volume or None."""
    percent = max(0, min(100, int(round(percent))))
    if _pactl("set-sink-volume", SINK, f"{percent}%") is None:
        return None
    if percent > 0:
        _pactl("set-sink-mute", SINK, "0")
    return get_volume()


def change(delta):
    """Up or down by delta % from where it is now. Returns the new volume or None."""
    now = get_volume()
    return None if now is None else set_volume(now + delta)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        print(f"Volume: {set_volume(float(sys.argv[1]))}%")
    else:
        print(f"Volume: {get_volume()}%")
