"""Tells the caregiver what happened, through the Photon service (services/photon/), and says if they sent STOP.

    from eyereach.notify import notify, stopped
    notify("picked", "Picked up the pill bottle.")     # returns at once; never blocks or crashes the arm loop
    if stopped(): ...                                   # the caregiver texted STOP (RESUME clears it)

If the service isn't running the message is just dropped (and printed once), so the arm keeps working.

Environment variables:
    EYEREACH_NOTIFY_URL   default http://127.0.0.1:8787/notify
"""

import json
import os
import threading
import urllib.request
from pathlib import Path

URL = os.environ.get("EYEREACH_NOTIFY_URL", "http://127.0.0.1:8787/notify")
FOLDER = Path.home() / ".eyereach"
STOP_FILE = FOLDER / "STOP"                    # written by the service when the caregiver texts STOP
TIMEOUT = 2.0                                  # seconds; the arm loop never waits for this
_warned = False


def _post(event, text, url):
    global _warned
    try:
        body = json.dumps({"event": event, "text": text}).encode()
        request = urllib.request.Request(url, body, {"Content-Type": "application/json"})
        urllib.request.urlopen(request, timeout=TIMEOUT).close()
    except Exception as e:                     # service down, timeout, anything: never reach the arm loop
        if not _warned:
            print(f"(caregiver messages are off: {e})")
            _warned = True


def notify(event, text, url=None):
    """Send `text` to the caregiver in the background. Returns the thread (tests wait on it)."""
    thread = threading.Thread(target=_post, args=(str(event), str(text), url or URL), name="notify")
    thread.start()                             # not a daemon: a short script still sends before it exits
    return thread


def stopped():
    """True while the caregiver's STOP is in force."""
    return STOP_FILE.exists()
