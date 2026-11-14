"""Records how a session went, one JSON line per event, in ~/.eyereach/sessions/.

    python3 -m eyereach.session_log            sums up every session so far (per person)

Recorded: each calibration (how long it took, the error in pixels, % of the screen width and, when the
screen width and the eyes-to-screen distance are given, in degrees; also the error at each dot), every
move (by looking), pick-ups (by blink or button, and whether something was held),
let-gos, STOPs, FIX MY AIMs, and each finished pick-and-place with its time.
"""

import json
import math
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path

FOLDER = Path.home() / ".eyereach" / "sessions"


def degrees(pixels, screen_px, screen_cm, distance_cm):
    """Pixels on the screen -> degrees of gaze angle, for eyes distance_cm from the screen."""
    if not (screen_cm and distance_cm):
        return None
    return math.degrees(math.atan2(pixels * screen_cm / screen_px, distance_cm))


class SessionLog:
    def __init__(self, person="anonymous", screen_px=None, screen_cm=None, distance_cm=None, arm="", folder=FOLDER):
        folder.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.path = folder / f"{stamp}_{person}.jsonl"
        self.screen = (screen_px, screen_cm, distance_cm)
        self.started = time.monotonic()
        self.file = self.path.open("a")
        self.event("start", person=person, arm=arm, screen_px=screen_px, screen_cm=screen_cm,
                   distance_cm=distance_cm, date=stamp)

    def event(self, name, **details):
        line = {"t": round(time.monotonic() - self.started, 2), "event": name, **details}
        self.file.write(json.dumps(line) + "\n")
        self.file.flush()

    def calibration(self, seconds, error_px, dot_errors_px):
        px, cm, dist = self.screen
        self.event("calibration", seconds=round(seconds, 1), error_px=round(error_px, 1),
                   error_pct=round(100 * error_px / px, 2) if px else None,
                   error_deg=_round(degrees(error_px, px, cm, dist)),
                   dot_errors_px=[round(e) for e in dot_errors_px])

    def close(self):
        self.event("end")
        self.file.close()


def _round(x, digits=2):
    return None if x is None else round(x, digits)


def summary(folder=FOLDER):
    """Per person: sessions, calibrations, their time and error, tasks and their time, pick-ups, STOPs."""
    people = {}
    for path in sorted(folder.glob("*.jsonl")):
        lines = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        if not lines:
            continue
        p = people.setdefault(lines[0].get("person", "?"), dict(
            sessions=0, cal_s=[], cal_pct=[], cal_deg=[], tasks=[], picks=0, held=0, blinks=0, stops=0, aims=0))
        p["sessions"] += 1
        for e in lines:
            kind = e["event"]
            if kind == "calibration":
                p["cal_s"].append(e["seconds"])
                if e.get("error_pct") is not None:
                    p["cal_pct"].append(e["error_pct"])
                if e.get("error_deg") is not None:
                    p["cal_deg"].append(e["error_deg"])
            elif kind == "task done":
                p["tasks"].append(e["seconds"])
            elif kind == "pick":
                p["picks"] += 1
                p["blinks"] += e.get("by") == "blink"
            elif kind == "picked":
                p["held"] += bool(e.get("holding"))
            elif kind == "stop":
                p["stops"] += 1
            elif kind == "fix aim":
                p["aims"] += 1
    return people


def describe(values, unit):
    if not values:
        return "-"
    if len(values) == 1:
        return f"{values[0]:.1f} {unit}"
    return f"median {statistics.median(values):.1f} {unit} (range {min(values):.1f}-{max(values):.1f}, n={len(values)})"


def main(folder=FOLDER):
    people = summary(folder)
    if not people:
        print(f"No sessions yet in {folder}")
        return
    for person, p in people.items():
        print(f"{person}: {p['sessions']} session(s)")
        print(f"  calibration time   {describe(p['cal_s'], 's')}")
        print(f"  calibration error  {describe(p['cal_pct'], '% of width')}")
        print(f"                     {describe(p['cal_deg'], 'degrees')}")
        print(f"  pick-and-place     {describe(p['tasks'], 's')}")
        print(f"  pick-ups           {p['picks']} tried ({p['blinks']} by blinking), {p['held']} held something")
        print(f"  STOP pressed       {p['stops']}    FIX MY AIM pressed {p['aims']}")


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else FOLDER)
