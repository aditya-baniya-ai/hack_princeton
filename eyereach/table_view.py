"""What the app draws on the robot camera's picture: the blue PLAYGROUND circle on the table (every point in it is
within the arm's reach), and Nova's boxes around the objects it finds there (green: in the playground, the arm can
grab it; yellow: the arm can reach it; red: out of reach,
too near or too far), each with a dot where the arm would go.

Both need the table camera's calibration (launch/eyereach.sh --agent calibrate --camera /dev/video0), which says
which spot of the picture is which spot of the table. Nova needs the AWS keys; it is asked every few seconds in
the background, so the app never waits for it.
"""

import json
import math
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "brain"))
import arm_link  # noqa: E402

CALIBRATION = Path.home() / ".eyereach" / "table_camera.json"     # written by brain/fetch.py calibrate
# The PLAYGROUND circle on the table, in mm from the arm (x straight ahead, y to the arm's left): the biggest circle
# inside the arm's reach (arm_link.NEAR to FAR from the base, SIDE either way), 1 cm in from its edges.
MARGIN_MM = 10.0
PLAYGROUND_RADIUS_MM = (arm_link.FAR - arm_link.NEAR) / 2 - MARGIN_MM                        # 60 mm
PLAYGROUND_MM = (arm_link.arm.BASE_X + (arm_link.FAR + arm_link.NEAR) / 2, 0.0, PLAYGROUND_RADIUS_MM)  # x, y, r
DETECT_EVERY = 4.0                                 # seconds between Nova's looks (each takes about 4 s)
BLUE, GREEN, YELLOW, RED, WHITE = (255, 120, 0), (60, 220, 60), (0, 210, 255), (60, 60, 255), (255, 255, 255)


def load_calibration(path=CALIBRATION):
    """The picture (0-1 across and down) -> table (mm) matrix, or None before the calibration."""
    try:
        return numpy.array(json.loads(Path(path).read_text())["matrix"], dtype=float)
    except (OSError, ValueError, KeyError, TypeError):
        return None


def to_table(matrix, u, v):
    x, y, w = matrix @ (u, v, 1.0)
    return x / w, y / w


def to_picture(matrix, x, y):
    u, v, w = numpy.linalg.inv(matrix) @ (x, y, 1.0)
    return u / w, v / w


def in_zone(x, y, zone=PLAYGROUND_MM):
    return math.hypot(x - zone[0], y - zone[1]) <= zone[2]


def reachable(x, y):
    u, v = arm_link.point_to_area(x, y)
    return 0.0 <= u <= 1.0 and 0.0 <= v <= 1.0


def zone_outline(matrix, zone=PLAYGROUND_MM, points=72):
    """The circle's edge in the picture (0-1): an oval there, seen at an angle."""
    x, y, r = zone
    return [to_picture(matrix, x + r * math.cos(a), y + r * math.sin(a))
            for a in numpy.linspace(0, 2 * math.pi, points, endpoint=False)]


def verdict(matrix, u, v):
    """(colour, words) for an object whose centre is at u, v of the picture."""
    x, y = to_table(matrix, u, v)
    if not (math.isfinite(x) and math.isfinite(y)):
        return RED, "off the table"
    if in_zone(x, y):
        return GREEN, "can grab"
    if reachable(x, y):
        return YELLOW, "reachable"
    return RED, "out of reach"


class TableBoxes:
    """Asks Nova what is on the table every DETECT_EVERY seconds, in the background."""

    def __init__(self, camera, detect=None, every=DETECT_EVERY):
        if detect is None:
            from eyereach.vision_nova import detect_objects as detect
        self.camera, self.detect, self.every = camera, detect, every
        self.found, self.size, self.at, self.status, self.stopped = [], None, 0.0, "Nova: starting", False
        threading.Thread(target=self._run, name="eyereach-table-boxes", daemon=True).start()

    def _run(self):
        while not self.stopped:
            frame = self.camera.frame
            if frame is None:
                time.sleep(0.5)
                continue
            started = time.monotonic()
            try:
                found = self.detect(frame)
            except Exception as error:                     # no network, no keys: keep the app going
                self.status = f"Nova: {type(error).__name__}"
                found = None
            if found is not None:
                self.found, self.size, self.at = list(found), frame.shape[:2], time.monotonic()
                self.status = f"Nova: {len(self.found)} found" if self.found else "Nova: nothing found"
            time.sleep(max(0.5, self.every - (time.monotonic() - started)))

    def close(self):
        self.stopped = True


def draw_table(frame, matrix, boxes=None):
    """A copy of the robot camera's picture with the PLAYGROUND circle and Nova's boxes on it."""
    if frame is None:
        return None
    picture = frame.copy()
    height, width = picture.shape[:2]
    if matrix is None:
        cv2.putText(picture, "Calibrate the table camera to see where to put things", (10, height - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 1, cv2.LINE_AA)
    else:
        corners = numpy.int32([(u * width, v * height) for u, v in zone_outline(matrix)])
        shade = picture.copy()
        cv2.fillPoly(shade, [corners], BLUE)
        cv2.addWeighted(shade, 0.18, picture, 0.82, 0, picture)
        cv2.polylines(picture, [corners], True, BLUE, 3, cv2.LINE_AA)
        top = corners[corners[:, 1].argmin()]
        cv2.putText(picture, "PLAYGROUND", (int(corners[:, 0].mean()) - 52, int(top[1]) - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, BLUE, 2, cv2.LINE_AA)
    if boxes is not None and boxes.found and boxes.size:
        sh, sw = boxes.size
        for d in boxes.found:
            x0, y0, x1, y1 = (int(c) for c in (d.box[0] * width / sw, d.box[1] * height / sh,
                                                d.box[2] * width / sw, d.box[3] * height / sh))
            u, v = (x0 + x1) / 2 / width, (y0 + y1) / 2 / height
            colour, words = verdict(matrix, u, v) if matrix is not None else (WHITE, "")
            cv2.rectangle(picture, (x0, y0), (x1, y1), colour, 2, cv2.LINE_AA)
            cv2.circle(picture, (int(u * width), int(v * height)), 5, colour, -1, cv2.LINE_AA)
            label = f"{d.name}: {words}" if words else d.name
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            ty = y0 - 6 if y0 > th + 10 else y1 + th + 6
            cv2.rectangle(picture, (x0, ty - th - 4), (x0 + tw + 6, ty + 4), (0, 0, 0), -1)
            cv2.putText(picture, label, (x0 + 3, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 1, cv2.LINE_AA)
    if boxes is not None:
        cv2.putText(picture, boxes.status, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, WHITE, 2, cv2.LINE_AA)
    return picture
