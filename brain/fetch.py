"""Fetch one object for the task agent: glide over it, grab it, carry it to the person, let go.

The agent runs in Python 3.12 (deepagents) and ROS needs the system Python 3.10, so the agent runs this
as its own program, with the arm already started (launch/eyereach.sh --agent does both):

    python3 brain/fetch.py calibrate --camera /dev/video2     once: where the table camera sees the arm
    python3 brain/fetch.py go 0.42 0.60                       fetch what is at 0.42 across, 0.60 down the photo
    add --virtual to either for the pretend arm (no ROS)

"go" takes the spot as a fraction of the table camera's photo (what Nova gives) and turns it into a point
on the table with the calibration. It exits 0 when the object was put down next to the person;
otherwise the last line it prints says why.
"""

import argparse
import json
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import arm  # noqa: E402
import arm_link  # noqa: E402

CALIBRATION = Path.home() / ".eyereach" / "table_camera.json"
# Ten nearby, non-collinear samples: about 45 x 44 mm around the middle of the reach.
# Start at the centre and visit neighbouring points instead of sweeping the whole table.
CALIBRATION_SPOTS = [
    (0.50, 0.50), (0.45, 0.50), (0.45, 0.35), (0.50, 0.35), (0.55, 0.35),
    (0.55, 0.50), (0.55, 0.65), (0.50, 0.65), (0.45, 0.65), (0.475, 0.425),
]
DROP_AREA = (0.5, 0.05)        # play-area spot next to the person (v = 0 is the reach toward them)
GRAB_MM = 50.0                 # same defaults as launch/eyereach.sh
TRAVEL_MM = 120.0


def to_table(matrix, u, v):
    """A spot in the photo (0-1 across and down) -> a point on the table (x, y mm), with the calibration."""
    x, y, w = (row[0] * u + row[1] * v + row[2] for row in matrix)
    return x / w, y / w


class FetchJob:
    """Grab what is at (x, y) and put it down next to the person, one step() per frame, so the eye app's loop
    (camera, buttons, STOP) keeps running. The caller calls robot.update() every frame; robot.halt() (STOP)
    just ends it: the next step() reports it. step() -> False while working, True when finished; then
    self.why is None (put down next to the person) or what went wrong."""

    GLIDE_SECONDS, ACTION_SECONDS = 15.0, 20.0

    def __init__(self, robot, x, y, grab_mm=GRAB_MM, drop=DROP_AREA, drop_mm=None):
        self.robot, self.grab_mm, self.why = robot, grab_mm, None
        self.drop_mm = grab_mm if drop_mm is None else drop_mm
        self.plan = [("glide", (x, y)), ("down", "grab"), ("glide", arm_link.area_to_point(*drop)), ("down", "release")]
        self.stage, self.deadline, self.started = None, 0.0, False
        robot.stage = "Fetching"                          # robot.halt() (STOP) changes it to "Stopped"
        u, v = arm_link.point_to_area(x, y)
        if not (0 <= u <= 1 and 0 <= v <= 1):
            self.why = f"that spot ({x:.0f}, {y:.0f} mm) is outside the arm's reach"

    def finish(self, why=None):
        self.why, self.plan, self.stage = why, [], None
        self.robot.carrying(False)
        return True

    def step(self):
        r = self.robot
        if self.why:
            return True
        if r.error:
            return self.finish(r.error)
        if r.stage == "Stopped":
            return self.finish("stopped")
        now = time.monotonic()
        if self.stage is None:                            # start the next part of the plan
            if not self.plan:
                return True
            self.stage = self.plan.pop(0)
            self.deadline = now + (self.GLIDE_SECONDS if self.stage[0] == "glide" else self.ACTION_SECONDS)
            self.started = False
        kind, what = self.stage
        if now > self.deadline:
            r.halt()
            return self.finish("the arm could not get there in time" if kind == "glide" else "the arm took too long")
        if kind == "glide":
            x, y = what
            u, v = arm_link.point_to_area(x, y)
            r.follow(u, v)
            arrived = r.aim is not None and math.hypot(r.aim[0] - x, r.aim[1] - y) < 1.0
            if arrived and max(abs(c - j) for c, j in zip(r.command, r.joints())) < 0.03:
                self.stage = None
        elif not self.started:
            if r.busy():
                return False
            ok = r.grab(self.grab_mm) if what == "grab" else r.release(self.drop_mm)
            if not ok:
                return self.finish(r.error or "the arm could not reach down there")
            self.started = True
        elif not r.busy():
            # the gripper's guess at whether it holds something is wrong on the real arm, so it isn't trusted
            r.carrying(what == "grab")
            self.stage = None
        return not self.plan and self.stage is None


def fetch(robot, x, y, grab_mm=GRAB_MM):
    """FetchJob to the end, on its own (brain/fetch.py go) -> None, or what went wrong."""
    job = FetchJob(robot, x, y, grab_mm)
    while not job.step():
        robot.update()
        time.sleep(0.02)
    return job.why


def glide(robot, x, y, seconds=15.0):
    """Slide at the travel height until the gripper is over (x, y) -> None, or what went wrong."""
    job = FetchJob(robot, x, y)
    job.plan = job.plan[:1]
    job.GLIDE_SECONDS = seconds
    while not job.step():
        robot.update()
        time.sleep(0.02)
    return job.why


def calibrate(robot, camera):
    """Visit ten nearby spots; click the table right under the fingertips at each."""
    import cv2
    import numpy

    capture = cv2.VideoCapture(int(camera) if str(camera).isdigit() else camera)
    try:
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)     # same picture dimensions as the eye app
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        clicks, photo, table = [], [], []
        cv2.namedWindow("table camera")
        cv2.setMouseCallback("table camera", lambda event, px, py, *_: event == cv2.EVENT_LBUTTONDOWN and
                             clicks.append((px, py)))
        total = len(CALIBRATION_SPOTS)
        for n, spot in enumerate(CALIBRATION_SPOTS, 1):
            why = glide(robot, *arm_link.area_to_point(*spot))
            if why:
                return why
            x, y, _, _ = arm.forward(robot.joints())
            clicks.clear()
            while not clicks:
                ok, frame = capture.read()
                if not ok:
                    return f"no picture from camera {camera}"
                cv2.putText(frame, f"{n}/{total}: click the table under the fingertips (Esc stops)", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                cv2.imshow("table camera", frame)
                robot.update()
                if cv2.waitKey(20) == 27:
                    return "stopped"
            height, width = frame.shape[:2]
            photo.append((clicks[0][0] / width, clicks[0][1] / height))
            table.append((x, y))
        # Least-squares homography uses every correspondence; getPerspectiveTransform takes only four.
        matrix, _ = cv2.findHomography(numpy.asarray(photo, dtype=float), numpy.asarray(table, dtype=float), method=0)
        if matrix is None or not numpy.isfinite(matrix).all() or numpy.linalg.matrix_rank(matrix) < 3:
            return "calibration could not be fitted: click distinct table spots and try again"
        CALIBRATION.parent.mkdir(parents=True, exist_ok=True)
        CALIBRATION.write_text(json.dumps({"camera": str(camera), "matrix": matrix.tolist()}))
        print(f"Saved {CALIBRATION} ({total} nearby points)")
        return None
    finally:
        capture.release()
        cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description="Fetch one object for the task agent.")
    parser.add_argument("action", choices=["go", "calibrate"])
    parser.add_argument("u", type=float, nargs="?", help="go: 0-1 across the table camera's photo")
    parser.add_argument("v", type=float, nargs="?", help="go: 0-1 down the photo")
    parser.add_argument("--camera", default="0", help="calibrate: the table camera (/dev/video2 or 2)")
    parser.add_argument("--virtual", action="store_true", help="the pretend arm, no ROS")
    parser.add_argument("--grab-mm", type=float, default=GRAB_MM)
    parser.add_argument("--travel-mm", type=float, default=TRAVEL_MM)
    args = parser.parse_args()

    if args.action == "go":
        if args.u is None or args.v is None:
            sys.exit("go needs the spot: go U V")
        if not CALIBRATION.exists() and not args.virtual:
            sys.exit(f"no table camera calibration yet ({CALIBRATION}): run calibrate first")
        # the pretend arm without a calibration: the photo is the play area itself
        matrix = json.loads(CALIBRATION.read_text())["matrix"] if CALIBRATION.exists() else None
        x, y = to_table(matrix, args.u, args.v) if matrix else arm_link.area_to_point(args.u, args.v)

    robot = arm_link.VirtualArm() if args.virtual else arm_link.RealArm()
    grab_mm = max(args.grab_mm, 30.0)                       # never closer to the table than 3 cm (like the app)
    robot.hover = max(args.travel_mm, grab_mm + 10)
    why = robot.start()
    if not why:
        try:
            why = calibrate(robot, args.camera) if args.action == "calibrate" else fetch(robot, x, y, grab_mm)
        finally:
            robot.stop()
    print(why or "Done: put it down next to the person.")
    sys.exit(1 if why else 0)


if __name__ == "__main__":
    main()
