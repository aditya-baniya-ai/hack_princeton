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
CALIBRATION_SPOTS = [(0.2, 0.2), (0.8, 0.2), (0.8, 0.8), (0.2, 0.8)]   # play-area spots the arm visits
DROP_AREA = (0.5, 0.05)        # play-area spot next to the person (v = 0 is the reach toward them)
GRAB_MM = 50.0                 # same defaults as launch/eyereach.sh
TRAVEL_MM = 120.0


def to_table(matrix, u, v):
    """A spot in the photo (0-1 across and down) -> a point on the table (x, y mm), with the calibration."""
    x, y, w = (row[0] * u + row[1] * v + row[2] for row in matrix)
    return x / w, y / w


def wait(robot, seconds=20.0):
    """Let a grab or release finish -> None, or what went wrong."""
    end = time.monotonic() + seconds
    while robot.busy():
        robot.update()
        if robot.error or time.monotonic() > end:
            robot.halt()
            return robot.error or "the arm took too long"
        time.sleep(0.02)
    return robot.error


def glide(robot, x, y, seconds=15.0):
    """Slide at the travel height until the gripper is over (x, y) -> None, or what went wrong."""
    u, v = arm_link.point_to_area(x, y)
    if not (0 <= u <= 1 and 0 <= v <= 1):
        return f"that spot ({x:.0f}, {y:.0f} mm) is outside the arm's reach"
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        robot.update()
        if robot.error:
            return robot.error
        robot.follow(u, v)
        arrived = math.hypot(robot.aim[0] - x, robot.aim[1] - y) < 1.0 if robot.aim else False
        if arrived and max(abs(c - j) for c, j in zip(robot.command, robot.joints())) < 0.03:
            return None
        time.sleep(0.02)
    robot.halt()
    return "the arm could not get there in time"


def fetch(robot, x, y, grab_mm=GRAB_MM):
    """Grab what is at (x, y) and put it down next to the person -> None, or what went wrong."""
    why = glide(robot, x, y)
    if why:
        return why
    if not robot.grab(grab_mm):
        return robot.error or "the arm could not reach down there"
    why = wait(robot)
    if why:
        return why
    if not robot.holding():
        robot.release_here()
        wait(robot)
        return "missed: the gripper closed on nothing"
    robot.carrying(True)
    why = glide(robot, *arm_link.area_to_point(*DROP_AREA)) or \
        (None if robot.release(grab_mm) else robot.error or "could not lower to let go") or wait(robot)
    robot.carrying(False)
    return why


def calibrate(robot, camera):
    """Visit four spots; at each, you click the table right under the fingertips in the camera view."""
    import cv2
    capture = cv2.VideoCapture(int(camera) if str(camera).isdigit() else camera)
    clicks, photo, table = [], [], []
    cv2.namedWindow("table camera")
    cv2.setMouseCallback("table camera", lambda event, px, py, *_: event == cv2.EVENT_LBUTTONDOWN and
                         clicks.append((px, py)))
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
            cv2.putText(frame, f"{n}/4: click the table right under the fingertips (Esc stops)", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            cv2.imshow("table camera", frame)
            robot.update()
            if cv2.waitKey(20) == 27:
                return "stopped"
        height, width = frame.shape[:2]
        photo.append((clicks[0][0] / width, clicks[0][1] / height))
        table.append((x, y))
    import numpy
    matrix = cv2.getPerspectiveTransform(numpy.float32(photo), numpy.float32(table))
    CALIBRATION.parent.mkdir(parents=True, exist_ok=True)
    CALIBRATION.write_text(json.dumps({"camera": str(camera), "matrix": matrix.tolist()}))
    print(f"Saved {CALIBRATION}")
    capture.release()
    cv2.destroyAllWindows()
    return None


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
        if args.virtual:
            robot.held = True                               # the pretend gripper always catches something
        try:
            why = calibrate(robot, args.camera) if args.action == "calibrate" else fetch(robot, x, y, grab_mm)
        finally:
            robot.stop()
    print(why or "Done: put it down next to the person.")
    sys.exit(1 if why else 0)


if __name__ == "__main__":
    main()
