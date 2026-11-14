"""EyeReach with a webcam: look at the table map on the screen and the arm follows your gaze.

    python3 -m eyereach.run_webcam              pretend arm: try this first
    python3 -m eyereach.run_webcam --real       the real arm (on the Jetson; hold the arm first)

Look at the dots to calibrate, then look inside the map. Hold your gaze on the claw to grab, move
it, and hold your gaze on the drop zone (the pink corner) to let go.
Keys: Esc close, C calibrate again, Space pause (the arm holds still).
"""

import argparse
import time
from pathlib import Path

import cv2
import numpy as np

from eyereach import eye_control
from eyereach.gaze_webcam import Camera, GazeMap, IrisGaze, Smooth

W, H = 1280, 720
WINDOW = "EyeReach"
PLAY = (100, 110, 1180, 650)                    # the table map on screen: left, top, right, bottom
SAVED = Path.home() / ".eyereach" / "calibration.json"
COLUMNS, ROWS = (0.10, 0.37, 0.63, 0.90), (0.12, 0.50, 0.88)
DIM, CYAN, PINK, GOLD, GREEN = (150, 140, 140), (230, 220, 90), (180, 110, 255), (90, 200, 255), (120, 220, 120)


def to_area(point):
    """Screen pixel -> (u, v) on the table map, or None outside the map."""
    left, top, right, bottom = PLAY
    u, v = (point[0] - left) / (right - left), (point[1] - top) / (bottom - top)
    return (u, v) if 0 <= u <= 1 and 0 <= v <= 1 else None


def to_screen(u, v):
    left, top, right, bottom = PLAY
    return int(left + u * (right - left)), int(top + v * (bottom - top))


def text(canvas, s, x, y, size=0.6, color=DIM, thick=1):
    cv2.putText(canvas, s, (x, y), cv2.FONT_HERSHEY_SIMPLEX, size, color, thick, cv2.LINE_AA)


def draw(ctl, point, tracking, paused, fps):
    canvas = np.full((H, W, 3), (34, 26, 26), np.uint8)
    left, top, right, bottom = PLAY
    cv2.rectangle(canvas, (left, top), (right, bottom), (150, 110, 130), 2)
    text(canvas, "toward you", (left + right) // 2 - 50, top - 12)
    text(canvas, "toward Robo", (left + right) // 2 - 55, bottom + 28)
    u0, v0, u1, v1 = eye_control.DROP
    cv2.rectangle(canvas, to_screen(u0, v0), to_screen(u1, v1), PINK, 2)
    text(canvas, "DROP", to_screen(u0, v0)[0] + 8, to_screen(u0, v0)[1] + 24, 0.6, PINK)
    robot = ctl.robot
    if robot.command is not None:                      # the claw: a ring, pink while closed
        u, v, _ = robot.gripper_area()
        claw = to_screen(min(1, max(0, u)), min(1, max(0, v)))
        cv2.circle(canvas, claw, 22, PINK if robot.closed else CYAN, 2, cv2.LINE_AA)
    if point is not None and not paused:               # the gaze, with a ring that fills while it rests
        p = (int(point[0]), int(point[1]))
        cv2.circle(canvas, p, 8, GOLD, -1, cv2.LINE_AA)
        if ctl.rest:
            done = min(1.0, (time.monotonic() - ctl.rest[2]) / ctl.dwell)
            cv2.ellipse(canvas, p, (20, 20), 0, -90, -90 + 360 * done, GREEN, 3, cv2.LINE_AA)
    state = "PAUSED" if paused else ("tracking" if tracking else "NO FACE / BLINK")
    text(canvas, "EYEREACH", 30, 50, 1.2, GOLD, 2)
    text(canvas, f"{robot.name}  |  {ctl.phase.upper()}  |  {state}  |  {fps:.0f} fps", 30, 85)
    text(canvas, "Esc close   C calibrate   Space pause", 30, H - 20)
    return canvas


def calibrate(camera, gaze, shown):
    """Show dots and learn the gaze map. Returns (map, error in pixels) or None if cancelled."""
    X, Y = [], []
    for row in ROWS:
        for column in COLUMNS:
            target, start = (int(column * W), int(row * H)), time.monotonic()
            while (age := time.monotonic() - start) < 1.9:
                canvas = np.full((H, W, 3), (34, 26, 26), np.uint8)
                cv2.circle(canvas, target, int(26 - 14 * min(age, 1.0)), GOLD, -1, cv2.LINE_AA)
                text(canvas, "Look at the dot. Keep your head still.", 30, 50, 0.8)
                shown(canvas)
                frame = camera.read()
                f = None if frame is None else gaze.features(frame)
                if age > 0.7 and f is not None:                # give the eyes time to settle
                    X.append(f), Y.append(target)
                if cv2.waitKey(1) & 0xFF == 27:
                    return None
    if len(X) < 100:
        print(f"Only {len(X)} good frames: is your face in view and well lit?")
        return None
    gmap = GazeMap.fit(X, Y)
    error = float(np.sqrt(np.mean(np.sum((gmap.predict(X) - np.asarray(Y)) ** 2, axis=1))))
    return gmap, error


def main():
    parser = argparse.ArgumentParser(description="Control the arm with your eyes, using a webcam.")
    parser.add_argument("--real", action="store_true", help="the real arm (default: a pretend arm)")
    parser.add_argument("--camera", default="0", help="camera number or path (default 0; Jetson /dev/video0)")
    parser.add_argument("--reuse", action="store_true", help="skip calibration, use the saved one")
    parser.add_argument("--dwell", type=float, default=eye_control.DWELL, help="seconds of steady gaze to confirm")
    parser.add_argument("--fullscreen", action="store_true")
    args, ros_args = parser.parse_known_args()

    camera, gaze = Camera(args.camera), IrisGaze()
    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW, W, H)
    if args.fullscreen:
        cv2.setWindowProperty(WINDOW, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    show = lambda canvas: cv2.imshow(WINDOW, canvas)       # noqa: E731
    robot = None
    try:
        gmap = GazeMap.load(SAVED) if args.reuse and SAVED.exists() else None
        while gmap is None:
            done = calibrate(camera, gaze, show)
            if done is None:
                return
            gmap, error = done
            print(f"Calibration done. Average error on the dots: {error:.0f} px "
                  f"({100 * error / W:.1f}% of the screen width; it looks better here than it will in use)")
            gmap.save(SAVED)
        robot = eye_control.arm_link.RealArm(ros_args=ros_args) if args.real else eye_control.arm_link.VirtualArm()
        why = robot.start()
        if why:
            print(f"Can't start the arm: {why}")
            return
        if not args.real:
            robot.held = True        # the pretend arm: every grab holds something, so carrying and dropping show
        ctl, smooth, paused, last = eye_control.EyeControl(robot, dwell=args.dwell), Smooth(), False, time.monotonic()
        while True:
            frame = camera.read()
            f = None if frame is None else gaze.features(frame)
            point = smooth(None if f is None else tuple(gmap.predict(f)[0]))
            ctl.step(None if paused or point is None else to_area(point))
            now = time.monotonic()
            show(draw(ctl, point, f is not None, paused, 1 / max(now - last, 1e-3)))
            last = now
            key = cv2.waitKey(1) & 0xFF
            if key == 27 or cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                break
            if key == ord(" "):
                paused = not paused
            elif key in (ord("c"), ord("C")):
                done = calibrate(camera, gaze, show)
                if done:
                    gmap = done[0]
                    gmap.save(SAVED)
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        gaze.close()
        camera.close()
        if robot is not None:
            if robot.closed:
                robot.open_here()
            robot.stop()


if __name__ == "__main__":
    main()
