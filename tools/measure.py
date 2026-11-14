"""Measures how well EyeReach works, for the README and the judges. Rows go to results/measurements.csv.

    python3 -m tools.measure accuracy --person aditya --screen-cm 29 --distance-cm 55
        Uses the saved calibration (or calibrates), then shows 9 test dots placed away from the calibration
        dots, so the error is measured on spots the tracker never learned. Prints the average and worst error
        in pixels, % of the screen width and degrees.

    python3 -m tools.measure trials --person aditya
        A helper times grab trials at the keyboard: Enter when the person starts, then y (it worked) or n.

Screen width (the lit picture, in cm) and eyes-to-screen distance are for the degrees; measure them per setup.
"""

import argparse
import csv
import math
import statistics
import time
from datetime import datetime
from pathlib import Path

from eyereach.session_log import degrees

OUT = Path(__file__).resolve().parent.parent / "results" / "measurements.csv"
FIELDS = ["date", "person", "kind", "trial", "seconds", "success", "mean_error_px", "worst_error_px",
          "mean_error_pct", "worst_error_pct", "mean_error_deg", "worst_error_deg", "pointer", "screen"]
TEST_COLUMNS, TEST_ROWS = (0.2, 0.5, 0.8), (0.25, 0.5, 0.75)   # none on a calibration dot (run_webcam COLUMNS)
SETTLE = 0.8                                                    # seconds at a dot before the eyes count
LOOK = 1.5                                                      # seconds of looking that count, per dot


def test_spots(width, height):
    """The 9 test dots in pixels, row by row."""
    return [(int(c * width), int(r * height)) for r in TEST_ROWS for c in TEST_COLUMNS]


def dot_error(points, spot):
    """Pixels between a dot and where the tracker put the gaze (the median of its guesses), or None."""
    if len(points) < 5:
        return None
    x, y = statistics.median(p[0] for p in points), statistics.median(p[1] for p in points)
    return math.hypot(x - spot[0], y - spot[1])


def summarize(errors, screen_px, screen_cm=None, distance_cm=None):
    """Average and worst error over the dots that were seen, in px, % of the width and degrees."""
    seen = [e for e in errors if e is not None]
    if not seen:
        return None
    mean, worst = statistics.fmean(seen), max(seen)
    rnd = lambda v: None if v is None else round(v, 2)             # noqa: E731
    return {"mean_error_px": round(mean, 1), "worst_error_px": round(worst, 1),
            "mean_error_pct": round(100 * mean / screen_px, 2), "worst_error_pct": round(100 * worst / screen_px, 2),
            "mean_error_deg": rnd(degrees(mean, screen_px, screen_cm, distance_cm)),
            "worst_error_deg": rnd(degrees(worst, screen_px, screen_cm, distance_cm)),
            "dots_seen": len(seen)}


def append_rows(rows, path=OUT):
    """Add rows to the CSV, writing the header first if the file is new."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    new = not path.exists()
    with path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
        if new:
            writer.writeheader()
        writer.writerows(rows)


def now():
    return datetime.now().strftime("%Y-%m-%d %H:%M")


# ---- accuracy: needs the camera and a screen ----------------------------------------------------------

def show_dots(camera, gaze, gmap, shown):
    """Each test dot in turn; returns the error at each (None where the face was not seen), or None on Esc."""
    import cv2
    from eyereach import run_webcam as app
    errors = []
    for n, spot in enumerate(test_spots(app.W, app.H)):
        began, points = time.monotonic(), []
        while time.monotonic() - began < SETTLE + LOOK:
            frame, f = app.look(camera, gaze)
            if f is not None and time.monotonic() - began > SETTLE:
                points.append(tuple(gmap.predict(f)[0]))
            canvas = app.blank()
            cv2.circle(canvas, spot, int(app.scale(18)), app.GOLD, -1, cv2.LINE_AA)
            cv2.circle(canvas, spot, int(app.scale(4)), app.BG, -1, cv2.LINE_AA)
            app.centered(canvas, f"Test dot {n + 1} of 9: look at the centre", app.W // 2, int(0.06 * app.H), 0.7)
            shown(canvas)
            if cv2.waitKey(1) & 0xFF == 27:
                return None
        errors.append(dot_error(points, spot))
    return errors


def accuracy(args):
    import cv2
    from eyereach import run_webcam as app
    from eyereach.gaze_webcam import Camera, GazeMap, IrisGaze

    size = None if args.windowed else app.screen_size()
    if size:
        app.set_size(*size)
    camera, gaze = Camera(args.camera), IrisGaze(args.pointer)
    cv2.namedWindow(app.WINDOW, cv2.WINDOW_NORMAL | cv2.WINDOW_FREERATIO)
    cv2.resizeWindow(app.WINDOW, app.W, app.H)
    if not args.windowed:
        cv2.setWindowProperty(app.WINDOW, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    shown = lambda canvas: cv2.imshow(app.WINDOW, canvas)        # noqa: E731
    try:
        saved = app.saved_path(args.pointer)
        gmap = None if args.recalibrate or not saved.exists() else GazeMap.load(saved)
        if gmap is None:
            done = app.calibrate(camera, gaze, shown, app.Voice(on=False))
            if done is None:
                return
            gmap = done[0]
            gmap.save(saved)
        errors = show_dots(camera, gaze, gmap, shown)
    finally:
        cv2.destroyAllWindows()
    if errors is None:
        print("Stopped (Esc).")
        return
    result = summarize(errors, app.W, args.screen_cm, args.distance_cm)
    if result is None:
        print("The face was not seen at any test dot: is it in view and well lit?")
        return
    deg = lambda v: "" if v is None else f", {v:.1f} deg"           # noqa: E731
    print(f"Accuracy on 9 test dots ({result['dots_seen']} seen), {args.pointer} pointer, {app.W}x{app.H}:")
    print(f"  average error {result['mean_error_px']:.0f} px ({result['mean_error_pct']:.1f}% of the width"
          f"{deg(result['mean_error_deg'])})")
    print(f"  worst error   {result['worst_error_px']:.0f} px ({result['worst_error_pct']:.1f}% of the width"
          f"{deg(result['worst_error_deg'])})")
    print("  each dot (px): " + " ".join("--" if e is None else f"{e:.0f}" for e in errors))
    append_rows([{"date": now(), "person": args.person, "kind": "accuracy", "pointer": args.pointer,
                  "screen": f"{app.W}x{app.H}", **result}], args.out)
    print(f"Added to {args.out}")


# ---- grab trials: a helper at the keyboard ----------------------------------------------------------------

def trials(args, ask=input, clock=time.monotonic):
    """Time grab trials until the helper types q. Returns the rows it added."""
    rows = []
    print(f"Timing grab trials for {args.person}. Enter when they start, q to finish.")
    while True:
        if ask(f"\nTrial {len(rows) + 1}: Enter to start, q to finish: ").strip().lower() == "q":
            break
        started = clock()
        answer = ""
        while answer not in ("y", "n"):
            answer = ask("  Done? y = it worked, n = it failed: ").strip().lower()
        seconds = round(clock() - started, 1)
        rows.append({"date": now(), "person": args.person, "kind": "grab", "trial": len(rows) + 1,
                     "seconds": seconds, "success": answer == "y"})
        print(f"  {seconds} s, {'worked' if answer == 'y' else 'failed'}")
    if rows:
        append_rows(rows, args.out)
        worked = [r["seconds"] for r in rows if r["success"]]
        print(f"\n{len(worked)} of {len(rows)} worked" +
              (f", average {statistics.fmean(worked):.1f} s when it worked" if worked else "") + f". Added to {args.out}")
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description="Measure gaze accuracy and grab times.")
    parser.add_argument("mode", choices=("accuracy", "trials"))
    parser.add_argument("--person", default="anonymous")
    parser.add_argument("--out", type=Path, default=OUT, help=f"the CSV to add to (default {OUT})")
    parser.add_argument("--screen-cm", type=float, help="width of the lit screen picture in cm (for degrees)")
    parser.add_argument("--distance-cm", type=float, help="eyes to screen in cm (for degrees)")
    parser.add_argument("--pointer", choices=("head", "eyes", "both"), default="head")
    parser.add_argument("--camera", default="0")
    parser.add_argument("--recalibrate", action="store_true", help="calibrate even if a saved calibration exists")
    parser.add_argument("--windowed", action="store_true", help="a 1280 x 720 window instead of full screen")
    args = parser.parse_args(argv)
    accuracy(args) if args.mode == "accuracy" else trials(args)


if __name__ == "__main__":
    main()
