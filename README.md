# EyeReach

A robot arm that a person with severe arm paralysis controls with their eyes: look at an
object, hold the gaze to confirm, and the arm picks it up and puts it in a drop zone.

Built at HackPrinceton Fall 2026 (Nov 13-15, 2026).

## Code

- `eyereach/`, `tests/`: written at the event. Gaze control: gaze point and dwell instead of hand tracking.
- `driver/open_manipulator_x/`: our robot-arm driver, written during the hackathon. EyeReach uses
  `brain/arm.py` and `game/arm_link.py`.

## What is new in this repo

- `eyereach/eye_control.py`: gaze point to table point, dwell to confirm, grab, hold, drop.
  The claw must be where the user is looking before it grabs or drops, and if gaze tracking is
  lost the arm holds still.
- `eyereach/gaze_webcam.py`: webcam eye tracking (MediaPipe iris landmarks), a calibration that maps
  the eyes to screen pixels, and smoothing.
- `eyereach/run_webcam.py`: the screen app. Calibrate on 12 dots, then look at a map of the table
  and the arm follows your gaze. Step one before the HoloLens, which will look at the real table instead.
- `tests/`: gaze maths, the screen drawing and the control loop, with the pretend arm.

## Status

| Part | State |
|---|---|
| Gaze-to-arm logic and the screen app | tested with made-up landmarks and the pretend arm; **not yet tried with a real face or the real arm** |
| Webcam gaze accuracy | unknown until measured on a person |
| HoloLens 2 gaze input | not built yet (next) |
| Uncertainty handling (ask for confirmation when unsure) | not built yet |

## Run

```bash
git clone https://github.com/aditya-baniya-ai/hack_princeton.git
cd hack_princeton
pip install -r requirements.txt
python3 -m unittest discover -s tests
python3 -m eyereach.run_webcam            # pretend arm first
python3 -m eyereach.run_webcam --real     # real arm, on the Jetson
```

This is a research prototype, not a medical device, and it has not been tested with people
who have disabilities.
