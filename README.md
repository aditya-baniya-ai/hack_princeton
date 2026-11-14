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
- `tests/test_eye_control.py`: runs with the driver's pretend arm.

## Status

| Part | State |
|---|---|
| Gaze-to-arm logic | tested with the pretend arm only (no robot, no HoloLens) |
| HoloLens 2 gaze input | not built yet |
| Gaze-to-table calibration | not built yet |
| Real arm | not tested with this code yet |
| Uncertainty handling (ask for confirmation when unsure) | not built yet |

## Run

```bash
git clone https://github.com/aditya-baniya-ai/hack_princeton.git
python3 -m unittest discover -s tests
```

This is a research prototype, not a medical device, and it has not been tested with people
who have disabilities.
