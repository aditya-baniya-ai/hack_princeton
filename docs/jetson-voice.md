# Jetson: hands-free voice and vision test branch

This branch adds offline microphone commands to the existing face/eye controls.
It uses the current one-camera setup plus a microphone (the webcam microphone
can work). It does not identify objects: “grab the item” acts where the gripper
is now. Existing defaults remain 120 mm travel and 50 mm pickup.

## 1. Get the test branch

From your existing repository on the Jetson:

```bash
git fetch origin
git switch --track origin/test/voice-vision-control
```

If that local branch already exists, use `git switch test/voice-vision-control`
then `git pull --ff-only` instead. The existing ROS 2 Humble arm setup and
`~/eyereach_venv` from the README are prerequisites.

## 2. Install optional speech support once

Use a terminal on the Jetson desktop, with a microphone connected and selected
as the system input. Remote Desktop audio input is not necessarily forwarded;
a USB microphone plugged directly into the Jetson is preferable for this test.

```bash
bash launch/setup_voice.sh
```

This installs PortAudio, `vosk==0.3.45`, and `sounddevice` in the EyeReach Python
environment. It downloads the small English model from Alpha Cephei into
`~/.eyereach/models/vosk-model-small-en-us-0.15`. Internet is required for setup;
recognition then runs locally without an API key. No microphone audio is uploaded
or saved. Recognized command text and whether it was accepted are recorded in
the existing session logs.

If your environment is elsewhere:

```bash
EYEREACH_PYTHON=/path/to/your/venv/bin/python bash launch/setup_voice.sh
```

Upstream references: [Vosk installation](https://alphacephei.com/vosk/install),
[streaming microphone example](https://github.com/alphacep/vosk-api/blob/master/python/example/test_microphone.py).

## 3. Test the microphone without connecting to any robot

```bash
~/eyereach_venv/bin/python -m eyereach.voice_commands --list-devices
~/eyereach_venv/bin/python -m eyereach.voice_commands
```

Say “Robot move left”, “Robot grab the item”, and “Stop”. The terminal should
print the recognized command and its action; this tool never connects to ROS or
moves the arm. Ctrl+C ends this diagnostic. If needed, select a microphone using
its device index or a distinctive substring of its name:

```bash
~/eyereach_venv/bin/python -m eyereach.voice_commands --device 2
```

## 4. Test the full application with the pretend arm

```bash
launch/eyereach.sh --no-arm --voice
# To choose an input explicitly:
launch/eyereach.sh --no-arm --voice --microphone 2
```

The initial vision calibration is unchanged and starts automatically. Afterwards,
the screen shows VISION or VOICE, microphone status, and the last recognized
command. Voice-model loading happens in a background thread; wait for Listening.

Start in VISION: point to move as before. Voice pickup, release, STOP, and mode
switching work alongside vision. Blink controls remain available in VISION.

To move using speech:

1. Say **“Robot use voice”**, or look at **USE VOICE** until it fills.
2. Say **“Robot resume”**, or look at **MOVE AGAIN**. If STOP interrupted a pickup
   below travel height, this first lifts the arm back to its travel height.
3. Say **“Robot move left”**. It moves 50 mm, then waits.
4. Say **“Robot grab the item”**. It opens, lowers, closes, then lifts.
5. Move using another direction command. Say **“Robot release the item”** to open
   at that location.
6. Say **“Robot use vision”**, then **“Robot resume”**, to return to face/eye movement.

Every switch first stops movement and clears the old target. In VOICE mode, gaze
movement and blink pickup/release are disabled so they cannot conflict with
speech. The gaze buttons, including STOP and USE VISION, remain usable.
Look away from a changing button before activating its replacement: holding your
gaze on STOP will not automatically press MOVE AGAIN, and USE VOICE will not
automatically switch back to USE VISION.

## Commands

All commands except **Stop** must begin with **Robot**. Speak one command at a
time and wait for the action to finish; commands spoken during an operation are
discarded rather than stored for later. Movement, gripping, and mode changes
require an exact complete recognized phrase and confident recognition.

| Say | Action |
| --- | --- |
| Robot use voice / Robot use vision | Select movement input, initially stopped |
| Robot resume | Enable movement; lift first if below travel height |
| Robot move left / Robot move right | Move 50 mm from the current position |
| Robot move toward me / Robot move away from me | Move 20 mm radially away from / toward the arm base |
| Robot a little left / Robot a little right | Move 5 mm |
| Robot a little toward me / Robot a little away from me | Move 5 mm |
| Robot grab / Robot grab the item / Robot pick up | Open, lower, close, lift |
| Robot release / Robot release the item / Robot let go | Open at the current location |
| Stop / Robot stop | Interrupt movement or pickup; retain the gripper command |
| Robot fix my aim | Stop and run vision calibration again |

Left and right follow the person's view, matching the map. “Toward me” increases
the gripper's distance from the arm base; “away from me” decreases it. Commands
that cross the configured workspace are rejected. Direction commands require
VOICE mode and MOVE AGAIN/resume; gripper commands work in either mode.

## 5. Test with the real arm

After verifying your microphone and the pretend arm:

```bash
launch/eyereach.sh --voice --microphone 2
```

Omit `--microphone 2` to use the default input. Keep the existing startup procedure
from the README: support the arm during power/torque transitions and keep its
power switch accessible. Verify the 50 mm left/right and 20 mm forward/back movements in a clear
workspace before picking anything up. The software STOP requests a controller
hold and cancels future pickup steps; it is not a hardware emergency stop.

Pickup stages now use measured arm arrival and gripper action results. Rejected
commands, lost joint feedback, or stage timeouts stop progression. A gripper
contact/stall result is not proof that an object was captured. The gripper stays
closed until release even if the position estimate reports no object.

The real hardware controller must report successful gripper completion. If
contact produces an aborted result, inspect the controller's `allow_stalling`
configuration and the servo overload/current limit described in README step 5c.
While the controller is running, inspect its setting with:

```bash
ros2 param get /gripper_controller allow_stalling
```

If the controller reports `false`, a helper can test the supported runtime change:

```bash
ros2 param set /gripper_controller allow_stalling true
```

Check that the command reports success. To persist it, add this entry under the
existing `gripper_controller: ros__parameters:` section in the installed arm
controller YAML, then rebuild/source the workspace and restart the controller:

```yaml
gripper_controller:
  ros__parameters:
    allow_stalling: true
```

Preserve the existing joint, interface and other settings; do not replace the
whole file with this snippet. Locate the actual config in your checkout using
`rg -n 'gripper_controller|GripperActionController' ~/colcon_ws/src/open_manipulator`
(or search those strings in the editor). If the runtime parameter is read-only,
use the YAML change and restart instead.

Do not mask an overload as a successful grasp. See the
[Humble gripper controller documentation](https://control.ros.org/humble/doc/ros2_controllers/gripper_controllers/doc/userdoc.html).

## Failures and fallback

- **Voice model missing:** rerun `bash launch/setup_voice.sh`, or select an existing
  folder with `--voice-model /path/to/model`.
- **No input / microphone disconnected / audio overflow:** voice becomes
  unavailable and the arm is stopped. Look at USE VISION if in VOICE, then MOVE
  AGAIN to continue with vision. A helper can reconnect the microphone and restart
  the application to restore voice. Voice mode does not require a tracked face to
  keep executing spoken commands, but a working face view is needed for gaze buttons.
- **Unrecognized speech:** speak an exact phrase, check the microphone device,
  and test the microphone-only diagnostic. No command is inferred from casual
  conversation. Recognition still needs testing with the intended person's voice,
  room noise, accent, and speech characteristics.
- **Old command after a switch or operation:** it is discarded intentionally.
  Speak the next command after the current action finishes.
- **No synthesized spoken prompts:** while microphone commands are enabled,
  `spd-say` prompts are disabled so the app cannot hear its own commands. Status
  and acknowledgement appear on screen. Standalone STOP remains listening.
- **Vision-only fallback:** `launch/eyereach.sh --no-voice` needs no speech packages.
- **STOP below travel height:** say Robot resume or look at MOVE AGAIN to lift
  vertically before moving sideways. A failed recovery is shown as an arm fault.

## Verification and limitations

Run `python3 -m unittest discover -s tests -v` from the repository. Tests cover
phrase matching, mocked microphone streaming, confidence rejection, mode ownership,
stale/duplicate/busy commands, STOP at pickup stages, gripper rejection and contact
feedback, recovery, bounded movements, and gaze button latching.

Local verification passed 64 automated tests and recognized 11/11 generated command
recordings with the actual offline model (macOS Vosk 0.3.44). The Linux ARM64
Vosk 0.3.45 wheel was also verified available. These checks do not establish
real microphone accuracy, Jetson response time,
physical stopping distance, fingertip height, or grasp reliability. Those require
Jetson and hardware measurements. The camera pipeline is otherwise unchanged in
this branch. Selecting voice for movement happens after the normal initial
calibration; this version does not provide a camera-free startup.
