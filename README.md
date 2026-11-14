# EyeReach

A robot arm that a person with severe arm paralysis controls with their head and eyes: point at a spot on a
table map on the screen and the arm follows, blink twice to pick something up, blink twice again to put it down.
Amazon Nova finds the objects and talks back; a LangChain deepagents task agent plans "bring me the pills".

Built at HackPrinceton Fall 2026 (Nov 13-15, 2026). This is a research prototype, not a medical
device, and it has not been tested with people who have disabilities.

## Status

| Part | State |
|---|---|
| Head/eye pointer, calibration and the screen app | **works on the real arm** (team tests): the arm follows the pointer live; double blink picks up and lets go |
| Pick-up on the real arm | **works**: opens, lowers to 5 cm, closes, lifts, holds until the next double blink; gripper fixes in README steps 5c and Problems |
| Pointer accuracy | measured on one teammate only: 7.0% of the screen width on the calibration dots (head and eyes alike); `tools/measure.py` measures it properly |
| Amazon Nova 2 Lite (find objects) | **works through Bedrock** (live call checked); not yet used by the eye app |
| Amazon Nova 2 Sonic (talk back) | **works through Bedrock**: spoken replies checked; a live conversation through the speaker not yet tried |
| Task agent (LangChain deepagents) | built and tested with Nova 2 Lite as its model; **its tools are pretend** (it doesn't move the real arm yet) |
| Photon caregiver messages | not built (the agent's `notify` tool is pretend) |
| HoloLens 2 gaze input | not built |
| Tested with people who have disabilities | no |

## Tracks and sponsors

Main track: **Healthcare**. EyeReach restores a basic action (picking something up) to a person who cannot
use their arms. Special tracks, and how each connects. The status column says what exists today.

| Track | How EyeReach connects | Status |
|---|---|---|
| **Best Hardware** | A physical system: OpenMANIPULATOR-X arm, OpenCR board, Jetson Orin NX, a webcam and a USB speaker with a microphone, wired and driven by our own control code. | **Working on the real arm** |
| **Best Physical AI** | AI that acts in the real world: face/eye tracking (MediaPipe Face Mesh) decides where a physical arm goes, with height, speed and reach limits on every move. | **Working on the real arm** |
| **Amazon: Best Practical AI (Nova)** | Three Amazon Nova uses, all on Bedrock (see [Amazon technology](#amazon-technology-we-use)): Nova 2 Lite names the objects on the table and boxes them, Nova 2 Lite is the task agent's brain, and Nova 2 Sonic listens and talks back. Nova's answers are only suggestions: the user still confirms before anything moves. | Vision and voice work through Bedrock; **not yet connected to the arm** |
| **LangChain deepagents** | An agent turns a task ("bring me the pills") into steps: find the object, ask the user to confirm with their eyes, pick, drop, tell the caregiver. Human in the loop: `pick` refuses until the user confirmed that exact object. | Built and tested; **tools still pretend** |
| **Photon (spectrum-ts)** | Planned: a caregiver gets an iMessage when the arm grabs or drops something, or when the user keeps looking at something it cannot reach. | Not built |

Next: connect Nova vision and the task agent to the real arm (needs a camera looking at the table, and a mapping from
that camera's picture to the arm's table positions), then Photon. Further ahead: HoloLens 2 gaze input, so the user
looks at real objects instead of a screen map.

## Amazon technology we use

Everything Amazon runs on **Amazon Bedrock** in `us-east-1`, with one IAM access key pair in `.env` (git-ignored;
`launch/eyereach.sh` and `launch/talk.sh` load it). That key is Bedrock-only, which is enough for all three uses.

| Amazon technology | What EyeReach uses it for | How it is called | Where |
|---|---|---|---|
| **Amazon Nova 2 Lite** (`us.amazon.nova-2-lite-v1:0`) | Looks at a camera picture and lists the objects a robot could pick up, each with a box; or says which object a request ("the pills") means, or that it isn't sure | Bedrock **Converse API** with the image (boto3 `bedrock-runtime`), one call per picture, no retries | `eyereach/vision_nova.py` |
| **Amazon Nova 2 Lite** as an agent model | The "brain" of the task agent: plans the steps, calls the tools, writes notes | LangChain `init_chat_model("bedrock_converse:…")` (langchain-aws) | `agent/task_agent.py` |
| **Amazon Nova 2 Sonic** (`amazon.nova-2-sonic-v1:0`) | Speech to speech: hears the person through the microphone and answers out loud in a natural voice (voice `matthew`) | Bedrock **bidirectional streaming** (`InvokeModelWithBidirectionalStream`) with Amazon's Python SDK `aws_sdk_bedrock_runtime` (needs Python 3.12) | `agent/voice_sonic.py`, `launch/talk.sh` |
| **Amazon Bedrock** | Hosts all three models; one key, one region | IAM keys from `.env` | — |

Not used, on purpose: **Amazon Polly** and **Amazon Transcribe**. Nova 2 Sonic already listens and speaks in one model,
and the key is Bedrock-only (Polly and Transcribe answer "access denied").

Design rules for the Amazon parts:
- **Nothing safety-critical depends on the network.** STOP is offline (the gaze STOP button, or saying "Stop" to the
  offline Vosk recogniser); if Bedrock can't be reached, the app keeps running and the arm still works by eye.
- **Nova suggests, the person decides.** The agent's `pick` refuses until the person confirmed that exact object.
- **Sonic only talks.** It never moves the arm, and it stops listening while it speaks so it can't hear itself.
- **Few calls.** One Nova call per picture, no retries; Sonic only runs while `launch/talk.sh` is open.

## LangChain and the other technology

| Technology | What it does in EyeReach |
|---|---|
| **LangChain deepagents** (sponsor) | `create_deep_agent` with two sub-agents (perception finds the object, caregiver writes the message), a notes file it keeps between runs (`FilesystemBackend`, `~/.eyereach/agent/notes.md`), and a written plan (`TodoListMiddleware`, `write_todos`). Model: Amazon Nova 2 Lite through `langchain-aws`. Runs in its own Python 3.12 environment (`~/eyereach_agent_venv`, made with `uv`), so ROS's Python 3.10 is untouched. |
| **Photon spectrum-ts** (sponsor) | Planned for caregiver iMessages; not built. |
| NVIDIA Jetson Orin NX 16GB | Runs everything on the robot: camera, face tracking, ROS 2, the arm. |
| ROS 2 Humble, ROBOTIS OpenMANIPULATOR-X, DYNAMIXEL servos, OpenCR | The arm and its drivers; our code sends joint paths through `arm_controller`. |
| MediaPipe Face Mesh | 478 face points per webcam frame: the head direction (eye corners, nose, forehead, chin) and the irises. |
| Vosk (offline) | The fixed voice commands ("Robot move left", "Stop") without the internet. |
| OpenCV, NumPy | The camera, the screen app and the maths. |
| One Euro filter (Casiez et al., CHI 2012) | Keeps the pointer still when you hold still, without lag on a deliberate move. |

## What we built

The main features (details in the sections below):
- **Head pointing with a visible ray:** the pointer follows where the face points (steadier than the irises alone); the
  live face is shown with its key points and an animated ray from the nose to the spot. `--pointer eyes|both` too.
- **Calibration that runs itself:** starts when the face is steady, a gliding dot that waits for you, one automatic
  retry when poor; saved per screen and pointer, so it is reused (FIX MY AIM redoes it).
- **Moving the arm:** live following (default) or look-and-hold; slow speeds (60 mm/s, 40 carrying); travels 12 cm
  above the table.
- **Double blink to pick up and let go**, ignored while the head turns and in the first 3 s; big gaze buttons (and
  mouse clicks) as a backup.
- **A safe pick-up:** gripper opens first, lowers to 5 cm, closes, lifts and holds until the next double blink;
  gripper opened before the arm moves at start and before it stops.
- **Gripper fixes:** a current limit so it can't overload (setup step 5c), and a start-up check that resets its turn
  count, which had made "open" drive it closed.
- **Session log and measurement tool:** every calibration, move and pick-up in `~/.eyereach/sessions/`;
  `tools/measure.py` for accuracy (degrees) and grab success.
- **Test everything in one command:** `launch/test_all.sh` (16 checks, never moves the real arm).
- **One command:** `launch/eyereach.sh` loads `.env`, finds the screen, checks the gripper and joints, starts the arm
  and the app, and adds voice commands when installed. `launch/setup_jetson.sh` sets up a Jetson in one go.
- **Voice:** offline commands (Vosk), Amazon Nova 2 Sonic conversation (`launch/talk.sh`); the webcam's microphone
  first, the speaker's as fallback; every phrase heard is shown, with why it was ignored.
- **Speaker volume on screen:** QUIETER / LOUDER buttons by gaze or mouse, `+`/`-` for the helper.
- **Amazon Nova 2 Lite** object finding and the **LangChain deepagents** task agent (not yet driving the real arm).

## Test everything (one command)

```bash
launch/test_all.sh              # everything, about 2 minutes (2 small AWS calls)
launch/test_all.sh --offline    # no AWS calls      --no-sim: skip the simulation      --no-sound: no beep or voice
```

It **never moves the real arm** and ends with a PASS / FAIL / SKIP table (exit code 1 if anything failed):
1. **Environments and keys:** both Python environments; `.env` loads the AWS keys (values never printed) and is git-ignored.
2. **Unit tests** in both environments, and the robot maths (`brain/arm.py test`).
3. **Devices:** a webcam frame, the OpenCR, the servos and the gripper's turn count (read only), which microphone is
   picked and that it records, and a beep on the speaker.
4. **The arm's motion on the ROS simulation** (`tools/sim_check.py`, fake hardware, same code as the real arm): the
   gripper starts closed and start-up opens it; a double blink goes down open to the pick-up height, closes, lifts and
   holds; the next double blink opens it; it is open after stopping.
5. **Amazon on Bedrock:** one Nova 2 Lite call on the webcam photo, and one Nova 2 Sonic reply played on the speaker.

Then try the real arm (someone holds it): `launch/eyereach.sh`, and talk to it: `launch/talk.sh`.

## Offline voice commands

Offline microphone commands (merged from the `test/voice-vision-control` branch) add gaze-based
mode switching, bounded spoken moves, and interruptible pickup with robot feedback.
See [Jetson voice setup and commands](docs/jetson-voice.md) for installation and testing.
Use `bash launch/setup_voice.sh` once, then `launch/eyereach.sh --no-arm --voice`.
`launch/eyereach.sh` turns voice commands on by itself once they are installed; `--no-voice` keeps vision only.

**When a voice command doesn't work,** the bottom of the screen says what was heard and why it was ignored, for
example `Heard: robot move lift - not a command` or `Heard: robot move left - not sure (0.62, needs 0.70)`; the session
log keeps them as `voice heard`. Every word must reach 0.70 confidence (lowered from 0.80, which dropped too many real
commands); "Stop" always acts at once. Practise without the robot: `~/eyereach_venv/bin/python -m eyereach.voice_commands`.

## How it works

```
eyereach/run_webcam.py     webcam -> face direction (or iris gaze) -> a point on the table map (gaze_webcam.py)
   ↓  eyereach/eye_control.py: dwell to confirm, grab, hold, drop
brain/arm_link.py          follow a table point, grab, release (or a pretend arm)
   ↓
brain/arm.py               inverse kinematics in mm, joint limits, never below the table
   ↓
gestures/gestures.py       sends the joint moves over ROS 2
   ↓  ROS 2 action
arm_controller             turns them into a smooth path   ┐ started by
   ↓                                                        │ hardware.launch.py
dynamixel_hardware_interface   sends servo commands        ┘
   ↓  USB /dev/ttyACM0
OpenCR (usb_to_dxl)        passes commands to the servos, which report their angles back (/joint_states)
```

How the control behaves. It is made for someone who can't use their arms: after a helper starts it, nothing
needs a keyboard, and the screen always says what to do next. With microphone commands off, `spd-say` can also speak prompts.

### Known problem: pointing with the eyes alone is jumpy

The first version moved the pointer from the irises only. In a webcam picture an iris is a few pixels wide,
so the pointer jumped around and moved too fast to follow, and it was hard to tell where the system thought
you were looking (calibration error 4.5-7% of the screen width, up to 200 px at the corners).

**What changed:** the pointer now comes from where the face points (`--pointer head`, the default). Face Mesh
gives 3D key points; the eye corners, nose tip, forehead and chin give the face's direction, drawn as a ray
out of the nose. Your face is shown live with its key points, and an animated ray runs from your nose on that
picture to the spot you are pointing at, so you always see where you point. A One Euro filter keeps the
pointer steadier while you hold still and follows deliberate moves with less smoothing. Actual latency
and accuracy still need measuring with the camera and arm together. Blinking doesn't move a head pointer, which also helps the double blink.
`--pointer eyes` keeps the eyes-only pointer for people who can't move their head, and `--pointer both`
uses the head for the rough spot and the eyes to fine-tune. Not yet measured on people.

- **Why calibrate:** the webcam sees which way the face turns (or where the irises sit), not what that points
  at on the screen. That depends on the person, where they sit and where the camera is, so it is learned each
  time (and saved per pointer mode and screen size).
- **Calibrate (about 40 s):** the screen says hello and starts by itself once the face has been steady in view
  for 3 seconds. A yellow dot glides around 12 spots; point your nose at it (with `--pointer eyes`, follow it
  with the eyes). Your face and its ray are shown under the dot. The dot waits while the face is lost. If it comes out poor (over 7% of the screen width) it tries once more.
- **Move in real time:** point at the table map and the arm follows immediately, with no movement dwell.
  A larger mirrored face panel stays to the left of the map, and one aim line connects the nose to the
  calibrated pointer. The arm keeps its limits of 60 mm/s (40 mm/s while carrying); the pointer uses
  responsive adaptive smoothing. Looking at the face panel or buttons stops live following.
  Closing the eyes, losing tracking, or pausing stops new movement commands. Pickup/release uses robot feedback; STOP interrupts the sequence and holds the arm. STOP stays stopped until you look at MOVE AGAIN.
  Use `--movement dwell` for the previous destination-based mode.
- **Pick up / let go anywhere:** the fingertips start and travel about 120 mm (12 cm) above the table,
  according to the robot model. Blink twice quickly, look at PICK UP for 0.65 s, or click it: the arm
  opens the gripper and waits one second, lowers to 50 mm (5 cm) with it open, closes it fully, waits one
  second for closure, then lifts back to
  its travel height. **The gripper is open the rest of the time:** at start-up it opens first (and waits until it
  reports open) before the arm moves, and when the app closes it opens and waits again before the arm stops.
  It only closes when you pick up, and then **stays closed, holding what it grabbed, until your next double
  blink** (or LET GO), which opens it where it is and drops the item. (The fingers' position is too unreliable
  to tell whether something was caught, so it is only written to the session log.) A double blink only counts
  while the head is holding still (turning
  or nodding changes how open the eyes look) and not in the first 3 s after the face appears.
- **Buttons:** PICK UP / LET GO, STOP / MOVE AGAIN, and FIX MY AIM. USE VOICE / USE VISION switches movement input hands-free. No fixed HOME or DROP-OFF
  buttons or map boxes. Buttons support gaze dwell (0.65 s by default, set with `--dwell`) and mouse clicks.
- **Speaker volume:** under the face picture, a bar with the volume and two big buttons, **QUIETER** and **LOUDER**
  (10% per press). Look and hold, like the other buttons, or click; keep looking and it keeps stepping. They always
  work, even mid pick-up. The helper can press **+** and **-**. It is the system speaker volume (`pactl`), so it is
  the same for Nova 2 Sonic (`launch/talk.sh`); `python3 -m eyereach.speaker_volume 70` sets it from a terminal.
- **Losing the eyes** (looking away, closing them, the face out of view) never starts anything.
- **Helper keys:** Esc closes, C calibrates again, Space pauses, V hides/shows the face picture. Options:
  `--pointer head|eyes|both`, `--dwell 2`,
  `--dot-seconds 3`, `--quiet` (no voice), `--reuse` (skip calibration), `--windowed`, `--grab-mm 50`
  (pick-up fingertip height above the table in mm; default 50, never below 30), `--travel-mm 120` (start-up
  and moving height; default 120). Heights come from the robot model, which assumes the arm's base sits
  directly on the table: if it stands on a plate, put the plate's thickness in `MOUNT_HEIGHT` in `brain/arm.py`.
- **Measuring:** every session is logged to `~/.eyereach/sessions/` (calibration time and error, error at each
  dot, every move, pick-up and let-go, and the time of each pick-and-place). Add `--person NAME`, and for the
  error in degrees `--screen-cm` (width of the picture on the screen) and `--distance-cm` (eyes to screen).
  `python3 -m eyereach.session_log` sums up all sessions per person.

## Talk to it (Amazon Nova 2 Sonic)

`launch/talk.sh` is a spoken conversation with EyeReach through Amazon Nova 2 Sonic (speech-to-speech on Bedrock,
model `amazon.nova-2-sonic-v1:0`, `us-east-1`): speak into the microphone (the C920 webcam has one) and it answers
through the speaker. It uses the same AWS keys from `.env` as Nova vision; Polly and Transcribe are not needed (the
current key is Bedrock-only and can't use them anyway).

```bash
launch/talk.sh                        # talk; Ctrl+C ends
launch/talk.sh --text "Hello"         # one typed sentence, reply saved to /tmp/eyereach_sonic_reply.wav (no mic/speaker)
launch/talk.sh --list                 # audio devices; pick with --microphone N / --speaker N
```

- It only talks: it never moves the arm. STOP stays offline (the gaze STOP button, or saying "Stop" to the eye app).
- It stops listening while it speaks, so it can't hear itself through the speaker.
- Sonic only answers while an audio input is open, so even `--text` sends silence while it waits.
- It runs in `~/eyereach_agent_venv` (Python 3.12, which the Sonic SDK needs); `launch/setup_jetson.sh` installs it.
- **Microphone: the webcam's first, the speaker's as fallback** (both voice programs, `eyereach/audio_devices.py`).
  Measured with the same speech played through the speaker: the C920 heard it twice as loud with word confidence 0.95,
  the speaker's microphone 0.87 (voice commands were often dropped on it). The webcam goes through PulseAudio and is
  made the default source each start; the speaker's microphone (which PulseAudio doesn't list) is opened directly
  when there is no webcam. `python3 -m eyereach.audio_devices` shows the choice; `EYEREACH_MIC=PnP` forces the
  speaker's.
- Sound devices: `pactl set-default-source` / `pactl set-default-sink` choose the system microphone and speaker
  (`pactl list short sources`, `pactl list short sinks` list them).

## Folders

```
eyereach/      the control: gaze_webcam.py (face direction, iris gaze, calibration, smoothing),
               eye_control.py (pointer following, dwell, double blink, pick up, let go), run_webcam.py (the screen app),
               session_log.py (the session log and its summary)
               vision_nova.py (Amazon Nova 2 Lite: names the objects on the table and gives each a box)
               voice_commands.py (offline Vosk commands), audio_devices.py (webcam mic first, speaker's as fallback)
               speaker_volume.py (the speaker volume behind QUIETER / LOUDER)
agent/         task_agent.py: the deepagents task agent ("bring me the pills": plan, confirm, pick, drop, tell)
               voice_sonic.py: talk with Amazon Nova 2 Sonic (microphone in, spoken answers out)
brain/         arm.py: the robot maths, plus commands to check it (test, where, go, home)
               arm_link.py: the arm follows a point, grabs and releases (or a pretend arm)
gestures/      gestures.py: the ROS 2 link that moves the joints
launch/        eyereach.sh (one command: the arm and the app), talk.sh (Nova Sonic conversation),
               test_all.sh (test everything without moving the real arm),
               setup_jetson.sh and setup_voice.sh (one-time setup), stop.sh (stop everything),
               fix_turns.py (joint and gripper turn checks before each start)
firmware/      servo_check: an Arduino test that stands the arm up and waves
tools/         measure.py: gaze accuracy on 9 test dots, and timed grab trials, into results/measurements.csv
               sim_check.py: the whole pick-up on the ROS simulation (run by launch/test_all.sh)
tests/         run with: python3 -m unittest discover -s tests
dependencies.repos   the ROBOTIS code to download (setup step 5)
```

## Try it on a laptop (no robot)

```bash
git clone https://github.com/aditya-baniya-ai/hack_princeton.git
cd hack_princeton
pip install -r requirements.txt
python3 -m unittest discover -s tests
python3 -m eyereach.run_webcam            # a pretend arm; asks for the camera, then calibrates
```

### Amazon Nova (finding the objects)

`eyereach/vision_nova.py` sends one camera image to Amazon Nova 2 Lite on Bedrock and gets back each object's name and
box. Keys come only from environment variables (never commit them):

| Variable | What |
|---|---|
| `AWS_BEARER_TOKEN_BEDROCK` | a Bedrock API key (or set `AWS_ACCESS_KEY_ID` and `AWS_SECRET_ACCESS_KEY` instead) |
| `AWS_REGION` | default `us-east-1` |
| `NOVA_MODEL_ID` | default `us.amazon.nova-2-lite-v1:0` |

```bash
python3 -m eyereach.vision_nova table.jpg                # lists the objects and their boxes
python3 -m eyereach.vision_nova table.jpg "the pills"    # also says which object is the pills, or that it is not sure
```

If Nova cannot be reached it logs a warning and returns no objects; the app keeps running. The tests use saved answers
in `tests/fixtures/`, so they need no network.

### The task agent (LangChain deepagents)

`agent/task_agent.py` takes a task like "bring me the pills" and plans it: a perception sub-agent finds the object,
the person confirms it with their eyes, the arm picks and drops it, and a caregiver sub-agent sends a message. It keeps
a notes file (`~/.eyereach/agent/notes.md`, e.g. "medicine: the pill bottle, next to the cup") and reads it on later
tasks. Nothing moves until the person confirms that exact object: `pick` refuses otherwise, whatever the model says.

The tools are pretend for now (no arm, camera or Photon needed). The model is Amazon Nova 2 Lite on Bedrock, with the
same AWS keys as above; `AGENT_MODEL_ID` changes it and `EYEREACH_NOTES` moves the notes folder.

```bash
python3 -m agent.task_agent "bring me the pills"                     # needs Python 3.11+ (Jetson: setup step 9)
python3 -m agent.task_agent "bring me the snack" --photo table.jpg   # objects found by Nova in a photo
```

### Measuring it (accuracy and grab times)

```bash
python3 -m tools.measure accuracy --person NAME --screen-cm 29 --distance-cm 55
python3 -m tools.measure trials --person NAME
```

`accuracy` uses the saved calibration (or calibrates; `--recalibrate` forces it), then shows 9 test dots placed away
from the calibration dots, so the error is measured on spots the tracker never learned. It prints the average and
worst error in pixels, % of the screen width and degrees. `--screen-cm` is the width of the lit screen picture and
`--distance-cm` is eyes to screen: measure both on each setup, they are only used for the degrees.
`trials` is for a helper at the keyboard while the person uses the arm: Enter when they start, then `y` if the grab
worked or `n` if not. Both add rows to `results/measurements.csv`.

## Hardware

- OpenMANIPULATOR-X arm (5 DYNAMIXEL XM430-W350-T servos, payload 500 g, reach 380 mm) on a base plate
- OpenCR 1.0 board and its 12 V 5 A power adapter, micro USB data cable
- NVIDIA Jetson Orin NX 16GB, Ubuntu 22.04 (JetPack 6), ROS 2 Humble
- A webcam facing you for the eye tracking
- A Mac or PC with the Arduino IDE, only to upload firmware to the OpenCR

| ID | Joint | ROS name |
|---|---|---|
| 11 | Base (rotation) | joint1 |
| 12 | Shoulder | joint2 |
| 13 | Elbow | joint3 |
| 14 | Wrist | joint4 |
| 15 | Gripper | gripper |

The servos share one daisy-chained TTL cable (1 Mbps, Protocol 2.0). 180° on the servo (2048 steps) is the
center and equals 0 rad in ROS. ROBOTIS's default position gains are P/I/D 800/100/100; this setup uses
400/0/0 for the base (step 5b) so it doesn't jitter.

- **Wiring:** keep the board's power switch OFF while connecting cables. The arm's white 3-pin cable goes to
  the first 3-pin TTL port on the bottom-left edge of the OpenCR (not the 4-pin ports). 12 V adapter to the round
  jack. The OpenCR micro USB (right edge) goes to the Mac (for Arduino) or the Jetson (for ROS 2).
- **Mounting direction:** with the base servo at 0° the arm should face your work area, because the base turns only
  about ±180° from there. To check, start the arm and run `ros2 topic echo /joint_states --once`: `joint1` should be
  close to 0 while the arm faces forward. If it is close to ±3.14, the arm is mounted backwards: turn off the 12 V,
  unscrew the base bracket, turn the **whole** arm 180° (hold it by the bottom servo), and screw it down again.
- **Power off = the arm slumps.** With the 12 V off the arm folds under its own weight, often past the joint
  limits. Before starting it again, hold it standing roughly straight up and keep holding it while you turn the
  12 V on. The programs check this at startup and refuse to move the arm if a joint is out of range.

## One-time setup (on the Jetson)

**Quick way:** do step 1 below on a Mac or PC, clone the repo (step 3), then run steps 2 and 4 to 9 with one script.
It checks each step first and skips what is done, so it is safe to run again. It asks for your password and never
moves the arm.

```bash
bash launch/setup_jetson.sh            # add --voice for the offline voice commands
```

The steps below are what it does, if you want to do them by hand.

### 1. Upload the firmware to the OpenCR (on a Mac or PC)

The Arduino IDE can't upload to the OpenCR from the Jetson, so use a Mac or PC.

1. Install the Arduino IDE.
2. Settings > Additional boards manager URLs:
   `https://raw.githubusercontent.com/ROBOTIS-GIT/OpenCR/master/arduino/opencr_release/package_opencr_index.json`
3. Boards Manager: install **OpenCR**.
4. Plug in the OpenCR and turn on its 12 V power. Tools > Board > OpenCR > OpenCR Board, and Tools > Port > the
   `usbmodem` port.
5. File > Examples > OpenCR > 10.Etc > **usb_to_dxl**, then Upload.

A good upload ends with `[OK] Download`. The arm won't move after this, which is expected: it now waits for
commands from the Jetson. If the upload freezes, hold **SW2**, press **RESET**, release SW2, and upload again.

To test the arm without ROS first, upload `firmware/servo_check/servo_check.ino` instead (it locks all joints,
turns the base 90°, stands the arm up and waves the wrist; open the Serial Monitor at 115200 baud to start it;
install the **Dynamixel2Arduino** library first). Uploading one sketch replaces the other, so put `usb_to_dxl`
back before using ROS 2.

### 2. Install ROS 2 Humble

```bash
sudo apt install -y software-properties-common curl
sudo add-apt-repository -y universe
sudo curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" | sudo tee /etc/apt/sources.list.d/ros2.list
sudo apt update
sudo apt install -y ros-humble-desktop ros-dev-tools
```

### 3. Download this repo

```bash
mkdir -p ~/Documents && cd ~/Documents
git clone https://github.com/aditya-baniya-ai/hack_princeton.git
```

### 4. Allow your user to use the USB port

```bash
sudo usermod -aG dialout $USER
```

Log out and back in (or reboot); a new terminal is not enough. Check that `groups` lists `dialout`.

### 5. Install and build the ROBOTIS arm software

```bash
sudo apt install -y ros-humble-ros2-control ros-humble-ros2-controllers ros-humble-moveit ros-humble-gripper-controllers
mkdir -p ~/colcon_ws/src
vcs import ~/colcon_ws/src < ~/Documents/hack_princeton/dependencies.repos
cd ~/colcon_ws
sudo rosdep init        # fine if it says it is already initialized
rosdep update
rosdep install --from-paths src --ignore-src -y -r --skip-keys "gazebo_ros gazebo_ros_pkgs gazebo_ros2_control gazebo_plugins"
colcon build --symlink-install
```

This takes a few minutes and should end with `Summary: 12 packages finished` (warnings are fine). If it freezes,
press Ctrl+C and run `colcon build --symlink-install --parallel-workers 1`.

### 5b. Calm the base servo (stops it jittering)

With ROBOTIS's default settings the base jitters about ±0.3° when the forearm sticks out. These commands give
only the base servo gentler settings (P 400, I 0, D 0). They are sent at every launch and not stored in the servo.
Keep a backup first:

```bash
X=~/colcon_ws/src/open_manipulator/open_manipulator_x_description/ros2_control/open_manipulator_x_system.ros2_control.xacro
cp $X ~/omx_ros2_control_backup.xacro
sed -i '/<gpio name="dxl1">/,/<\/gpio>/ { s|"Position P Gain">800<|"Position P Gain">400<|; s|"Position I Gain">100<|"Position I Gain">0<|; s|"Position D Gain">100<|"Position D Gain">0<|; }' $X
sed -n '/<gpio name="dxl1">/,/<\/gpio>/p' $X | grep "Gain"      # should show P 400, I 0, D 0
cd ~/colcon_ws && colcon build --symlink-install --packages-select open_manipulator_x_description
```

To undo, copy the backup back and rebuild the same way. Repeat this step if you ever run `vcs import` again.

### 5c. Limit the gripper's squeeze (stops it overloading)

Without this the gripper squeezes at full current (about 3.2 A) whenever it closes on something, overloads within
seconds, and switches itself off (every later grab then "holds nothing"). This caps it at about 0.8 A:

```bash
X=~/colcon_ws/src/open_manipulator/open_manipulator_x_description/ros2_control/open_manipulator_x_system.ros2_control.xacro
cp $X ~/omx_ros2_control_backup_before_gripper.xacro
sed -i '/<gpio name="dxl5">/,/<\/gpio>/ { /<param name="Return Delay Time">0<\/param>/a\        <param name="Operating Mode">5</param>\n        <param name="Goal Current">300</param>
}' $X
sed -n '/<gpio name="dxl5">/,/<\/gpio>/p' $X | grep param      # should list Operating Mode 5, Goal Current 300
cd ~/colcon_ws && colcon build --symlink-install --packages-select open_manipulator_x_description
```

If the gripper already overloaded (it won't move; `/tmp/robo_arm.log` says "Hardware error occurred" for ID 15),
turn the 12 V off and on again to clear it.

### 6. Set up the terminal and the USB rules

```bash
echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc
echo "source ~/colcon_ws/install/setup.bash" >> ~/.bashrc
source ~/.bashrc
ros2 run open_manipulator_x_bringup create_udev_rules      # asks for your password
```

### 7. Check the OpenCR is connected

Plug the OpenCR's USB into the Jetson, turn on the 12 V power, then `ls /dev/ttyACM*` should print `/dev/ttyACM0`.

### 8. Install MediaPipe (the eye tracker), once

It goes in its own environment, `~/eyereach_venv`, so nothing else on the Jetson changes. `launch/eyereach.sh` uses it
automatically when it exists.

```bash
python3 -m venv --system-site-packages ~/eyereach_venv
python3 -c "import site; print(site.getusersitepackages())" > ~/eyereach_venv/lib/python3.10/site-packages/user_site.pth
~/eyereach_venv/bin/pip install --no-deps mediapipe==0.10.18
~/eyereach_venv/bin/pip install absl-py attrs flatbuffers "protobuf>=4.25.3,<5" sentencepiece
~/eyereach_venv/bin/python -c "import mediapipe, cv2, numpy; print('mediapipe', mediapipe.__version__, 'numpy', numpy.__version__)"
```

NumPy must still be 1.x. To undo all of it: `rm -rf ~/eyereach_venv`.

### 9. Python 3.12 for the task agent, once

The task agent uses LangChain `deepagents`, which needs Python 3.11 or newer, and the Jetson has 3.10. It gets its own
environment, `~/eyereach_agent_venv`. `uv` downloads Python 3.12 for it, so the system Python and ROS are not touched.
The eye tracker and the arm stay in `~/eyereach_venv` (Python 3.10, because ROS 2 Humble needs it).

```bash
python3 -m pip install --user uv
export PATH="$HOME/.local/bin:$PATH"
uv venv --python 3.12 ~/eyereach_agent_venv
uv pip install --python ~/eyereach_agent_venv/bin/python "deepagents>=0.7" "langchain-aws>=1.0" boto3 numpy opencv-python-headless
~/eyereach_agent_venv/bin/python -m unittest tests.test_task_agent           # from the repo folder
~/eyereach_agent_venv/bin/python -m agent.task_agent "bring me the pills"
```

To undo: `rm -rf ~/eyereach_agent_venv`. On a laptop with Python 3.11+ none of this is needed.

## Run it with the arm

Always start with the pretend arm, then the simulated arm, then the real one.

```bash
launch/eyereach.sh --no-arm      # pretend arm, no ROS
launch/eyereach.sh --sim         # simulated arm in RViz
launch/eyereach.sh               # the real arm
launch/eyereach.sh --calibrate   # the real arm, with a new calibration
```

**One command runs everything that is connected:** `launch/eyereach.sh` loads `.env` (AWS keys), finds the screen,
checks the gripper and joints, starts the real arm, opens the gripper, and runs the head/eye pointer with the
double blink, using the saved calibration. Voice commands join in by themselves once `bash launch/setup_voice.sh`
has been run and a microphone is plugged in (otherwise it says "Voice commands: off" and runs without them).
Nova object finding and the task agent are not wired to the arm yet (the agent's tools are still pretend).

The launcher has its settings at the top of `launch/eyereach.sh`: it **reuses the saved calibration** (only the
first run, or a new screen size or pointer mode, calibrates; FIX MY AIM or C recalibrates any time), and the
fingertips **travel 120 mm** above the table and **go down to 50 mm** to pick up (the gripper opens first, closes at
the bottom and stays closed until the next double blink or LET GO). Change `TRAVEL_MM`, `GRAB_MM` or
`REUSE` there, or override one run with `--travel-mm 120`, `--grab-mm 50` or `--calibrate`.

For the real arm: turn on the 12 V power, make sure the OpenCR's USB is in the Jetson, **hold the arm** (it goes limp
when it starts and stops), keep the power switch within reach, then run it. If it stays on "Waiting for the arm",
look in `/tmp/robo_arm.log`. Esc in the app stops the arm too. Other options go to the app (`--camera /dev/video0`,
`--movement dwell`, `--windowed` (full screen is the default), `--dot-seconds 4`, `--dwell 1.2`).

**Start the arm by hand** (for other programs): `ros2 launch open_manipulator_x_bringup hardware.launch.py port_name:=/dev/ttyACM0`,
and wait for `Configured and activated arm_controller`. **Watch the real arm in RViz:** add `start_rviz:=true`.

**Check the joint angles:** `ros2 topic echo /joint_states --once`

**Stop everything** (hold the arm first): `launch/stop.sh`. It should say `Everything is stopped.` To power down, press
Ctrl+C on the arm launch while holding the arm, then turn off the 12 V.

### Check the robot maths and the arm

The point is the middle of the gripper fingertips: **x** forward from the base, **y** to the left, **z** up from the table, in mm.
Pointing straight down it reaches about 100 mm high and 80 to 240 mm in front of the base. It refuses any point it
can't reach safely and never goes lower than 10 mm above the table.

```bash
python3 brain/arm.py test           # the maths only, no arm needed
python3 brain/arm.py where          # where the gripper is now, and the joint angles
python3 brain/arm.py go 200 0 100   # move there pointing down (add --level to point forward)
python3 brain/arm.py home
```

Try the simulation first. On the real arm try `go 200 0 100` first and measure with a ruler: the fingertips should be about
200 mm in front of the base and 100 mm above the table. If the arm stands on a plate, put its thickness in
`MOUNT_HEIGHT` at the top of `brain/arm.py`.

## Problems

- **The gripper stays closed and never opens (every command "accepted"):** the gripper servo counts whole turns
  and had picked up an extra one (it read 5168 with the fingers open; ROS expects about 1057 open and 2648 closed),
  so "open" drove it closed against its stop. Restarting the servo resets the count. `launch/fix_turns.py` (run by
  `launch/eyereach.sh` before every real-arm start) now checks this and restarts the gripper servo if needed;
  `python3 launch/fix_turns.py --check` shows the reading. If it stays off, turn the 12 V off and on.
- **"could not connect to display" / "Qt platform plugin xcb" and the app aborts:** it was started from a terminal
  with no screen (an SSH terminal has an empty `DISPLAY`). `launch/eyereach.sh` now uses the desktop you're
  logged into (the Jetson's monitor or Remote Desktop) and refuses to start the arm if there is none. To choose
  one yourself: `DISPLAY=:10 launch/eyereach.sh` (`ls /tmp/.X11-unix` lists them; X10 is usually Remote Desktop).
- **Permission denied on /dev/ttyACM0:** your user isn't in the `dialout` group. Run the `usermod` line in step 4, then
  log out and back in.
- **The arm doesn't move after uploading usb_to_dxl:** normal. Start it with ROS 2 (see "Run it with the arm").
- **Upload to the OpenCR freezes:** hold SW2, press RESET, release SW2, upload again.
- **"The arm is outside its safe range, so it won't be moved":** the arm slumped with the power off, or the base is at its
  end stop because it is mounted backwards. Hold the arm and turn the 12 V off, stand it straight up facing forward,
  keep holding it while you turn the 12 V on, and start again. `joint1`, `joint2`, `joint3` should be close to 0.
- **The arm (and RViz) keeps shaking:** two programs are controlling the arm (for example a leftover simulation plus the
  real arm). Hold the arm, run `launch/stop.sh`, and start just one thing.
- **The arm stays stiff after everything is stopped:** the servos hold their last position until told to let go. Hold the
  arm, turn the 12 V off, and back on before the next launch.
- **The base jitters (about ±1°) while holding a pose:** do step 5b.
- **"System throttled due to over-current":** the Jetson is briefly slowing itself down. Use the power adapter that came
  with the kit and the 25 W mode (`sudo nvpmodel -m 3` on the Orin NX 16GB, it asks to reboot).
- **A servo ignores a target near 0° or 360°** (Arduino test): keep targets inside 0 to 360; `servo_check.ino` already does.
- **Nodes from another computer show up:** on a shared network, other computers' ROS nodes can appear in
  `ros2 node list`. Use a private network or set a different `ROS_DOMAIN_ID`.
- **`fake.launch.py` refuses to run on the Jetson:** use the `base.launch.py ... use_fake_hardware:=true` command that
  `launch/eyereach.sh --sim` runs.
