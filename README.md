# aHandForMyAunt

[![tests](https://github.com/aditya-baniya-ai/hack_princeton/actions/workflows/tests.yml/badge.svg)](https://github.com/aditya-baniya-ai/hack_princeton/actions/workflows/tests.yml)
Every pull request and push to `main` runs all the tests on GitHub (`.github/workflows/tests.yml`).

A hands-free robot arm for people who have limited use of their arms and hands.
**It tracks the person's eyes to control the robot.** With both eyes and the mouth open, the person steers an
OpenMANIPULATOR-X. Tracking the eyes alone was jumpy, so we **added nose tracking for stability**: the arm follows
the nose and eyes, which is much steadier than the irises. Closing the mouth picks something up. If a park spot is
saved, the arm carries the object there and lets go by itself. Voice is a second way to control it, using Amazon Nova
for natural speech and for finding objects, and offline Vosk commands. **Everything works on the real arm, tested.**

Built at HackPrinceton Fall 2026 (Nov 13–15, 2026). The code and the saved settings keep the internal name `eyereach`.
This is a research prototype, not a medical device. We have not yet tested it with the people it is meant for.

## Why we built it

A teammate's aunt is paralyzed from the neck down. She can still move her face, eyes and mouth. We started this for
her: a way to use that movement to get an everyday object when nobody is nearby to help. We have not tested it with
her yet, so this is our goal, not a proven result.

We bought the OpenMANIPULATOR-X, a Jetson Orin, a camera and a speaker one day before the hackathon. The team says all
the setup, firmware, networking, software and integration happened during HackPrinceton. We used purchased hardware,
open-source robotics packages and pretrained models.

## The build

Buying the arm gave us a robot to move. Our work is everything that lets a person control it without a joystick or
keyboard.

What we set up, and where it lives:
- **Robot and controller:** the OpenMANIPULATOR-X, its OpenCR board (`usb_to_dxl` for ROS) and a servo test sketch (`firmware/servo_check/`).
- **Jetson:** Ubuntu/JetPack, USB permissions, ROS 2 Humble, ROBOTIS packages, Python environments (`launch/setup_jetson.sh`, `dependencies.repos`).
- **Servos:** gentler base gains, a gripper current limit and a start-up check of the turn counts (`launch/fix_turns.py`).
- **Cameras, mic, speaker:** the face and table cameras, microphone choice and speaker volume (`eyereach/audio_devices.py`, `eyereach/speaker_volume.py`).
- **Network and launch:** Wi-Fi, SSH from the Mac, the laptop webcam streamed to the Jetson (`tools/laptop_camera.py`), and one launcher that
  loads settings, checks the arm, opens the gripper, finds the screen and starts the app (`launch/eyereach.sh`, `launch/stop.sh`).

The Mac is for development; the Jetson runs the robot app and ROS. The laptop's webcam is streamed **to the Jetson**
for face tracking, and a USB camera on the Jetson watches the table. Only the newest frame is used, so it never lags.

**Robot maths.** `brain/arm.py` has our forward and inverse kinematics. The app turns a head direction or a voice move
into a target in millimetres, solves the joint angles and checks the path (joint ranges, reach, table and wrist
clearance, speed limits) before sending it. `brain/arm_link.py` keeps the input method apart from the robot:
`VirtualArm` runs without ROS, `RealArm` uses the ROS link in `gestures/gestures.py` (joint paths to the arm
controller, gripper actions to the gripper controller, real angles back on `/joint_states`). The app watches for old
feedback and failures. **Pick-up is a sequence, not one blind close:** open the gripper, lower, close, lift, each step
waiting for feedback. A saved park spot adds carry, lower, release, lift. STOP cancels it and holds the measured position.
These are prototype checks, not certified robot safety.

**Computer vision.** MediaPipe Face Mesh (pretrained) gives face and eye points. We measure the nose position, head
direction, how open each eye is and how open the mouth is, divided by face size so camera distance matters less.
`eyereach/head_tilt.py` turns that into movement (with smoothing and a dead zone); `eyereach/eye_control.py` owns the
on/off logic and the pick-up state, so the face model never sends motor commands directly.

**We track the eyes to control the robot and added the nose for stability.** Both eyes open (with the mouth open) means
"steer". With `--pointer eyes` the eyes can also point at the table, but an iris is only a few pixels wide in a webcam
picture, so the pointer jumped. The default steering follows the **nose**, measured from the point between the eyes,
which is much steadier. Closing the mouth after steering picks up; opening the mouth alone does nothing.

The table camera finds objects. In calibration the arm visits 10 nearby spots, a helper clicks under the fingertips each
time, and a homography maps picture points to robot X/Y. The blue playground shows the reachable area; it does not
promise every object in it can be gripped.

| Work | Where | Job |
|---|---|---|
| MediaPipe face points | on the Jetson | head, eye and mouth state from video |
| Vosk speech | on the Jetson | offline key words, including STOP |
| Kinematics, gesture logic, safety checks | on the Jetson, plain code | turn allowed input into bounded moves |
| Amazon Nova 2 Lite vision | Amazon Bedrock | name objects in a table picture and box them |
| Nova 2 Lite in the deepagents task agent | agent process, uses Bedrock | plan a "bring me X" task with limited tools |
| Amazon Nova 2 Sonic | Bedrock streaming | understand natural speech, speak short prompts |

The control code checks everything a model proposes, and cloud answers never replace the local reach and feedback
checks. Fetching a named object always asks the person to confirm first. The ROS app runs on Python 3.10 and the
Sonic/task-agent part on Python 3.12; they talk through process messages. Local inference still needs benchmarking.

**Simulation and tests.** We test with a light virtual arm and with the ROBOTIS ROS stack on fake hardware shown in
RViz (`launch/test_all.sh`, `tools/sim_check.py`; `use_fake_hardware:=true`, not a physics simulation). The automated
tests cover kinematics, voice bounds, gesture changes, pick-up order, park saving, automatic drop, STOP, cameras and
speech cut-off. They show the code works, not how often the real robot succeeds. We have footage of a full orange
transfer; repeated success counts and timings are still to be recorded.

**Credits and libraries:** ROBOTIS (arm, OpenCR, arm packages), ROS 2 and ros2_control, MediaPipe (478 face points per
frame: head direction and irises) and Vosk (offline key words) as pretrained models, OpenCV and NumPy, the One Euro filter
(Casiez et al., CHI 2012, keeps the pointer still when you hold still without lag on a deliberate move), Amazon (Nova and
Bedrock) and the **deepagents** library for the task agent (`create_deep_agent` with a perception and a caregiver sub-agent, a
notes file and a written plan, running in its own Python 3.12 environment `~/eyereach_agent_venv` made with `uv`, so ROS's
Python is untouched). We added the assistive controls, robot maths, calibration, park workflow, app screen, launch tools
and tests.

## Status

**Everything works on the real robot, tested:** eye-and-nose steering, mouth pick-up and release, voice moves,
named-object fetching, Nova vision, Sonic speech, task-agent planning and automatic park/drop. Footage shows the real arm
gripping an orange, carrying it and letting go at the park spot, and a screen recording shows a spoken request, an
orange detection and a confirmation. These are demo observations, not measured accuracy.

| Part | State |
|---|---|
| Eye and nose control, mouth pick-up and release | works on the real robot, tested; the recording shows face points, steering and pick-up/release |
| Voice moves and Nova 2 Sonic speech | works on the real robot, tested; only necessary prompts, 3-second cut-off |
| Named-object fetch, task agent and park/drop | works on the real robot, tested; footage shows one orange transfer |
| Nova 2 Lite object finding | works with the table camera (orange found in the footage); no accuracy benchmark yet |
| Software tests | 209 run, 205 passed, 4 skipped (need optional dependencies): a software result, not a robot success rate |
| ROS simulation | fake-hardware ROS path and RViz used during development |
| Tests with the intended users | not done yet |

## Tracks and sponsors

Main track: **Healthcare**. aHandForMyAunt explores picking up everyday things for people with limited arm and hand
movement. The hoped-for benefit is more control over daily tasks; user studies are still needed.

| Track | How we connect | Status |
|---|---|---|
| **Best Hardware Hack** | A physical system: OpenMANIPULATOR-X, OpenCR, Jetson Orin NX, webcam, USB speaker and mic, driven by our own control code | **Works on the real arm** |
| **Justin Bojarski: Best Physical AI Hack** | AI that acts in the real world: face and eye tracking decides where the arm goes, with height, speed and reach limits on every move | **Works on the real arm** |
| **Amazon: Best Practical AI Innovation** | Three Nova uses on Bedrock (see [Amazon technology](#amazon-technology-we-use)): vision, the task agent's brain, and Sonic for voice. Nova only suggests; the user confirms before anything moves | Works in the real-robot demo, tested |

Next: repeated real-robot trials with success counts and timing, an object-detection benchmark on labeled images, and tests with
the intended users. HoloLens 2 gaze input is a future idea. The [HackPrinceton Fall 2026 page](https://hackprinceton-fall-26.devpost.com/)
judges creativity, utility, charity and avidity; using a sponsor's technology is not the same as winning its prize, so check each rule.

## Amazon technology we use

Everything Amazon runs on **Amazon Bedrock** in `us-east-1` with one IAM key pair in `.env` (git-ignored; the launchers
load it). The key is Bedrock-only, which is enough for all three uses.

| Technology | What we use it for | How it is called | Where |
|---|---|---|---|
| **Nova 2 Lite** (`us.amazon.nova-2-lite-v1:0`) | Lists the objects a robot could pick up in a camera picture, each with a box; or says which object a request ("the pills") means, or that it is not sure | Bedrock **Converse API** with the image (boto3), one call per picture, no retries | `eyereach/vision_nova.py` |
| **Nova 2 Lite** as agent model | The task agent's brain: plans, calls tools, writes notes | LangChain `init_chat_model("bedrock_converse:…")` (langchain-aws) | `agent/task_agent.py` |
| **Nova 2 Sonic** (`amazon.nova-2-sonic-v1:0`) | The app's voice, and free speech: with **tool use** it turns "a bit to the left" into a command from a fixed list and "bring me the banana" into a task; also a chat in `launch/talk.sh` | Bedrock **bidirectional streaming** with the `aws_sdk_bedrock_runtime` SDK (Python 3.12) | `agent/voice_sonic.py`, `launch/talk.sh` |

Not used on purpose: **Amazon Polly** and **Amazon Transcribe**. Sonic already listens and speaks, and our key is
Bedrock-only (they answer "access denied").

- **Nothing safety-critical needs the network.** STOP works offline (the gaze STOP button, or "Stop" to Vosk). If Bedrock
  can't be reached the app keeps running.
- **Nova suggests, the person decides.** The agent's `pick` refuses until the person confirmed that exact object.
- **Sonic never moves the arm itself.** It can only pick a command from the same fixed list as the offline phrases, and the
  app checks it like any spoken command. A fetch request only starts the agent, which still needs a "yes". Sonic stops
  listening while it speaks and never says "robot", so the offline listener can't mistake its replies for commands.
- **Few calls.** One Nova call per picture, no retries; one Sonic call per prompt (newer prompts replace waiting ones).

All Amazon models get a short project description from `eyereach/project_context.py` (be brief, ask when unsure, STOP always
wins, never claim something happened); Nova 2 Lite vision gets a shorter one, because the full one made it list nothing on a
real robot-table photo. Sonic calls `InvokeModelWithBidirectionalStream` and talks in the voice `matthew`.

## Control: the nose follows, eyes and mouth open to steer, close the mouth to pick up

This is the default (`launch/eyereach.sh`, or `--control tilt`). There is no setup step: the first time the app sees
your face it takes that as "straight", and from then on **the arm follows your nose live**, like a mouse.

| Do this | The arm |
|---|---|
| Look straight | goes to the middle of the table |
| Move your nose right / left (turn or tilt the head) | follows to your right / left; about 20° reaches the edge |
| Move your nose up / down | goes away from you / toward you |
| **Eyes and mouth open** (mouth open about 0.35 s) | turns on slow steering; closing either eye pauses it |
| **Close your mouth after steering** | stops, lowers, grabs once and lifts; if already holding, lets go instead |
| Lose the face | stops |

- **Pick-up:** after the mouth closes the app holds the position, waits 0.15 s, then runs one full pick-up (open, lower,
  close, lift). Keeping the mouth closed does not repeat it. Face loss, a pause or switching to voice cannot trigger
  one. Once started it finishes, but STOP and arm faults still interrupt it. The screen shows **Holding**. Open your eyes
  and mouth to carry the object; close the mouth again to let go. LET GO and a voice "drop" work too.
- **The nose is the guide:** the reference is the point between your eyes, and the arm follows the nose tip compared with
  where it sat when you looked straight (in eye-widths), so it reads the same wherever your face is in the picture. (A 3D
  head angle read 41° "straight" with the webcam off to the side, so we dropped it.)
- **Speed:** capped at 18.75 mm/s (11.25 while carrying); readings are smoothed and under 1.5° counts as straight.
  **FIX MY AIM** (or **C**) takes "straight" again from where your nose is now.
- **On screen:** your face big in the middle, mirrored, with LEFT / RIGHT / AWAY / TOWARD YOU around it (the one the arm heads
  for lights up), an arrow, a cross between the eyes, a dot on the nose tip, a green ring for "straight" and a **nose joystick**.
- Blinking does nothing. "Stop" by voice always works. Check your head without the arm:
  `~/eyereach_venv/bin/python -m eyereach.head_tilt` (settings are at the top of that file).
- **"I can't see your face" when it was there:** MediaPipe found the face in 98% of frames; other checks caused it and the
  face was low in the picture. The app now warns the helper near an edge, keeps a turned face longer and ignores drop-outs
  under 0.25 s. **Put the camera so the face is in the middle of the picture.**

## Talking to it: just say it

No fixed phrases and no "robot" first: the app finds the key word anywhere in what you say.

| Say (in any sentence) | It does |
|---|---|
| **voice** | voice mode: face tracking stops completely, only speech moves the arm |
| **vision** | face mode: the arm follows your nose; speech is ignored except "voice" and "stop" |
| **stop** | stops at once, in either mode |
| **start** ("go on", "continue") | carries on after a stop |
| left / right | moves 50 mm (5 cm); "a bit" or "a little" moves 5 mm |
| closer, toward me, back / away, forward | moves 20 mm toward / away from you; "a little" moves 5 mm |
| grab, pick up, take / drop, let go, put down, release | picks up / lets go |
| bring me / find / get / grab + a thing | the task agent fetches it ("bring me the banana"); "pick it up" alone just closes the gripper |
| yes / no | answers "Is it the banana?" |

"Can you move it a bit to the left please" is a small step left. "Don't move left", questions and small talk do nothing.
The old "Robot ..." phrases still work but are never needed. Voice moves are bounded steps at up to 80 mm/s, within the
arm's joint and workspace limits.

**Face and voice share the arm.** Speaking a command switches to voice and the face stops moving the arm. After 8 quiet
seconds the face takes over again, from where the arm is, so it doesn't jump. Choosing voice on purpose ("use my voice" or
USE VOICE) keeps it in voice until you say "use my face". STOP is never undone by waiting.

**Only necessary speech (default):** the robot stays quiet for normal moves. It speaks object questions, mode changes, stop
and error notices, task results and calibration prompts, as one short sentence (at most 40 characters) cut off after
**3 seconds**. Repeated messages are held back for 10 seconds. The screen always shows the instructions.

Change it with `--speech necessary|off|all` (`off` is silent but still listens; `all` adds short routine replies) and
`--speech-max-seconds 2` (0.5 to 10). `SPEECH` and `SPEECH_MAX_SECONDS` are also at the top of the launcher; `--quiet` means silent.

**Who listens:** Nova 2 Sonic (when the AWS keys are there) writes down what you say word for word and the app finds the
command in it (`voice_commands.understand`). Offline Vosk listens for the same key words (a 74-word vocabulary, any order)
and always handles "stop", even without the network; it handles the rest only when Sonic isn't available, so a command
never arrives twice. While the app speaks, what is heard is ignored.

**Offline voice setup:** run `bash launch/setup_voice.sh` once (see [Jetson voice setup](docs/jetson-voice.md)); the launcher
then turns voice on by itself (`--no-voice` keeps vision only; try `launch/eyereach.sh --no-arm --voice`). When the agent asks "Is it the banana?" say "Robot yes" /
"Robot no" (with Sonic just "yes", "that's it", "not that one"); STOP counts as no. When a command doesn't work, the screen
shows what was heard and why it was ignored, e.g. `Heard: robot move left - not sure (0.62, needs 0.70)` (words must reach
0.70 confidence; 0.80 dropped real commands). Practise without the robot:
`~/eyereach_venv/bin/python -m eyereach.voice_commands`.

## Park mode: return and drop after each pickup

Park mode turns on by itself whenever a destination is saved. After a pick-up the arm carries the object at travel height to
the saved X/Y spot, lowers, opens the gripper and lifts. Face steering is paused meanwhile; STOP still interrupts it. Agent
pick-ups use it too. Use `--no-park` for manual carrying.

**Save the destination:** start the app, move the empty gripper above the tray or drop spot, and have the helper press
**P**. The app holds the arm, saves the measured X/Y to `~/.eyereach/park.json` (release height 50 mm above the table by
default) and turns park on. Then press START or say "vision". Press P only while the gripper is empty and the arm isn't
picking up, releasing or fetching.

```bash
./launch/eyereach.sh                                        # next time: the saved destination is used automatically
./launch/eyereach.sh --park-position 162 20 --park-drop-mm 50   # or enter coordinates for one session
```

X/Y are millimetres in the robot's frame (X forward, positive Y to its left), not camera pixels; the example is only an
example. `--park-drop-mm` is the release height (minimum 30 mm; start with it before pressing P to save a different one).
`--park` still works and needs a saved destination. Reachability is checked. Park/drop works on the real arm.

## Two cameras: the laptop watches your face, the USB camera watches the robot

The screen shows both live: **ROBOT** (the Jetson's USB camera on the arm and table) and **YOU** (the laptop's webcam).

1. **Laptop, once:** `pip install opencv-python`, then `scp jetson@192.168.55.1:~/Documents/hack_princeton/tools/laptop_camera.py .`
2. **Laptop, each time:** `python3 laptop_camera.py` (on a Mac, allow camera access for Terminal). It streams on port 8080.
3. **Jetson:** `launch/eyereach.sh` as usual. `FACE_CAMERA` (top of the launcher) is `http://192.168.55.100:8080/stream.mjpg`
   (the laptop's address on the USB cable); `TABLE_CAMERA` is `/dev/video0`. Either can start first; the face picture says
   "Waiting for the laptop camera" and connects by itself. `FACE_CAMERA=/dev/video0` (and `TABLE_CAMERA=""`) goes back to one USB camera.

- Only the newest picture is used, so the face stays live (measured on the Jetson: 19.8 fresh pictures a second from a
  20-a-second stream, connecting in 0.08 s). The app saves the robot camera's newest picture every half second to
  `~/.eyereach/table_camera.jpg` for the task agent, which can't open the camera while the app has it.
- Once the table camera is calibrated (`launch/eyereach.sh --agent calibrate --camera /dev/video0`), the ROBOT picture shows a
  blue **PLAYGROUND** circle (13 cm across, 15 cm in front of the base, all within reach: put things in it) and
  **Nova 2 Lite's boxes**, asked every 4 seconds: green "can grab", yellow "reachable", red "out of reach", with a dot where
  the arm would go (`eyereach/table_view.py`; `--no-boxes` turns them off).

## The task agent: "bring me the banana"

`agent/task_agent.py` takes a task like "bring me the pills" and plans it: a perception sub-agent finds the object, the person
confirms it, the arm picks and drops it, and a caregiver sub-agent writes a message. It keeps a notes file
(`~/.eyereach/agent/notes.md`, e.g. "medicine: the pill bottle, next to the cup") and reads it later. Nothing moves until the
person confirms that exact object: `pick` refuses otherwise, whatever the model says.

Without `--arm` the arm is pretend. With `--arm`, `pick` runs `brain/fetch.py` in the arm's Python (ROS needs 3.10, the agent
3.12): it glides over the object, grabs it, carries it next to the person and lets go. The model is Nova 2 Lite (`AGENT_MODEL_ID`
changes it, `EYEREACH_NOTES` moves the notes); the notify tool just prints.

```bash
python3 -m agent.task_agent "bring me the pills"                     # needs Python 3.11+ (Jetson: setup step 9)
python3 -m agent.task_agent "bring me the snack" --photo table.jpg   # objects found by Nova in a photo
python3 -m agent.task_agent "bring me the pills" --arm virtual       # the pretend arm really moves (no ROS)
launch/eyereach.sh --agent calibrate --camera /dev/video2            # real arm, once (saved in ~/.eyereach/table_camera.json)
launch/eyereach.sh --agent "grab me a banana" --camera /dev/video2   # add --sim or --no-arm to try it first
```

The real arm needs a second camera looking down at the table and one calibration: the arm visits 10 nearby spots (a patch about
5 cm across near the middle of its reach) and you click the table right under the fingertips at each. On its own you answer
y/n in the terminal. For flat things like a banana try `--grab-mm 30`. The gripper's guess at whether it holds something is
not trusted (on the real arm it said "nothing" while holding), so the arm always carries on to the person.

### Both at once: face control, voice and the agent (two cameras)

Set `TABLE_CAMERA=/dev/video2` at the top of `launch/eyereach.sh` (or add `--table-camera /dev/video2`), calibrate the table
camera once, then run `launch/eyereach.sh`. It needs voice commands and Nova 2 Sonic on.

- **The face moves the arm** while nobody is talking. **Say "voice"** to talk to it and **"vision"** to go back (see
  [Talking to it](#talking-to-it-just-say-it)); "bring me the banana" works in voice mode.
- **"Bring me the banana"** (any named thing): Sonic hands it to the task agent, which takes a table picture, asks Nova which
  object it is and asks on screen and out loud "Is it the banana? Say yes or no". On yes, the app's own arm glides there, grabs it,
  carries it to the person and lets go, one step per frame, so the camera, the gaze STOP button and voice STOP keep working.
  STOP during a question counts as no.
- The agent is its own program (`eyereach/agent_link.py` starts `agent.task_agent --serve` in the Python 3.12 environment); the
  app keeps the arm and only answers its questions.

## How it works

```
eyereach/run_webcam.py     webcam -> face direction (or iris gaze) -> a point on the table map (gaze_webcam.py)
   ↓  eyereach/eye_control.py: confirm, grab, hold, drop
brain/arm_link.py          follow a table point, grab, release (or a pretend arm)
   ↓
brain/arm.py               inverse kinematics in mm, joint limits, never below the table
   ↓
gestures/gestures.py       sends the joint moves over ROS 2 -> arm_controller -> dynamixel_hardware_interface
   ↓  USB /dev/ttyACM0     (started by hardware.launch.py)
OpenCR (usb_to_dxl)        passes commands to the servos, which report their angles back (/joint_states)

microphone -> Vosk (offline, exact phrases, STOP)  ┐
           -> Nova 2 Sonic (free speech, tools)    ┴-> eye_control.py: the same checks for every command
"bring me the banana" -> eyereach/agent_link.py -> agent/task_agent.py --serve (Python 3.12, deepagents)
   table picture -> Nova 2 Lite: which object, where -> "is it the banana?" -> yes
   -> brain/fetch.py FetchJob: glide, grab, carry to the person, let go (the app's own arm, one step a frame)
```

After a helper starts it, nothing needs a keyboard, and the screen always says what to do next (`spd-say` speaks if voice is off).

### Pointer mode, calibration and buttons

The first version moved a screen pointer from the irises only; it jumped (calibration error 4.5–7% of the screen width, up to 200 px
at the corners). Now the pointer comes from where the face points (`--pointer head`, the default): the eye corners, nose tip,
forehead and chin give the face's direction, drawn as a ray out of the nose and steadied by a One Euro filter. `--pointer eyes` is
for people who can't move their head; `--pointer both` uses the head for the rough spot and the eyes to fine-tune (accuracy still
needs measuring on people). Pointer mode (`--control point`) needs a calibration, because the webcam sees which way the face turns,
not what that points at on the screen. It starts by itself once your face has been steady for 3 seconds (about 40 s): a yellow dot
glides around 12 spots and you point your nose (or, with `--pointer eyes`, your eyes) at it. It waits if the face is lost, tries once
more if the result is poor (over 7% of the screen width), and is saved per pointer mode and screen size. The arm then follows the
pointer live at the same speed limits; looking at the face panel or buttons, closing an eye or the mouth, losing tracking or pausing
stops new steering, and STOP holds until you look at MOVE AGAIN (`--movement dwell` is the older destination mode).

- **Pick-up details:** the fingertips travel about 120 mm above the table. Pick-up opens the gripper and waits 1 s, lowers to 50 mm,
  closes fully, waits 1 s, then lifts. **The gripper is open the rest of the time** (it opens first at start-up and again before the
  arm stops) and **stays closed after a pick-up until the next mouth closure, LET GO or a voice drop**. (The finger position is too
  unreliable to tell if something was caught, so it is only logged.)
- **Buttons:** PICK UP / LET GO, STOP / START, FIX MY AIM, USE VOICE / USE VISION, by gaze (0.65 s, set with `--dwell`) or click.
  **QUIETER / LOUDER** under the face picture change the system volume 10% per press (`pactl`, shared with `launch/talk.sh`); the helper
  can press **+** and **-**, or run `python3 -m eyereach.speaker_volume 70`.
- **Helper keys:** Esc closes, C calibrates again, Space pauses, V hides the face picture, P saves the park spot. Options:
  `--pointer head|eyes|both`, `--dwell 2`, `--dot-seconds 3`, `--quiet`, `--reuse`, `--windowed`, `--grab-mm 50` (never below 30),
  `--travel-mm 120`. Heights assume the base sits on the table; on a plate, put its thickness in `MOUNT_HEIGHT` in `brain/arm.py`.
- **Logging:** every session goes to `~/.eyereach/sessions/` (calibration error, moves, pick-ups, time per pick-and-place). Add
  `--person NAME`, `--screen-cm` and `--distance-cm` for errors in degrees; `python3 -m eyereach.session_log` sums them up.

## Talk to it (Amazon Nova 2 Sonic)

`launch/talk.sh` is a spoken chat with the robot through Nova 2 Sonic: speak into the microphone and it answers through the
speaker, with the same AWS keys. It only talks and never moves the arm. It runs in `~/eyereach_agent_venv` (Python 3.12,
installed by `launch/setup_jetson.sh`) and stops listening while it speaks.

```bash
launch/talk.sh                        # talk; Ctrl+C ends
launch/talk.sh --text "Hello"         # one typed sentence, reply saved to /tmp/eyereach_sonic_reply.wav (no mic/speaker)
launch/talk.sh --list                 # audio devices; pick with --microphone N / --speaker N
```

**In the app** (the launcher turns Sonic on when the keys and SDK are there; `--no-sonic` turns it off) Sonic's voice reads the
prompts instead of `spd-say`, and with voice commands on it also listens for free speech ("move it a bit to the left", "hold
on"). While the app talks, the offline microphone hears silence and anything Sonic hears (and for 2 s after) is dropped.

**Microphone: the webcam's first, the speaker's as the fallback** (`eyereach/audio_devices.py`). The C920 webcam heard the same
speech twice as loud, with word confidence 0.95 against 0.87. `python3 -m eyereach.audio_devices` shows the choice;
`EYEREACH_MIC=PnP` forces the speaker's; `pactl set-default-source` / `set-default-sink` pick devices.

## Test everything (one command)

```bash
launch/test_all.sh              # everything, about 2 minutes (2 small AWS calls)
launch/test_all.sh --offline    # no AWS calls      --no-sim: skip the simulation      --no-sound: no beep or voice
```

It **never moves the real arm** and ends with a PASS / FAIL / SKIP table (exit code 1 if anything failed): both Python
environments and the `.env` keys (values never printed); unit tests in both environments and the robot maths; the devices
(webcam frame, OpenCR, servos, gripper turn count, microphone, speaker beep); the arm's motion on the ROS simulation
(`tools/sim_check.py`: start-up opens the gripper, a mouth closure goes down, closes, lifts and holds, LET GO opens it); and one
Nova 2 Lite call and one Nova 2 Sonic reply on Bedrock. Then try the real arm (someone holds it): `launch/eyereach.sh`.

## Folders

```
eyereach/      the control: gaze_webcam.py (face direction, calibration, smoothing), eye_control.py (steering, mouth-close
               pick-up, buttons), run_webcam.py (the screen app), head_tilt.py (nose steering), voice_commands.py (Vosk),
               audio_devices.py, speaker_volume.py, vision_nova.py (Nova 2 Lite: names and boxes objects), session_log.py,
               agent_link.py (runs the task agent inside the app)
agent/         task_agent.py (the deepagents task agent; --serve runs it in the app), voice_sonic.py (Sonic: talk, the app's voice, free speech)
brain/         arm.py (robot maths; commands test, where, go, home), arm_link.py (follow a point, grab, release, or a pretend arm),
               fetch.py (one fetch for the agent, FetchJob steps it per frame; table camera calibration)
gestures/      gestures.py: the ROS 2 link that moves the joints
launch/        eyereach.sh, talk.sh, test_all.sh, setup_jetson.sh, setup_voice.sh, stop.sh, fix_turns.py
firmware/      servo_check: an Arduino test that stands the arm up and waves
tools/         laptop_camera.py (run ON THE LAPTOP), measure.py (gaze accuracy and grab trials), sim_check.py
tests/         python3 -m unittest discover -s tests
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

**Finding objects with Nova:** `eyereach/vision_nova.py` sends one camera image to Nova 2 Lite and gets back each object's name
and box. Keys come only from environment variables (never commit them): `AWS_BEARER_TOKEN_BEDROCK` (or `AWS_ACCESS_KEY_ID` and
`AWS_SECRET_ACCESS_KEY`), `AWS_REGION` (default `us-east-1`), `NOVA_MODEL_ID` (default `us.amazon.nova-2-lite-v1:0`). If Nova
can't be reached it logs a warning and returns no objects. The tests use saved answers in `tests/fixtures/`, so they need no network.

```bash
python3 -m eyereach.vision_nova table.jpg                # lists the objects and their boxes
python3 -m eyereach.vision_nova table.jpg "the pills"    # also says which object is the pills, or that it is not sure
python3 -m tools.measure accuracy --person NAME --screen-cm 29 --distance-cm 55
python3 -m tools.measure trials --person NAME
```

`accuracy` shows 9 test dots away from the calibration dots and prints the average and worst error in pixels, % of screen width and
degrees (`--recalibrate` forces a new calibration; measure `--screen-cm` and `--distance-cm` each time). `trials` is for a helper: Enter
when the person starts, then `y` if the grab worked or `n` if not. Both add rows to `results/measurements.csv`.

## Hardware

- OpenMANIPULATOR-X arm (5 DYNAMIXEL XM430-W350-T servos, payload 500 g, reach 380 mm) on a base plate
- OpenCR 1.0 board with its 12 V 5 A adapter and a micro USB data cable
- NVIDIA Jetson Orin NX 16GB, Ubuntu 22.04 (JetPack 6), ROS 2 Humble, and a webcam facing you
- A Mac or PC with the Arduino IDE, only to upload firmware to the OpenCR

Servos: 11 base (`joint1`), 12 shoulder (`joint2`), 13 elbow (`joint3`), 14 wrist (`joint4`), 15 gripper (`gripper`). They share
one daisy-chained TTL cable (1 Mbps, Protocol 2.0). 180° on the servo (2048 steps) is the centre and equals 0 rad in ROS.
ROBOTIS's default position gains are P/I/D 800/100/100; we use 400/0/0 for the base (step 5b) so it doesn't jitter.

- **Wiring:** keep the board's power switch OFF while connecting cables. The arm's white 3-pin cable goes to the first 3-pin TTL
  port on the bottom-left edge of the OpenCR (not the 4-pin ports). The 12 V adapter goes to the round jack. The OpenCR micro USB
  (right edge) goes to the Mac (for Arduino) or the Jetson (for ROS 2).
- **Mounting direction:** with the base servo at 0° the arm should face your work area (the base turns only about ±180° from
  there). Start the arm and run `ros2 topic echo /joint_states --once`: `joint1` should be close to 0 while the arm faces
  forward. If it is close to ±3.14 the arm is mounted backwards: turn off the 12 V, unscrew the base bracket, turn the **whole**
  arm 180° (hold it by the bottom servo) and screw it down again.
- **Power off = the arm slumps,** often past the joint limits. Before starting it again, hold it standing roughly straight up and
  keep holding it while you turn the 12 V on. The programs check this at start-up and refuse to move a joint that is out of range.

## One-time setup (on the Jetson)

**Quick way:** do step 1 on a Mac or PC, clone the repo (step 3), then run steps 2 and 4 to 9 with one script. It checks each
step first and skips what is done, so it is safe to run again. It asks for your password and never moves the arm. The steps
below are what it does, if you want to do them by hand.

```bash
bash launch/setup_jetson.sh            # add --voice for the offline voice commands
```

### 1. Upload the firmware to the OpenCR (on a Mac or PC)

The Arduino IDE can't upload to the OpenCR from the Jetson. Install the Arduino IDE, then:
1. Settings > Additional boards manager URLs:
   `https://raw.githubusercontent.com/ROBOTIS-GIT/OpenCR/master/arduino/opencr_release/package_opencr_index.json`
2. Boards Manager: install **OpenCR**.
3. Plug in the OpenCR and turn on its 12 V power. Tools > Board > OpenCR > OpenCR Board, and Tools > Port > the `usbmodem` port.
4. File > Examples > OpenCR > 10.Etc > **usb_to_dxl**, then Upload.

A good upload ends with `[OK] Download`. The arm won't move after this; that is expected, it now waits for the Jetson. If the
upload freezes, hold **SW2**, press **RESET**, release SW2 and upload again. To test the arm without ROS first, upload
`firmware/servo_check/servo_check.ino` instead (install the **Dynamixel2Arduino** library; it stands the arm up and waves the
wrist when you start it from the Serial Monitor at 115200 baud). Put `usb_to_dxl` back before using ROS 2.

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
mkdir -p ~/Documents && cd ~/Documents && git clone https://github.com/aditya-baniya-ai/hack_princeton.git
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

This takes a few minutes and should end with `Summary: 12 packages finished` (warnings are fine). If it freezes, press Ctrl+C and
run `colcon build --symlink-install --parallel-workers 1`.

### 5b. Calm the base servo (stops it jittering)

With ROBOTIS's default settings the base jitters about ±0.3° when the forearm sticks out. These commands give only the base
servo gentler settings (P 400, I 0, D 0), sent at every launch and not stored in the servo. Keep a backup first:

```bash
X=~/colcon_ws/src/open_manipulator/open_manipulator_x_description/ros2_control/open_manipulator_x_system.ros2_control.xacro
cp $X ~/omx_ros2_control_backup.xacro
sed -i '/<gpio name="dxl1">/,/<\/gpio>/ { s|"Position P Gain">800<|"Position P Gain">400<|; s|"Position I Gain">100<|"Position I Gain">0<|; s|"Position D Gain">100<|"Position D Gain">0<|; }' $X
sed -n '/<gpio name="dxl1">/,/<\/gpio>/p' $X | grep "Gain"      # should show P 400, I 0, D 0
cd ~/colcon_ws && colcon build --symlink-install --packages-select open_manipulator_x_description
```

To undo, copy the backup back and rebuild the same way. Repeat this step if you ever run `vcs import` again.

### 5c. Limit the gripper's squeeze (stops it overloading)

Without this the gripper squeezes at full current (about 3.2 A) when it closes on something, overloads within seconds and switches
itself off (every later grab then "holds nothing"). This caps it at about 0.8 A:

```bash
X=~/colcon_ws/src/open_manipulator/open_manipulator_x_description/ros2_control/open_manipulator_x_system.ros2_control.xacro
cp $X ~/omx_ros2_control_backup_before_gripper.xacro
sed -i '/<gpio name="dxl5">/,/<\/gpio>/ { /<param name="Return Delay Time">0<\/param>/a\        <param name="Operating Mode">5</param>\n        <param name="Goal Current">300</param>
}' $X
sed -n '/<gpio name="dxl5">/,/<\/gpio>/p' $X | grep param      # should list Operating Mode 5, Goal Current 300
cd ~/colcon_ws && colcon build --symlink-install --packages-select open_manipulator_x_description
```

If the gripper already overloaded (it won't move; `/tmp/robo_arm.log` says "Hardware error occurred" for ID 15), turn the 12 V off
and on again.

### 6. Set up the terminal and the USB rules

```bash
echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc
echo "source ~/colcon_ws/install/setup.bash" >> ~/.bashrc
source ~/.bashrc
ros2 run open_manipulator_x_bringup create_udev_rules      # asks for your password
```

### 7. Check the OpenCR is connected

Plug the OpenCR's USB into the Jetson and turn on the 12 V power. `ls /dev/ttyACM*` should print `/dev/ttyACM0`.

### 8. Install MediaPipe (the face tracker), once

It goes in its own environment, `~/eyereach_venv`, so nothing else on the Jetson changes; `launch/eyereach.sh` uses it automatically.

```bash
python3 -m venv --system-site-packages ~/eyereach_venv
python3 -c "import site; print(site.getusersitepackages())" > ~/eyereach_venv/lib/python3.10/site-packages/user_site.pth
~/eyereach_venv/bin/pip install --no-deps mediapipe==0.10.18
~/eyereach_venv/bin/pip install absl-py attrs flatbuffers "protobuf>=4.25.3,<5" sentencepiece
~/eyereach_venv/bin/python -c "import mediapipe, cv2, numpy; print('mediapipe', mediapipe.__version__, 'numpy', numpy.__version__)"
```

NumPy must still be 1.x. To undo all of it: `rm -rf ~/eyereach_venv`.

### 9. Python 3.12 for the task agent, once

The task agent uses the `deepagents` library, which needs Python 3.11 or newer, and the Jetson has 3.10. It gets its own environment,
`~/eyereach_agent_venv`; `uv` downloads Python 3.12 for it, so the system Python and ROS are not touched. The face tracker and the
arm stay in `~/eyereach_venv` (Python 3.10, because ROS 2 Humble needs it).

```bash
python3 -m pip install --user uv
export PATH="$HOME/.local/bin:$PATH"
uv venv --python 3.12 ~/eyereach_agent_venv
uv pip install --python ~/eyereach_agent_venv/bin/python "deepagents>=0.7" "langchain-aws>=1.0" boto3 numpy opencv-python-headless \
  "aws_sdk_bedrock_runtime>=0.1.0,<0.2.0" smithy-aws-core sounddevice   # Nova 2 Sonic
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

**One command runs everything that is connected:** `launch/eyereach.sh` loads `.env` (AWS keys), finds the screen, checks the
gripper and joints, starts the real arm, opens the gripper and runs the face control with the saved calibration. Voice commands join
in by themselves once `bash launch/setup_voice.sh` has been run and a microphone is plugged in (otherwise it says "Voice commands:
off"). Nova 2 Sonic joins in the same way when `.env` has the AWS keys. With `TABLE_CAMERA` set at the top of the launcher (or
`--table-camera /dev/video2`) the task agent runs inside the app: say "bring me the banana" (see
[both at once](#both-at-once-face-control-voice-and-the-agent-two-cameras)). Without the app: `--agent "grab me a banana"`.

**The AWS keys in `.env`:** one `NAME=value` per line, with **no spaces around `=`** (a space makes the shell drop the key;
`launch/test_all.sh` reports it as "no AWS_ACCESS_KEY_ID=... line"):

```
AWS_ACCESS_KEY_ID=AKIA...
AWS_SECRET_ACCESS_KEY=...
AWS_REGION=us-east-1
```

**Cameras:** `--camera` is the face camera and `--table-camera` the table camera (`v4l2-ctl --list-devices` shows their numbers); either can
be a stream address. A Mac's webcam over Wi-Fi (not yet tried; the picture arrives a little late, so the arm may trail the head):
`ffmpeg -f avfoundation -framerate 30 -video_size 640x480 -i "0" -f mpjpeg -q:v 5 -listen 1 http://0.0.0.0:8080/face.mjpg`, then
`launch/eyereach.sh --camera http://LAPTOP_IP:8080/face.mjpg`.

**Settings** are at the top of `launch/eyereach.sh`: it **reuses the saved calibration** (only the first run, or a new screen size or
pointer mode, calibrates; FIX MY AIM or C recalibrates any time); the fingertips **travel 120 mm** above the table and **go down to
50 mm** to pick up. Change `TRAVEL_MM`, `GRAB_MM` or `REUSE` there, or override one run with `--travel-mm 120`, `--grab-mm 50` or
`--calibrate`. Other options go to the app (`--camera /dev/video0`, `--movement dwell`, `--windowed`, `--dot-seconds 4`, `--dwell 1.2`).

**For the real arm:** turn on the 12 V power, make sure the OpenCR's USB is in the Jetson, **hold the arm** (it goes limp when it starts
and stops), keep the power switch within reach, then run it. If it stays on "Waiting for the arm", look in `/tmp/robo_arm.log`. Esc in
the app stops the arm too.

- **Start the arm by hand:** `ros2 launch open_manipulator_x_bringup hardware.launch.py port_name:=/dev/ttyACM0`, and wait for
  `Configured and activated arm_controller` (add `start_rviz:=true` to watch it in RViz). Joint angles: `ros2 topic echo /joint_states --once`.
- **Stop everything** (hold the arm first): `launch/stop.sh`. It should say `Everything is stopped.` To power down, press Ctrl+C on the
  arm launch while holding the arm, then turn off the 12 V.

### Check the robot maths and the arm

The point is the middle of the fingertips: **x** forward from the base, **y** to the left, **z** up from the table, in mm. Pointing down
it reaches about 100 mm high and 80 to 240 mm in front of the base, refuses any point it can't reach safely and never goes lower than
10 mm above the table.

```bash
python3 brain/arm.py test           # the maths only, no arm needed
python3 brain/arm.py where          # where the gripper is now, and the joint angles
python3 brain/arm.py go 200 0 100   # move there pointing down (add --level to point forward)
python3 brain/arm.py home
```

Try the simulation first. On the real arm, `go 200 0 100` should put the fingertips about 200 mm in front of the base and 100 mm up (check with a ruler).

## Problems

- **The gripper stays closed and never opens (every command "accepted"):** the gripper servo counts whole turns and had picked up an extra
  one (it read 5168 with the fingers open; ROS expects about 1057 open and 2648 closed), so "open" drove it closed. Restarting the servo
  resets the count. `launch/fix_turns.py` (run by `launch/eyereach.sh` before every real-arm start) checks this and restarts the servo if
  needed; `python3 launch/fix_turns.py --check` shows the reading. If it stays off, turn the 12 V off and on.
- **"could not connect to display" / "Qt platform plugin xcb":** it was started from a terminal with no screen (an SSH terminal has an empty
  `DISPLAY`). The launcher now uses the desktop you're logged into (the Jetson's monitor or Remote Desktop) and refuses to start the arm if
  there is none. To choose one: `DISPLAY=:10 launch/eyereach.sh` (`ls /tmp/.X11-unix` lists them; X10 is usually Remote Desktop).
- **Permission denied on /dev/ttyACM0:** your user isn't in the `dialout` group. Run the `usermod` line in step 4, then log out and back in.
- **The arm doesn't move after uploading usb_to_dxl:** normal. Start it with ROS 2. **The upload freezes:** hold SW2, press RESET, release SW2.
- **"The arm is outside its safe range, so it won't be moved":** the arm slumped with the power off, or the base is at its end stop because it
  is mounted backwards. Hold the arm, turn the 12 V off, stand it straight up facing forward, keep holding it while you turn the 12 V on, and
  start again. `joint1`, `joint2`, `joint3` should be close to 0.
- **The arm (and RViz) keeps shaking:** two programs are controlling the arm (for example a leftover simulation plus the real arm). Hold the
  arm, run `launch/stop.sh`, and start just one thing.
- **The arm stays stiff after everything is stopped:** the servos hold their last position until told to let go. Hold the arm, turn the 12 V
  off and back on before the next launch.
- **The base jitters (about ±1°) while holding a pose:** do step 5b.
- **"System throttled due to over-current":** the Jetson is briefly slowing itself down. Use the power adapter that came with the kit and the
  25 W mode (`sudo nvpmodel -m 3` on the Orin NX 16GB, it asks to reboot).
- **A servo ignores a target near 0° or 360°** (Arduino test): keep targets inside 0 to 360. **Nodes from another computer show up in
  `ros2 node list`:** use a private network or set a different `ROS_DOMAIN_ID`.
- **"Nova 2 Sonic: off" or Nova finds nothing:** the keys didn't load. Check `.env` has `NAME=value` lines with no spaces around `=`, and that
  `~/eyereach_agent_venv` has the Sonic SDK (`bash launch/setup_jetson.sh`).
- **"Task agent: off":** it needs voice commands and Sonic on, and on the real arm the table camera calibration
  (`launch/eyereach.sh --agent calibrate --camera /dev/video2`). If you move the table camera the arm reaches for the wrong spot: calibrate again.
- **The arm grabs a little off the object:** calibrate the table camera again, clicking exactly under the fingertips; for flat things lower
  `GRAB_MM` (never below 30).
- **`fake.launch.py` refuses to run on the Jetson:** use the `base.launch.py ... use_fake_hardware:=true` command that `launch/eyereach.sh --sim` runs.

