# aHandForMyAunt

[![tests](https://github.com/aditya-baniya-ai/hack_princeton/actions/workflows/tests.yml/badge.svg)](https://github.com/aditya-baniya-ai/hack_princeton/actions/workflows/tests.yml)

**A robot arm for people who can't use their hands, controlled with the eyes, the nose, the mouth and the voice.**
Built at HackPrinceton Fall 2026 (Nov 13-15) for a teammate's aunt, who is paralyzed from the neck down and still
moves her face, eyes and mouth. Everything below works on the real arm, tested.

> A research prototype, not a medical device. Not yet tested with people who have disabilities.

## What it does

- **Eyes and nose steer.** We track the eyes to control the robot (both open means "steer"). Tracking the eyes alone was
  jumpy, so we added the **nose** for stability: with eyes and mouth open, the arm follows your nose and eyes, slowly and live.
- **The mouth grabs.** Close your mouth to pick up, close it again to let go. With a park spot saved, the arm
  carries the object there, lowers, lets go and lifts by itself.
- **Voice.** Say "a little left", "closer", "grab". "Stop" always works, even offline. Speaking takes over from the
  face; a few quiet seconds give it back.
- **Ask for a thing.** Say "bring me the banana". A camera over the table and Amazon Nova find it, the screen and a
  voice ask "Is it the banana?", and only after your "yes" does the arm bring it to you.

## How it works

```
face camera -> MediaPipe Face Mesh -> eyes / nose / mouth -> checks -> arm
microphone  -> offline Vosk (STOP) and Nova 2 Sonic (free speech) -> checks -> arm
table camera -> Amazon Nova 2 Lite finds the object -> you confirm -> arm fetches it
```

Every move is checked locally (reach, speed, 12 cm above the table, table clearance) before it reaches the arm, and
STOP never depends on the network. Nova only suggests: the person decides.

| Part | What we used |
|---|---|
| Arm and computer | ROBOTIS OpenMANIPULATOR-X, OpenCR board, NVIDIA Jetson Orin NX, ROS 2 Humble |
| Cameras | one on the face (a laptop webcam or USB camera), one over the table |
| Face tracking | MediaPipe Face Mesh, our own arm maths (`brain/arm.py`) |
| Voice | offline Vosk for commands and STOP, Amazon Nova 2 Sonic for natural speech and spoken prompts |
| Finding objects | Amazon Nova 2 Lite on Bedrock, one call per picture; a table camera calibration turns "it's here in the picture" into arm moves |
| Tests | over 200 automated tests, run on GitHub for every change |

## Run it

Start with the pretend arm, then the simulated one, then the real one:

```bash
launch/eyereach.sh --no-arm      # pretend arm, no ROS: works on any computer
launch/eyereach.sh --sim         # simulated arm in RViz
launch/eyereach.sh               # the real arm (hold it first: it goes limp when it starts and stops)
```

Setup on a fresh Jetson, in one script:

```bash
bash launch/setup_jetson.sh --voice
```

The AWS keys for Nova go in `.env` (git-ignored), one `NAME=value` per line with **no spaces around `=`**:
`AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_REGION=us-east-1`. Without them the arm still works by face and
offline voice.

```bash
launch/test_all.sh               # every check; never moves the real arm
```

## Status

| Part | State |
|---|---|
| Eye and nose steering, mouth pick-up and release | works on the real arm, tested |
| Voice commands, Nova 2 Sonic speech | works on the real arm, tested |
| "Bring me the banana": find, confirm, fetch, park | works on the real arm, tested |
| Accuracy and success rates | not measured yet |
| Testing with intended users | not done yet |

Next: record repeated real-arm trials with success counts and timing, and test with people who have disabilities.

## More

- **[Full guide](docs/full-guide.md):** the build story, hardware wiring, setup step by step, every control, park mode,
  two cameras, voice, measuring it, and the problems we hit and fixed.
- The code's folders and settings keep the internal name `eyereach` (`eyereach/` is the app, `brain/` the arm, `agent/`
  the Nova task agent and voice).
- Built with purchased hardware, upstream ROBOTIS and ROS 2 packages, MediaPipe, Vosk and Amazon Nova; everything that
  connects them is ours.
