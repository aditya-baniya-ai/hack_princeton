#!/usr/bin/env bash
# Start the arm, then EyeReach: look at the map on the screen and the arm follows your gaze.
#
#   launch/eyereach.sh              real arm
#   launch/eyereach.sh --sim        simulated arm in RViz
#   launch/eyereach.sh --no-arm     pretend arm, no ROS (works on any computer)
#
#   launch/eyereach.sh --calibrate  do a new calibration instead of reusing the saved one
#
#   launch/eyereach.sh --agent "grab me a banana" --camera /dev/video2
#                                   the task agent instead of the eye app: Nova finds it with the table
#                                   camera, you answer y/n, the arm brings it (add --sim or --no-arm too).
#                                   Once first: --agent calibrate --camera /dev/video2 (click 4 spots).
#
# Other options go to the app, e.g. --camera /dev/video0, --windowed, --travel-mm 120 (they override
# the settings below). The arm runs in the background (log: /tmp/robo_arm.log). Esc in the app (or
# Ctrl+C here) stops everything. On the real arm, hold it first: it goes limp when it starts and stops.

# ---- settings: change them here ----
VOICE=true      # offline voice commands, used only once launch/setup_voice.sh is done; --no-voice turns off
SONIC=true      # Amazon Nova 2 Sonic: its voice for the spoken prompts, and free speech ("a bit to the left")
                # with the voice commands; used only with AWS keys in .env and ~/eyereach_agent_venv; --no-sonic turns off
TRAVEL_MM=120   # fingertips this high above the table (mm) at start-up and while moving
GRAB_MM=50      # fingertips this high above the table (mm) when picking up (the app never goes below 30)
REUSE=true      # reuse the saved calibration (the first run, or a new screen size, still calibrates);
                # recalibrate any time with FIX MY AIM or C in the app, or start with --calibrate
# -------------------------------------

set -m   # the arm runs in its own process group, so it can be stopped cleanly
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(dirname "$HERE")"
SIM=false
NO_ARM=false
AGENT_TASK=""
NEXT_IS_TASK=false
APP_ARGS=(--travel-mm "$TRAVEL_MM" --grab-mm "$GRAB_MM")   # given first, so options typed after win
for arg in "$@"; do
  if [ "$NEXT_IS_TASK" = true ]; then AGENT_TASK="$arg"; NEXT_IS_TASK=false; continue; fi
  case "$arg" in
    --agent) NEXT_IS_TASK=true ;;
    --sim) SIM=true ;;
    --no-arm) NO_ARM=true ;;
    --calibrate) REUSE=false ;;
    --reuse) REUSE=true ;;
    --voice) VOICE=true ;;
    --no-voice) VOICE=false ;;
    --sonic) SONIC=true ;;
    --no-sonic) SONIC=false ;;
    *) APP_ARGS+=("$arg") ;;
  esac
done
[ "$REUSE" = true ] && APP_ARGS+=(--reuse)

# Voice only when it is installed: asking the app for voice without it makes it press STOP at start.
VOICE_MODEL_DIR="$HOME/.eyereach/models/vosk-model-small-en-us-0.15"
VOICE_PY=python3
[ -x "$HOME/eyereach_venv/bin/python" ] && VOICE_PY="$HOME/eyereach_venv/bin/python"
VOICE_ON=false
if [ "$VOICE" = true ]; then
  if [ -d "$VOICE_MODEL_DIR" ] && "$VOICE_PY" -c "import vosk, sounddevice" > /dev/null 2>&1; then
    APP_ARGS+=(--voice)
    VOICE_ON=true
    echo "Voice commands: on"
  else
    echo "Voice commands: off (not installed: run 'bash launch/setup_voice.sh' once, and plug in a microphone)"
  fi
fi

# Keys and settings (AWS for Nova and the task agent) from the repo's .env, if there is one.
# .env is git-ignored: never commit it.
if [ -f "$REPO/.env" ]; then
  set -a; . "$REPO/.env"; set +a
  echo "Loaded settings from .env"
fi

# Sonic only when it can work: keys, and its SDK in the agent's Python 3.12
if [ "$SONIC" = true ]; then
  if [ -n "${AWS_ACCESS_KEY_ID:-}" ] && "$HOME/eyereach_agent_venv/bin/python" -c "import aws_sdk_bedrock_runtime, sounddevice" > /dev/null 2>&1; then
    APP_ARGS+=(--sonic-voice)
    [ "$VOICE_ON" = true ] && APP_ARGS+=(--sonic)
    echo "Nova 2 Sonic: on (voice$([ "$VOICE_ON" = true ] && echo ", free speech"))"
  else
    echo "Nova 2 Sonic: off (needs AWS keys in .env and its SDK: see launch/talk.sh)"
  fi
fi

# MediaPipe lives in its own environment on the Jetson (see the README), if it's set up
PYTHON=python3
[ -x "$HOME/eyereach_venv/bin/python" ] && PYTHON="$HOME/eyereach_venv/bin/python"
cd "$REPO"

# The app needs a screen. In an SSH terminal DISPLAY is empty, so use a desktop this user is logged
# into (like a Remote Desktop session). Checked before the arm starts, so a missing screen never
# starts and then drops the arm.
if [ -z "$DISPLAY" ]; then
  for socket in /tmp/.X11-unix/X*; do
    [ -e "$socket" ] || continue
    if DISPLAY=":${socket##*/X}" xdpyinfo > /dev/null 2>&1; then
      export DISPLAY=":${socket##*/X}"
      echo "Showing EyeReach on display $DISPLAY (the desktop you're logged into)."
      break
    fi
  done
fi
if ! xdpyinfo > /dev/null 2>&1; then
  echo "No screen to show EyeReach on (DISPLAY='$DISPLAY'). Nothing was started."
  echo "Log in to the Jetson's desktop (on its monitor or by Remote Desktop), or run this from a"
  echo "terminal on that desktop, or set it yourself, e.g.:  DISPLAY=:10 $0 $*"
  exit 1
fi

# --agent: the task agent (Python 3.12, its own environment) instead of the eye app
run_app() {   # $1: real or virtual
  if [ -z "$AGENT_TASK" ]; then
    if [ "$1" = virtual ]; then "$PYTHON" -m eyereach.run_webcam "${APP_ARGS[@]}"; else "$PYTHON" -m eyereach.run_webcam --real "${APP_ARGS[@]}"; fi
    return
  fi
  local camera=() rest=() a prev=""
  for a in "${APP_ARGS[@]}"; do   # keep only what the agent understands
    case "$prev" in --camera) camera=(--camera "$a") ;; --grab-mm|--travel-mm) rest+=("$prev" "$a") ;; esac
    prev="$a"
  done
  if [ "$AGENT_TASK" = calibrate ]; then
    local flags=(); [ "$1" = virtual ] && flags=(--virtual)
    "$PYTHON" brain/fetch.py calibrate "${camera[@]}" "${flags[@]}" "${rest[@]}"
  else
    local agent_py="$HOME/eyereach_agent_venv/bin/python"; [ -x "$agent_py" ] || agent_py=python3
    EYEREACH_ARM_PYTHON="$PYTHON" "$agent_py" -m agent.task_agent "$AGENT_TASK" --arm "$1" "${camera[@]}" "${rest[@]}"
  fi
}

if [ "$NO_ARM" = true ]; then
  run_app virtual
  exit
fi

# Two arm programs at once fight each other (and only one can open the OpenCR's USB port),
# so refuse to start a second.
if pgrep -f "open_manipulator_x_bringup|ros2_control_node" > /dev/null ||
   ros2 node list 2>/dev/null | grep -q "^/controller_manager$"; then
  echo "An arm (real or simulated) is already running. Stop it first:"
  echo "  $HERE/stop.sh"
  echo "If it's the real arm, hold it first: it may go limp."
  exit 1
fi

# If a joint reads a whole turn off (like the base saying 353° when it faces forward),
# fix_turns.py corrects it so the safe-range check passes. Real arm only.
if [ "$SIM" = false ] && [ -e /dev/ttyACM0 ]; then
  /usr/bin/python3 "$HERE/fix_turns.py" --port /dev/ttyACM0
fi

ARM_GROUP=""
stop_arm() {
  [ -z "$ARM_GROUP" ] && return
  echo "Stopping the arm ..."
  kill -INT -- -"$ARM_GROUP" 2>/dev/null
  for _ in $(seq 100); do
    kill -0 -- -"$ARM_GROUP" 2>/dev/null || return
    sleep 0.1
  done
  kill -KILL -- -"$ARM_GROUP" 2>/dev/null
}
trap stop_arm EXIT

if [ "$SIM" = true ]; then
  echo "Starting the simulated arm (RViz will open) ...   log: /tmp/robo_arm.log"
  ros2 launch open_manipulator_x_bringup base.launch.py use_sim:=false \
    use_fake_hardware:=true fake_sensor_commands:=true start_rviz:=true > /tmp/robo_arm.log 2>&1 &
else
  echo "Starting the real arm ...   log: /tmp/robo_arm.log"
  ros2 launch open_manipulator_x_bringup hardware.launch.py port_name:=/dev/ttyACM0 \
    > /tmp/robo_arm.log 2>&1 &
fi
ARM_GROUP=$!
echo "If it stays on 'Waiting for the arm', look in /tmp/robo_arm.log for errors."
run_app real
