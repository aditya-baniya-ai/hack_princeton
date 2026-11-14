#!/usr/bin/env bash
# Start the arm, then EyeReach: look at the map on the screen and the arm follows your gaze.
#
#   launch/eyereach.sh              real arm
#   launch/eyereach.sh --sim        simulated arm in RViz
#   launch/eyereach.sh --no-arm     pretend arm, no ROS (works on any computer)
#
# Other options go to the app, e.g. --camera /dev/video0, --reuse, --fullscreen.
# The arm runs in the background (log: /tmp/robo_arm.log). Esc in the app (or Ctrl+C here) stops
# everything. On the real arm, hold it first: it goes limp when it starts and stops.
set -m   # the arm runs in its own process group, so it can be stopped cleanly
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(dirname "$HERE")"
SIM=false
NO_ARM=false
APP_ARGS=()
for arg in "$@"; do
  case "$arg" in
    --sim) SIM=true ;;
    --no-arm) NO_ARM=true ;;
    *) APP_ARGS+=("$arg") ;;
  esac
done

# MediaPipe lives in its own environment on the Jetson (see the README), if it's set up
PYTHON=python3
[ -x "$HOME/game_venv/bin/python" ] && PYTHON="$HOME/game_venv/bin/python"
cd "$REPO"

if [ "$NO_ARM" = true ]; then
  exec "$PYTHON" -m eyereach.run_webcam "${APP_ARGS[@]}"
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
"$PYTHON" -m eyereach.run_webcam --real "${APP_ARGS[@]}"
