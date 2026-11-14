#!/usr/bin/env bash
# Test everything in one go. It NEVER moves the real arm: the arm's motion is checked on the ROS
# simulation (fake hardware), and the real servos are only read.
#
#   launch/test_all.sh              everything (2 small AWS calls: one Nova 2 Lite, one Nova 2 Sonic)
#   launch/test_all.sh --offline    no AWS calls
#   launch/test_all.sh --no-sim     skip the ROS simulation (about 40 s)
#   launch/test_all.sh --no-sound   no tone or spoken reply on the speaker
#
# Ends with a PASS / FAIL / SKIP table; exit code 1 if anything failed. Then try the real arm (hold it):
#   launch/eyereach.sh
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(dirname "$HERE")"
EYE="$HOME/eyereach_venv/bin/python"
AGENT="$HOME/eyereach_agent_venv/bin/python"
WS="$HOME/colcon_ws"
LOGS="$(mktemp -d /tmp/eyereach_test.XXXX)"
CLOUD=true SIM=true SOUND=true
for arg in "$@"; do
  case "$arg" in
    --offline) CLOUD=false ;;
    --no-sim) SIM=false ;;
    --no-sound) SOUND=false ;;
    *) echo "Unknown option: $arg (see the top of $0)"; exit 2 ;;
  esac
done
cd "$REPO"
RESULTS=()
record() { RESULTS+=("$1|$2|$3"); printf "  %-5s %s%s\n" "$1" "$2" "${3:+  ($3)}"; }
section() { echo; echo "=== $*"; }
ros_env() { set +u; . /opt/ros/humble/setup.bash; [ -f "$WS/install/setup.bash" ] && . "$WS/install/setup.bash"; }
arm_running() { pgrep -f "ros2_control_node|open_manipulator_x_bringup" > /dev/null; }

section "1. Environments and keys"
[ -x "$EYE" ] && record PASS "eye-tracker environment" "$("$EYE" --version 2>&1)" \
             || record FAIL "eye-tracker environment" "missing: run bash launch/setup_jetson.sh"
[ -x "$AGENT" ] && record PASS "agent environment" "$("$AGENT" --version 2>&1)" \
               || record FAIL "agent environment" "missing: run bash launch/setup_jetson.sh"
if [ -f .env ]; then
  set -a; . ./.env > /dev/null 2>&1; set +a     # values are never printed
  if [ -n "${AWS_ACCESS_KEY_ID:-}${AWS_BEARER_TOKEN_BEDROCK:-}" ]; then
    record PASS ".env loads AWS keys" "region ${AWS_REGION:-us-east-1}"
  else
    record FAIL ".env loads AWS keys" "no AWS_ACCESS_KEY_ID=... line; Amazon checks will be skipped"
    CLOUD=false
  fi
  git check-ignore -q .env && record PASS ".env is git-ignored" || record FAIL ".env is git-ignored" "add .env to .gitignore"
else
  record SKIP ".env" "no .env: Amazon checks skipped"
  CLOUD=false
fi

section "2. Unit tests and robot maths"
out=$("$EYE" -m unittest discover -s tests 2>&1 | grep -E "^(Ran|OK|FAILED)" | tr '\n' ' ')
[[ "$out" == *OK* ]] && record PASS "unit tests (eye-tracker environment)" "$out" || record FAIL "unit tests (eye-tracker environment)" "$out"
out=$("$AGENT" -m unittest tests.test_task_agent tests.test_voice_sonic 2>&1 | grep -E "^(Ran|OK|FAILED)" | tr '\n' ' ')
[[ "$out" == *OK* ]] && record PASS "unit tests (agent environment)" "$out" || record FAIL "unit tests (agent environment)" "$out"
out=$(python3 brain/arm.py test 2>&1 | tail -1)
[ "$out" = "All good." ] && record PASS "robot maths (brain/arm.py test)" || record FAIL "robot maths (brain/arm.py test)" "$out"

section "3. Devices"
out=$("$EYE" -c "
import cv2
c = cv2.VideoCapture('/dev/video0'); [c.read() for _ in range(10)]; ok, f = c.read(); c.release()
if ok: cv2.imwrite('$LOGS/photo.jpg', f)
print(f'{f.shape[1]}x{f.shape[0]}' if ok else 'no frame')" 2>/dev/null | tail -1)
[[ "$out" == *x* ]] && record PASS "webcam /dev/video0" "$out" || record FAIL "webcam /dev/video0" "$out"
if ls /dev/ttyACM0 > /dev/null 2>&1; then
  record PASS "OpenCR USB" "/dev/ttyACM0"
  if arm_running; then
    record SKIP "servos (read only)" "an arm program is running"
  else
    out=$(timeout 20 /usr/bin/python3 launch/fix_turns.py --check 2>&1)
    if echo "$out" | grep -q "Wrist"; then
      bad=$(echo "$out" | grep -vc "fine")
      [ "$bad" -eq 0 ] && record PASS "servos answer, angles and gripper fine (read only)" \
                       || record FAIL "servos (read only)" "$(echo "$out" | grep -v fine | head -2 | tr '\n' ' ')"
    else
      record FAIL "servos (read only)" "no answer: is the 12 V on? $(echo "$out" | tail -1)"
    fi
  fi
else
  record SKIP "OpenCR USB" "not plugged in (or 12 V off)"
fi
out=$("$EYE" -c "
import time, numpy as np, sounddevice as sd
from eyereach.audio_devices import pick_microphone
dev = pick_microphone(say=lambda s: print(s))
frames = []
with sd.InputStream(samplerate=16000, channels=1, dtype='int16', device=dev, callback=lambda d, n, t, s: frames.append(d.copy())):
    time.sleep(1.5)
a = np.concatenate(frames).astype(float)
print(f'level {np.sqrt(np.mean(a**2)):.0f}')" 2>&1 | tail -2 | tr '\n' ' ')
[[ "$out" == *level* && "$out" != *"level 0 "* ]] && record PASS "microphone records" "$out" || record FAIL "microphone records" "$out"
if [ "$SOUND" = true ]; then
  out=$("$AGENT" -c "
import numpy as np, sounddevice as sd
t = np.arange(int(24000 * 0.6)) / 24000
sd.play((0.3 * np.sin(2 * np.pi * 660 * t) * 32767).astype(np.int16), 24000); sd.wait()
print('played on', sd.query_devices(kind='output')['name'])" 2>&1 | tail -1)
  [[ "$out" == played* ]] && record PASS "speaker tone (did you hear a beep?)" "$out" || record FAIL "speaker tone" "$out"
else
  record SKIP "speaker tone" "--no-sound"
fi

section "4. Arm motion on the ROS simulation (fake hardware, the real arm does not move)"
if [ "$SIM" != true ]; then
  record SKIP "simulated pick-up" "--no-sim"
elif arm_running; then
  record SKIP "simulated pick-up" "an arm program is running: stop it first (launch/stop.sh)"
else
  ros_env
  ros2 launch open_manipulator_x_bringup base.launch.py use_sim:=false use_fake_hardware:=true \
    fake_sensor_commands:=true start_rviz:=false > "$LOGS/sim.log" 2>&1 &
  SIM_PID=$!
  for _ in $(seq 40); do grep -q "activated.*gripper_controller" "$LOGS/sim.log" && break; sleep 1; done
  if grep -q "activated.*gripper_controller" "$LOGS/sim.log"; then
    timeout 90 /usr/bin/python3 tools/sim_check.py 2>&1 | grep -E "^  (ok|FAIL)" | tee "$LOGS/sim_check.txt"
    fails=$(grep -c FAIL "$LOGS/sim_check.txt"); oks=$(grep -c "ok  " "$LOGS/sim_check.txt")
    [ "$fails" -eq 0 ] && [ "$oks" -gt 0 ] && record PASS "simulated pick-up" "$oks steps" \
                                          || record FAIL "simulated pick-up" "$fails failed, see above"
  else
    record FAIL "simulated pick-up" "the simulation didn't start: $LOGS/sim.log"
  fi
  kill -INT "$SIM_PID" 2> /dev/null; pkill -INT -f "use_fake_hardware:=true" 2> /dev/null
  sleep 3; pkill -f "ros2_control_node|robot_state_publisher|spawner" 2> /dev/null
fi

section "5. Amazon on Bedrock (Nova 2 Lite vision, Nova 2 Sonic voice)"
if [ "$CLOUD" != true ]; then
  record SKIP "Amazon Nova 2 Lite" "offline"
  record SKIP "Amazon Nova 2 Sonic" "offline"
else
  if [ -f "$LOGS/photo.jpg" ]; then
    timeout 60 "$EYE" -m eyereach.vision_nova "$LOGS/photo.jpg" > "$LOGS/nova.txt" 2>&1
    if grep -q "objects (" "$LOGS/nova.txt" && ! grep -qiE "warning|error|denied" "$LOGS/nova.txt"; then
      record PASS "Amazon Nova 2 Lite answers" "$(grep "objects (" "$LOGS/nova.txt" | head -1)"
    else
      record FAIL "Amazon Nova 2 Lite answers" "$(grep -iE "warning|error|denied" "$LOGS/nova.txt" | grep -v AKIA | head -1 | cut -c1-120)"
    fi
  else
    record SKIP "Amazon Nova 2 Lite" "no webcam photo"
  fi
  timeout 60 "$AGENT" -u -m agent.voice_sonic --text "Say exactly: EyeReach test complete." --out "$LOGS/sonic.wav" \
    > "$LOGS/sonic.txt" 2>&1
  if grep -q "^Reply:" "$LOGS/sonic.txt"; then
    record PASS "Amazon Nova 2 Sonic speaks" "$(grep "^Reply:" "$LOGS/sonic.txt" | cut -d' ' -f2-4)"
    if [ "$SOUND" = true ]; then
      "$AGENT" -c "
import wave, numpy as np, sounddevice as sd
w = wave.open('$LOGS/sonic.wav'); sd.play(np.frombuffer(w.readframes(w.getnframes()), np.int16), w.getframerate()); sd.wait()" \
        > /dev/null 2>&1 && record PASS "Sonic's reply played on the speaker (did you hear it?)" \
                         || record FAIL "Sonic's reply on the speaker" "couldn't play $LOGS/sonic.wav"
    fi
  else
    record FAIL "Amazon Nova 2 Sonic speaks" "$(grep -v AKIA "$LOGS/sonic.txt" | tail -1 | cut -c1-120)"
  fi
fi

section "Summary"
pass=0 fail=0 skip=0
for r in "${RESULTS[@]}"; do
  case "${r%%|*}" in PASS) pass=$((pass + 1)) ;; FAIL) fail=$((fail + 1)) ;; SKIP) skip=$((skip + 1)) ;; esac
done
for r in "${RESULTS[@]}"; do
  IFS='|' read -r status name detail <<< "$r"
  [ "$status" = PASS ] || printf "  %-5s %s%s\n" "$status" "$name" "${detail:+  ($detail)}"
done
echo "  $pass passed, $fail failed, $skip skipped.   Logs: $LOGS"
echo
echo "Then the real arm (someone holds it, 12 V switch within reach):  launch/eyereach.sh"
echo "Talk to it (Nova 2 Sonic):                                       launch/talk.sh"
[ "$fail" -eq 0 ]
