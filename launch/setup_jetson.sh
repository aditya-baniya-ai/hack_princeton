#!/usr/bin/env bash
# One-time setup of the Jetson (Ubuntu 22.04, JetPack 6, aarch64) for EyeReach: README steps 2 to 8,
# plus the Python 3.12 environment the task agent needs. Safe to run again: each step checks first and
# skips what is already done.
#
#   bash launch/setup_jetson.sh            everything except voice
#   bash launch/setup_jetson.sh --voice    also the offline voice commands (launch/setup_voice.sh)
#   bash launch/setup_jetson.sh --no-agent skip the Python 3.12 task-agent environment
#
# It asks for your password (sudo). It never moves the arm. Not done here: the OpenCR firmware
# (README step 1, from a Mac or PC) and logging out and back in after the USB permission step.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(dirname "$HERE")"
WS="$HOME/colcon_ws"
EYE_VENV="$HOME/eyereach_venv"            # Python 3.10 + system ROS: the app and the arm
AGENT_VENV="$HOME/eyereach_agent_venv"    # Python 3.12: the deepagents task agent (needs 3.11+)
XACRO="$WS/src/open_manipulator/open_manipulator_x_description/ros2_control/open_manipulator_x_system.ros2_control.xacro"
VOICE=false
AGENT=true
for arg in "$@"; do
  case "$arg" in
    --voice) VOICE=true ;;
    --no-agent) AGENT=false ;;
    *) echo "Unknown option: $arg"; exit 1 ;;
  esac
done

step() { echo; echo "=== $*"; }
skip() { echo "    already done: $*"; }
ros_env() { set +u; source /opt/ros/humble/setup.bash; [ -f "$WS/install/setup.bash" ] && source "$WS/install/setup.bash"; set -u; }

step "0. Check the machine"
[ "$(id -u)" -ne 0 ] || { echo "Run this as your normal user, not with sudo."; exit 1; }
. /etc/os-release
echo "    $PRETTY_NAME, $(uname -m), $(python3 --version)"
if [ "${VERSION_ID:-}" != "22.04" ] || [ "$(uname -m)" != "aarch64" ]; then
  echo "    This script is for Ubuntu 22.04 on a Jetson (aarch64). Stopping so nothing is installed by mistake."
  exit 1
fi
sudo -v   # ask for the password once, at the start

step "1. Basic tools"
sudo apt-get update
sudo apt-get install -y software-properties-common curl git unzip python3-venv python3-pip v4l-utils x11-utils

step "2. ROS 2 Humble (README step 2)"
if [ -f /opt/ros/humble/setup.bash ]; then
  skip "ROS 2 Humble is installed"
else
  sudo add-apt-repository -y universe
  sudo curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key \
    -o /usr/share/keyrings/ros-archive-keyring.gpg
  echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $UBUNTU_CODENAME main" \
    | sudo tee /etc/apt/sources.list.d/ros2.list > /dev/null
  sudo apt-get update
  sudo apt-get install -y ros-humble-desktop ros-dev-tools
fi

step "3. USB permission (README step 4)"
if id -nG "$USER" | grep -qw dialout; then
  skip "$USER is in the dialout group"
else
  sudo usermod -aG dialout "$USER"
  echo "    Added. You must LOG OUT AND BACK IN (or reboot) before the arm can use the USB port."
fi

step "4. ROBOTIS arm software (README step 5)"
sudo apt-get install -y ros-humble-ros2-control ros-humble-ros2-controllers ros-humble-moveit \
  ros-humble-gripper-controllers
mkdir -p "$WS/src"
if [ -d "$WS/src/open_manipulator" ]; then
  skip "the ROBOTIS code is downloaded"
else
  vcs import "$WS/src" < "$REPO/dependencies.repos"
fi
[ -f /etc/ros/rosdep/sources.list.d/20-default.list ] || sudo rosdep init
rosdep update
ros_env
(cd "$WS" && rosdep install --from-paths src --ignore-src -y -r \
  --skip-keys "gazebo_ros gazebo_ros_pkgs gazebo_ros2_control gazebo_plugins")

step "5. Base servo gains and gripper current limit (README steps 5b and 5c)"
REBUILD=false
if sed -n '/<gpio name="dxl1">/,/<\/gpio>/p' "$XACRO" | grep -q '"Position P Gain">400<'; then
  skip "base servo gains are P 400, I 0, D 0"
else
  cp "$XACRO" "$HOME/omx_ros2_control_backup.xacro"
  sed -i '/<gpio name="dxl1">/,/<\/gpio>/ { s|"Position P Gain">800<|"Position P Gain">400<|; s|"Position I Gain">100<|"Position I Gain">0<|; s|"Position D Gain">100<|"Position D Gain">0<|; }' "$XACRO"
  REBUILD=true
fi
if sed -n '/<gpio name="dxl5">/,/<\/gpio>/p' "$XACRO" | grep -q '"Goal Current">300<'; then
  skip "gripper current is limited"
else
  cp "$XACRO" "$HOME/omx_ros2_control_backup_before_gripper.xacro"
  sed -i '/<gpio name="dxl5">/,/<\/gpio>/ { /<param name="Return Delay Time">0<\/param>/a\        <param name="Operating Mode">5</param>\n        <param name="Goal Current">300</param>
}' "$XACRO"
  REBUILD=true
fi
sed -n '/<gpio name="dxl1">/,/<\/gpio>/p' "$XACRO" | grep "Gain" | sed 's/^/    base: /'
sed -n '/<gpio name="dxl5">/,/<\/gpio>/p' "$XACRO" | grep -E "Operating Mode|Goal Current" | sed 's/^/    gripper: /'

step "6. Build the arm workspace"
if [ -f "$WS/install/open_manipulator_x_bringup/share/open_manipulator_x_bringup/package.xml" ] && [ "$REBUILD" = false ]; then
  skip "the workspace is built"
else
  # one worker: the parallel build can freeze the Jetson
  (cd "$WS" && colcon build --symlink-install --parallel-workers 1)
fi
ros_env

step "7. Terminal and USB rules (README step 6)"
for line in "source /opt/ros/humble/setup.bash" "source $WS/install/setup.bash"; do
  grep -qxF "$line" "$HOME/.bashrc" || echo "$line" >> "$HOME/.bashrc"
done
ros2 run open_manipulator_x_bringup create_udev_rules

step "8. Eye tracker environment ($EYE_VENV, Python 3.10) (README step 8)"
python3 -c "import cv2" 2>/dev/null || sudo apt-get install -y python3-opencv
if [ -x "$EYE_VENV/bin/python" ]; then
  skip "$EYE_VENV exists"
else
  python3 -m venv --system-site-packages "$EYE_VENV"
  python3 -c "import site; print(site.getusersitepackages())" > "$EYE_VENV/lib/python3.10/site-packages/user_site.pth"
fi
"$EYE_VENV/bin/python" -c "import mediapipe" 2>/dev/null || {
  "$EYE_VENV/bin/pip" install --no-deps mediapipe==0.10.18
  "$EYE_VENV/bin/pip" install absl-py attrs flatbuffers "protobuf>=4.25.3,<5" sentencepiece
}
"$EYE_VENV/bin/python" -c "import boto3" 2>/dev/null || "$EYE_VENV/bin/pip" install boto3   # Nova
"$EYE_VENV/bin/python" -c "import mediapipe, cv2, numpy; print('    mediapipe', mediapipe.__version__, '| numpy', numpy.__version__, '| cv2', cv2.__version__)"
"$EYE_VENV/bin/python" -c "import numpy, sys; sys.exit(0 if numpy.__version__.startswith('1.') else 'NumPy must be 1.x for ROS and MediaPipe')"

if [ "$AGENT" = true ]; then
  step "9. Task agent environment ($AGENT_VENV, Python 3.12)"
  # deepagents needs Python 3.11+, and the Jetson has 3.10. uv brings its own Python 3.12, so the
  # system Python (and ROS) stay untouched.
  export PATH="$HOME/.local/bin:$PATH"
  command -v uv > /dev/null || python3 -m pip install --user uv
  [ -x "$AGENT_VENV/bin/python" ] || uv venv --python 3.12 "$AGENT_VENV"
  uv pip install --python "$AGENT_VENV/bin/python" "deepagents>=0.7" "langchain-aws>=1.0" boto3 numpy opencv-python-headless \
    "aws_sdk_bedrock_runtime>=0.1.0,<0.2.0" smithy-aws-core sounddevice \
    photonhq-api   # Nova Sonic voice (launch/talk.sh); Photon caregiver messages (services/photon/)
  "$AGENT_VENV/bin/python" -c "import sys, deepagents; print('    deepagents ok on Python', sys.version.split()[0])"
fi

if [ "$VOICE" = true ]; then
  step "10. Offline voice commands"
  EYEREACH_PYTHON="$EYE_VENV/bin/python" bash "$HERE/setup_voice.sh"
fi

step "Checks (no arm needed; a failure here is reported, not fatal)"
cd "$REPO"
set +e
python3 brain/arm.py test | tail -1
echo "    unit tests ($EYE_VENV):"
"$EYE_VENV/bin/python" -m unittest discover -s tests 2>&1 | grep -E "^(Ran|OK|FAILED)" | sed 's/^/      /'
if [ "$AGENT" = true ]; then
  echo "    agent tests ($AGENT_VENV):"
  "$AGENT_VENV/bin/python" -m unittest tests.test_task_agent 2>&1 | grep -E "^(Ran|OK|FAILED)" | sed 's/^/      /'
fi
echo "    cameras:"; v4l2-ctl --list-devices 2>/dev/null | sed 's/^/      /' || echo "      none found"
if ls /dev/ttyACM* > /dev/null 2>&1; then echo "    OpenCR: $(ls /dev/ttyACM*)"; else echo "    OpenCR: not connected (plug in its USB and turn on the 12 V)"; fi

cat <<EOF

=== Done. Left for you:
  - If step 3 added you to dialout: log out and back in (or reboot), then check: groups | grep dialout
  - The OpenCR firmware (usb_to_dxl) is uploaded from a Mac or PC: README step 1.
  - Then, in order (hold the arm for the real one):
      launch/eyereach.sh --no-arm       pretend arm
      launch/eyereach.sh --sim          simulated arm in RViz
      launch/eyereach.sh                the real arm
  - The task agent: $AGENT_VENV/bin/python -m agent.task_agent "bring me the pills"
EOF
