# Jetson setup prompt for Claude Code

Open Claude Code on the Jetson and paste everything inside the box below.

```text
You are setting up an NVIDIA Jetson Orin NX 16GB (Ubuntu 22.04, JetPack 6) to run EyeReach: a webcam eye
tracker that drives an OpenMANIPULATOR-X robot arm through ROS 2 Humble and an OpenCR board.
Repo: https://github.com/aditya-baniya-ai/hack_princeton . The README in that repo has the exact commands
(section "One-time setup (on the Jetson)", steps 2 to 8). Follow it; do not invent other install methods.

Rules
- Work step by step. Before each step check whether it is already done and skip it if so.
- After each step run its check and show me the result. If a check fails, stop and fix it before moving on.
- Run sudo commands yourself; if a password is needed, tell me and wait.
- NEVER move the arm or start any arm program until I say the arm is held and the 12 V is on.
- Do not touch the OpenCR firmware: that is uploaded from a Mac or PC (README step 1), not from here.
- Do not commit or push anything unless I ask.

Steps
0. Look first: `cat /etc/os-release`, `uname -m`, `python3 --version` (expect 22.04, aarch64, 3.10), `df -h ~`
   (need about 10 GB free), `ls ~/Documents ~/colcon_ws ~/eyereach_venv 2>&1`. Report what exists.
1. ROS 2 Humble (README step 2). Check: `source /opt/ros/humble/setup.bash && ros2 --help`.
2. Repo (step 3): clone to ~/Documents/hack_princeton, or `git pull` if it exists.
3. USB permission (step 4): `sudo usermod -aG dialout $USER`. Tell me I must log out and back in or reboot, and
   check `groups` lists dialout afterwards.
4. Arm software (step 5): apt packages, `vcs import` of dependencies.repos into ~/colcon_ws/src, rosdep,
   `colcon build --symlink-install` (use `--parallel-workers 1` if it freezes). Check: it ends with
   "Summary: 12 packages finished".
5. Base servo gains (step 5b): back up the xacro, apply the sed, rebuild open_manipulator_x_description only.
   Check: the grep shows P 400, I 0, D 0.
6. Terminal and udev (step 6): add the two `source` lines to ~/.bashrc once (no duplicates), then
   `ros2 run open_manipulator_x_bringup create_udev_rules`.
7. MediaPipe (step 8): create ~/eyereach_venv exactly as the README says. Check: it prints mediapipe 0.10.18
   and a NumPy 1.x version, and `import cv2` works. If the system has no cv2, install the Jetson-compatible
   OpenCV for the system python (apt `python3-opencv`), not a pip wheel that replaces NumPy.
8. Camera: `ls /dev/video*`, `v4l2-ctl --list-devices` (apt install v4l-utils if missing). Note which device is
   the webcam.
9. Tests without the arm: from the repo run `python3 -m unittest discover -s tests` with
   ~/eyereach_venv/bin/python (11 tests should pass), then `python3 brain/arm.py test`.
10. Pretend-arm run: `launch/eyereach.sh --no-arm --camera /dev/videoN` (use the right device). Tell me to look
    at the dots; I report the calibration error.
11. Stop here. Then give me a short checklist: what is installed, what is left for me (OpenCR firmware upload
    on a Mac, log out/in for dialout, `sudo nvpmodel -m 3` for 25 W if "System throttled due to over-current"
    shows up, plug in the OpenCR and check `ls /dev/ttyACM*`), and the exact commands for the next runs:
    `launch/eyereach.sh --sim`, then `launch/eyereach.sh` (real arm, hold it first).

If something in the README is wrong for this machine, tell me what you changed and why.
```
