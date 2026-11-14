# Jetson setup prompt for Claude Code

Open Claude Code on the Jetson and paste everything inside the box below.

```text
You are setting up an NVIDIA Jetson Orin NX 16GB (Ubuntu 22.04, JetPack 6) to run EyeReach: a webcam eye
tracker that drives an OpenMANIPULATOR-X robot arm through ROS 2 Humble and an OpenCR board.
Repo: https://github.com/aditya-baniya-ai/hack_princeton . The README in that repo has the exact commands
(section "One-time setup (on the Jetson)", steps 2 to 9), and `launch/setup_jetson.sh` runs all of them.
Use that script; do not invent other install methods.

Rules
- Work step by step. Before each step check whether it is already done and skip it if so.
- After each step run its check and show me the result. If a check fails, stop and fix it before moving on.
- Run sudo commands yourself; if a password is needed, tell me and wait.
- NEVER move the arm or start any arm program until I say the arm is held and the 12 V is on.
- Do not touch the OpenCR firmware: that is uploaded from a Mac or PC (README step 1), not from here.
- Do not commit or push anything unless I ask.

Steps
0. Look first: `cat /etc/os-release`, `uname -m`, `python3 --version` (expect 22.04, aarch64, 3.10), `df -h ~`
   (need about 10 GB free), `ls ~/Documents ~/colcon_ws ~/eyereach_venv ~/eyereach_agent_venv 2>&1`.
   Report what exists.
1. Repo: clone to ~/Documents/hack_princeton, or `git pull` on main if it exists.
2. Run `bash launch/setup_jetson.sh` (add `--voice` if I want voice). It is safe to run again; it skips finished
   steps. It installs ROS 2 Humble, the USB permission, the ROBOTIS arm workspace with the base gains and gripper
   limit, the udev rules, ~/eyereach_venv (Python 3.10: MediaPipe, the app, the arm) and ~/eyereach_agent_venv
   (Python 3.12: the deepagents task agent, which needs 3.11+). If it stops on an error, show me the error,
   fix the cause, and run it again. Do not replace the system Python or install pip packages system-wide.
3. Read its final "Checks" section: `brain/arm.py test` says All good, the unit tests pass in both
   environments, and note which /dev/video device is the webcam. If it added me to dialout, tell me to log out
   and back in.
4. Pretend-arm run: `launch/eyereach.sh --no-arm --camera /dev/videoN` (use the right device). Tell me to look
    at the dots; I report the calibration error.
5. Stop here. Then give me a short checklist: what is installed, what is left for me (OpenCR firmware upload
    on a Mac, log out/in for dialout, `sudo nvpmodel -m 3` for 25 W if "System throttled due to over-current"
    shows up, plug in the OpenCR and check `ls /dev/ttyACM*`), and the exact commands for the next runs:
    `launch/eyereach.sh --sim`, then `launch/eyereach.sh` (real arm, hold it first).

If something in the README is wrong for this machine, tell me what you changed and why.
```
