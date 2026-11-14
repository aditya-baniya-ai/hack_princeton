# EyeReach

A robot arm that a person with severe arm paralysis controls with their eyes: look at an
object, hold the gaze to confirm, and the arm picks it up and puts it in a drop zone.

Built at HackPrinceton Fall 2026 (Nov 13-15, 2026). This is a research prototype, not a medical
device, and it has not been tested with people who have disabilities.

## Status

| Part | State |
|---|---|
| Gaze-to-arm logic and the screen app | tested with made-up landmarks and the pretend arm; **not yet tried with a real face or the real arm** |
| Webcam gaze accuracy | unknown until measured on a person |
| HoloLens 2 gaze input | not built yet (next) |
| Uncertainty handling (ask for confirmation when unsure) | not built yet |

## Tracks and sponsors

Main track: **Healthcare**. EyeReach restores a basic action (picking something up) to a person who cannot
use their arms. Special tracks, and how each connects. The status column says what exists today.

| Track | How EyeReach connects | Status |
|---|---|---|
| **Best Hardware** | A physical system: OpenMANIPULATOR-X arm, OpenCR board, Jetson Orin NX and a webcam, wired and driven by our own control code. | Arm driver and gaze control written; real-arm test pending |
| **Best Physical AI** | AI that acts in the real world: gaze estimation (MediaPipe iris landmarks) decides where a physical arm goes, with safety limits on every move. | Same as above |
| **Amazon: Best Practical AI (Nova)** | Gaze alone cannot say *what* to pick up. Planned: send the camera view to Amazon Nova to name the objects on the table and give each a box; the user looks at an object's box, holds to confirm, and the arm goes to it. Nova's answer is only a suggestion, so the user still confirms, and when it is unsure the system asks instead of guessing. | Not built |
| **LangChain deepagents** | Planned: an agent turns a spoken or typed task ("bring me the pills") into steps: find the object, ask the user to confirm with their eyes, move, grab, drop. It uses human-in-the-loop so the arm never acts on a plan the user has not approved. | Not built |
| **Photon (spectrum-ts)** | Planned: a caregiver gets an iMessage when the arm grabs or drops something, or when the user keeps looking at an object it cannot reach, so the user does not have to be watched in person. | Not built |

Planned order: the real-arm eye control first, then Nova, then deepagents, then Photon. If time runs short, the later
ones are dropped, and nothing in the table above is claimed as working until its status says so. Further ahead:
HoloLens 2 gaze input, so the user looks at real objects instead of a screen map.

## How it works

```
eyereach/run_webcam.py     webcam -> iris gaze -> a point on the table map (gaze_webcam.py)
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

How the eye control behaves:

- **Calibrate:** look at 12 dots (about 25 seconds). It learns how your eyes map to screen positions and prints
  its average error on the dots, which looks better than real use will.
- **Follow:** the arm glides to where you look inside the table map ("toward you" at the top, "toward Robo" at
  the bottom). If your face is lost, you blink, or you look outside the map, the arm holds still.
- **Grab:** hold your gaze on the claw for 0.8 seconds. The claw must already be where you are looking.
- **Drop:** carry it (slower), then hold your gaze on the pink DROP corner.
- **Keys:** Esc closes, C calibrates again, Space pauses.

## Folders

```
eyereach/      the eye control: gaze_webcam.py (iris gaze, calibration), eye_control.py (dwell, grab, drop),
               run_webcam.py (the screen app)
brain/         arm.py: the robot maths, plus commands to check it (test, where, go, home)
               arm_link.py: the arm follows a point, grabs and releases (or a pretend arm)
gestures/      gestures.py: the ROS 2 link that moves the joints
launch/        eyereach.sh (start the arm and the app), stop.sh (stop everything), fix_turns.py
firmware/      servo_check: an Arduino test that stands the arm up and waves
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

## Run it with the arm

Always start with the pretend arm, then the simulated arm, then the real one.

```bash
launch/eyereach.sh --no-arm      # pretend arm, no ROS
launch/eyereach.sh --sim         # simulated arm in RViz
launch/eyereach.sh               # the real arm
```

For the real arm: turn on the 12 V power, make sure the OpenCR's USB is in the Jetson, **hold the arm** (it goes limp
when it starts and stops), keep the power switch within reach, then run it. If it stays on "Waiting for the arm",
look in `/tmp/robo_arm.log`. Esc in the app stops the arm too. Other options go to the app (`--camera /dev/video0`,
`--reuse`, `--fullscreen`, `--dwell 1.2`).

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
