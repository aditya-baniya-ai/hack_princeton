"""
The arm: follow a point on the table map smoothly, grab and release.

Callers work in "play area" positions (0 to 1, as the user sees the screen) and
this turns them into points in front of the arm, then into joint angles with brain/arm.py
(which also checks every point is safe).

    RealArm     the real arm, or the simulated one in RViz (through ROS)
    VirtualArm  no robot at all: a pretend arm that behaves like the real one
                (same speed limit and delay), to try it on any computer
"""

import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "brain"))
import arm  # noqa: E402  (also makes ../gestures importable)

HOVER_HEIGHT = 60.0      # mm: fingertips this high while moving around (a caller can
                         # set its own: robot.hover)
MAX_TILT = 30            # degrees the gripper may tilt from straight down while gliding
                         # high up (it can't reach far pointing straight down up there)
TOUCH_HEIGHT = 20.0      # mm: fingertips this high at the bottom of a whack
NEAR, FAR = 80.0, 220.0   # mm: closest and farthest the gripper goes from the base
                          # (it can reach 60-227 mm at every height callers use)
SIDE = math.radians(75)  # the furthest the base turns left or right
MAX_SPEED = 1.5          # rad/s: fastest any joint moves while following the hand
MAX_GLIDE = 250.0        # mm/s: fastest the gripper slides across the table
CARRY_GLIDE = 90.0       # mm/s: while carrying something (callers slow it down) ...
CARRY_SPEED = 0.8        # rad/s: ... and the fastest any joint moves then
WHACK_TIME = 0.30        # seconds down, and the same back up
GRAB_HEIGHT = 15.0       # mm: fingertips this high to grab a ball off the table
RELEASE_HEIGHT = 40.0    # mm: fingertips this high to let go of a ball
GRIPPER_WAIT = 1.0       # seconds for the gripper to close or open
OPEN_ENOUGH = 0.012      # gripper position: at least this = open (fully open is 0.019, closed -0.01)
HOLDING = -0.006         # gripper position: stopped above this when closing = a ball is
                         # in it (empty, it closes all the way to -0.01)
LAG = 0.12               # seconds the pretend arm trails behind (like the real one)


def area_to_point(u, v):
    """Play-area position to a point on the table in front of Robo (x, y in mm).

    Robo faces the player, so the player's right (u = 1) is Robo's left (+y). Hand up
    (v = 0) brings the gripper toward the player (far from the base); hand down
    (v = 1) sends it back toward Robo."""
    angle = (u - 0.5) * 2 * SIDE
    reach = FAR - v * (FAR - NEAR)
    return arm.BASE_X + reach * math.cos(angle), reach * math.sin(angle)


def point_to_area(x, y):
    """The opposite of area_to_point (for drawing where the gripper is)."""
    reach = math.hypot(x - arm.BASE_X, y)
    angle = math.atan2(y, x - arm.BASE_X)
    return 0.5 + angle / (2 * SIDE), (FAR - reach) / (FAR - NEAR)


def limit_step(start, goal, seconds, speed=MAX_SPEED):
    """Move from start toward goal, no joint faster than speed (rad/s)."""
    most = speed * seconds
    return [s + max(-most, min(most, g - s)) for s, g in zip(start, goal)]


class ArmBase:
    """What both arms share: following, whacking, and where the gripper is."""

    def __init__(self, touch_height):
        self.touch_height = touch_height
        self.command = None          # joint angles last sent
        self.aim = None              # the point (x, y mm) the gripper is sliding to
        self.glide = MAX_GLIDE       # mm/s: the sliding speed right now
        self.speed = MAX_SPEED       # rad/s: the fastest joint speed right now
        self.normal = (MAX_GLIDE, MAX_SPEED)       # (glide, speed) when not carrying
        self.carry = (CARRY_GLIDE, CARRY_SPEED)    # ... and while carrying
        self.hover = HOVER_HEIGHT    # mm: the gliding height
        self.whack_started = None    # time the current whack began, or None
        self.whack_spot = None       # where the gripper was when it whacked (x, y)
        self.whack_checked = True    # the current whack has been checked for a hit
        self.last_follow = time.monotonic()
        self.steps = []              # (time, what to do) for grab and release
        self.busy_until = 0.0        # the arm is grabbing or releasing until then
        self.operation = None
        self.stage = "Ready"
        self.error = None
        self.closed = False          # the gripper is closed (holding, or tried to)

    def whacking(self):
        return self.whack_started is not None and \
            time.monotonic() - self.whack_started < 2 * WHACK_TIME

    def hover_joints(self, x, y):
        """Joint angles for gliding over (x, y) at the gliding height: pointing straight
        down if it can, otherwise tilted a little (up to MAX_TILT), or None."""
        for tilt in range(0, MAX_TILT + 1, 2):
            joints, _ = arm.solve(x, y, self.hover, arm.DOWN - math.radians(tilt))
            if joints is not None:
                return joints
        return None

    def set_speeds(self, glide, speed, carry_glide, carry_speed):
        """Slower limits (mm/s and rad/s), normal and while carrying. Never above the defaults."""
        self.normal = (min(glide, MAX_GLIDE), min(speed, MAX_SPEED))
        self.carry = (min(carry_glide, CARRY_GLIDE), min(carry_speed, CARRY_SPEED))
        self.glide, self.speed = self.normal

    def carrying(self, on):
        """Slower while carrying something (on), back to normal speed after."""
        self.glide, self.speed = self.carry if on else self.normal

    def busy(self):
        """Whacking, grabbing or releasing: the arm doesn't follow the hand meanwhile."""
        return self.operation is not None or self.whacking() or time.monotonic() < self.busy_until

    def follow(self, u, v):
        """Move toward this play-area position (does nothing while busy)."""
        now = time.monotonic()
        seconds, self.last_follow = min(0.1, now - self.last_follow), now
        if self.busy() or self.command is None:
            return
        # Slide the gripper level toward the hand's point (a straight, flat path, so a
        # carried ball can't sag into the others), no faster than MAX_GLIDE ...
        if self.aim is None:
            self.aim = arm.forward(self.command)[:2]
        gx, gy = area_to_point(u, v)
        distance = math.hypot(gx - self.aim[0], gy - self.aim[1])
        share = min(1.0, self.glide * seconds / distance) if distance > 0 else 1.0
        # ... and no joint faster than MAX_SPEED: if a joint would have to (the base,
        # moving sideways close to it), take a shorter step instead, so all the joints
        # stay in step and the gripper stays level
        goal = None
        for _ in range(4):
            aim = (self.aim[0] + (gx - self.aim[0]) * share,
                   self.aim[1] + (gy - self.aim[1]) * share)
            goal = self.hover_joints(*aim)
            if goal is None:
                return
            fastest = max(abs(g - c) for g, c in zip(goal, self.command)) / max(seconds, 1e-3)
            if fastest <= self.speed:
                break
            share *= 0.95 * self.speed / fastest
        self.aim = aim
        self.command = limit_step(self.command, goal, seconds, self.speed)   # (safety net)
        self.send([self.command], [max(0.1, seconds * 2)])

    def whack(self):
        """Tap straight down where the gripper is now, and back up. Returns False if
        it's already whacking."""
        if self.whacking() or self.command is None:
            return False
        x, y, _, _ = arm.forward(self.command)
        down, why = arm.solve(x, y, self.touch_height)
        if down is None:
            return False
        self.whack_started = time.monotonic()
        self.whack_checked = False
        self.whack_spot = (x, y)
        self.send([down, self.command], [WHACK_TIME, 2 * WHACK_TIME])
        return True

    def grab(self, height=GRAB_HEIGHT):
        """Go down, close the gripper, come back up (without freezing the app: the
        steps happen in run_steps). Returns False if it can't right now."""
        return self.down_and_up(height, closed=True)

    def release(self, height=RELEASE_HEIGHT):
        """Go down a little, open the gripper, come back up."""
        return self.down_and_up(height, closed=False)

    def _grip_stage(self, closed, following):
        self.stage = "Closing" if closed else "Opening"
        self.set_gripper(closed)
        self.operation = dict(kind="grip", closed=closed, following=following,
                              deadline=time.monotonic() + 5.0)

    def _move_stage(self, goal, following=None, label="Lifting", path=None, times=None):
        start = list(self.joints())
        move = max(0.45, max(abs(g - j) for g, j in zip(goal, start)) / self.speed)
        self.stage = label
        self.operation = dict(kind="move", goal=goal, following=following,
                              deadline=time.monotonic() + (times[-1] if times else move) + 3.0)
        self.send(path or [goal], times or [move])

    def grip_ready(self, closed):
        """The virtual gripper completes instantly; the real one uses action feedback."""
        return self.closed == closed

    def release_here(self):
        if self.busy() or self.command is None:
            return False
        self.error = None
        self._grip_stage(False, None)
        return True

    def down_and_up(self, height, closed):
        if self.busy() or self.command is None:
            return False
        up = list(self.joints())
        x, y, _, _ = arm.forward(up)
        down, why = arm.solve(x, y, height)
        if down is None or arm.path_problem(up, down):
            self.error = why or "Pickup path is outside the workspace"
            return False
        self.error = None
        self.command, self.aim = list(up), (x, y)
        def lift():
            self._move_stage(up, label="Lifting")
        def close():
            self._grip_stage(closed, lift)
        def lower():
            self._move_stage(down, close, "Lowering")
        if closed:
            self._grip_stage(False, lower)
        else:
            lower()
        return True

    def recover(self):
        """Lift vertically before allowing lateral travel after an interrupted pickup."""
        if self.busy():
            return False
        x, y, z, _ = arm.forward(self.joints())
        goal = self.hover_joints(x, y)
        if goal is None or arm.path_problem(self.joints(), goal):
            self.error = "Cannot recover to the travel height here"
            return False
        self.error = None
        self._move_stage(goal, label="Lifting")
        return True

    def jog(self, direction, distance):
        """Bounded movement in the person's left/right or toward/away directions."""
        if self.busy() or direction not in ("left", "right", "toward me", "away from me"):
            return False
        if distance not in (5.0, 20.0):
            return False
        start = list(self.joints())
        x, y, z, _ = arm.forward(start)
        if abs(z - self.hover) > 5:
            self.error = "Resume first to return to the travel height"
            return False
        angle = math.atan2(y, x - arm.BASE_X)
        dx, dy = ((-math.sin(angle), math.cos(angle)) if direction in ("left", "right")
                  else (math.cos(angle), math.sin(angle)))
        sign = -1 if direction in ("left", "away from me") else 1
        path, times, previous, elapsed = [], [], start, 0.0
        steps = int(math.ceil(distance / 2))
        for i in range(1, steps + 1):
            px, py = x + sign * dx * distance * i / steps, y + sign * dy * distance * i / steps
            u, v = point_to_area(px, py)
            goal = self.hover_joints(px, py)
            if not (0 <= u <= 1 and 0 <= v <= 1) or goal is None or arm.path_problem(previous, goal):
                self.error = "That move is outside the workspace"
                return False
            elapsed += max(0.04, distance / steps / self.glide,
                           max(abs(g - j) for g, j in zip(goal, previous)) / self.speed)
            path.append(goal)
            times.append(elapsed)
            previous = goal
        self.error = None
        self.aim = (px, py)
        self._move_stage(path[-1], label="Moving", path=path, times=times)
        return True

    def halt(self):
        """Cancel future steps and replace the arm trajectory with a measured-position hold."""
        self.steps = []
        self.operation = None
        self.busy_until = 0.0
        self.whack_started = None
        self.cancel_gripper()
        if self.command is not None:
            self.command = list(self.joints())
            self.aim = arm.forward(self.command)[:2]
            self.send([self.command], [0.1])
        self.stage = "Stopped"

    def cancel_gripper(self):
        pass

    def run_operation(self):
        op = self.operation
        if op is None:
            return
        if op["kind"] == "grip":
            ready = self.grip_ready(op["closed"])
        else:
            ready = max(abs(g - j) for g, j in zip(op["goal"], self.joints())) <= 0.025
        if self.error or time.monotonic() > op["deadline"]:
            self.error = self.error or f'{self.stage} timed out'
            self.halt()
            return
        if ready:
            self.operation = None
            if op["kind"] == "move":
                self.command = list(op["goal"])
                self.aim = arm.forward(self.command)[:2]
            if op["following"]:
                op["following"]()
            else:
                self.stage = "Ready"

    def open_here(self):
        """Open the gripper right where it is (the prize drops from there)."""
        self.steps = []
        self.operation = None
        self.set_gripper(False)

    def run_steps(self):
        """Do the grab/release steps that are due (call every frame)."""
        self.run_operation()
        now = time.monotonic()
        while self.steps and now >= self.steps[0][0]:
            self.steps.pop(0)[1]()

    def bottom_of_whack(self):
        """True once, when the current whack reaches the table: the moment to check
        for a hit."""
        if self.whack_checked or time.monotonic() - self.whack_started < WHACK_TIME + LAG:
            return False
        self.whack_checked = True
        return True

    def gripper_area(self):
        """Where the gripper really is now, as a play-area position."""
        x, y, z, _ = arm.forward(self.joints())
        u, v = point_to_area(x, y)
        return u, v, z


class VirtualArm(ArmBase):
    """A pretend arm: follows its commands with a delay and the same speed limit."""

    name = "virtual arm"

    def __init__(self, touch_height=TOUCH_HEIGHT):
        super().__init__(touch_height)
        self.state = None
        self.plan = []           # (time, joints) still to reach
        self.updated = time.monotonic()

    def start(self):
        goal = self.hover_joints(*area_to_point(0.5, 0.5))
        self.state = list(goal)
        self.command = list(goal)
        self.updated = time.monotonic()
        return None

    def send(self, path, times):
        now = time.monotonic()
        self.plan = [(now + t, list(j)) for j, t in zip(path, times)]

    def joints(self):
        now = time.monotonic()
        seconds, self.updated = now - self.updated, now
        while len(self.plan) > 1 and now >= self.plan[0][0]:
            self.plan.pop(0)
        if self.plan:
            goal = self.plan[0][1]
            # Ease toward the goal like a real servo (a little behind, speed-limited)
            eased = [s + (g - s) * min(1.0, seconds / LAG) for s, g in zip(self.state, goal)]
            self.state = limit_step(self.state, eased, seconds * 2.5)
        return self.state

    def set_gripper(self, closed):
        self.closed = closed

    def holding(self):
        """Is the gripper holding something? (The pretend table sets self.held.)"""
        return self.closed and getattr(self, "held", None) is not None

    def update(self):
        self.run_steps()

    def stop(self):
        pass


class RealArm(ArmBase):
    """The real arm (or the simulated one), through ROS."""

    name = "Robo's arm"

    def __init__(self, touch_height=TOUCH_HEIGHT, ros_args=None):
        super().__init__(touch_height)
        import rclpy
        from gestures import Gesturer
        rclpy.init(args=ros_args)
        self.rclpy = rclpy
        self.node = Gesturer()
        self.grip_future = self.grip_handle = self.grip_result = None
        self.cancel_pending = False

    def start(self):
        """Check the arm, open the gripper (and wait until it reports open), then move to the
        middle of the play area. Returns None, or why it can't start."""
        from gestures import connect
        if not connect(self.node):
            return "the arm isn't ready (see the messages above)"
        if not self.node.gripper.wait_for_server(timeout_sec=3.0):
            return "the gripper controller isn't running"
        if self.open_and_wait():
            print(f"Gripper open ({self.gripper_position():+.4f}).")
        else:
            print(f"Warning: the gripper didn't report open (position {self.gripper_position()}); "
                  "check it isn't blocked or overloaded (README step 5c).")
        goal = self.hover_joints(*area_to_point(0.5, 0.5))
        why = "the middle of the play area can't be reached" if goal is None \
            else arm.move_joints_safely(self.node, goal)
        if why:
            return why
        self.command = list(goal)
        return None

    def gripper_position(self):
        return self.node.joints.get("gripper_left_joint")

    def open_and_wait(self, seconds=3.0):
        """Open the gripper and wait until /joint_states shows it open. True if it did."""
        self.set_gripper(False)
        end = time.monotonic() + seconds
        while time.monotonic() < end:
            self.rclpy.spin_once(self.node, timeout_sec=0.05)
            position = self.gripper_position()
            if position is not None and position >= OPEN_ENOUGH:
                return True
        return False

    def set_gripper(self, closed):
        """Open or close the gripper without waiting for it."""
        from control_msgs.action import GripperCommand
        from gestures import GRIPPER_POSITION
        goal = GripperCommand.Goal()
        goal.command.position = GRIPPER_POSITION["close" if closed else "open"]
        goal.command.max_effort = 100.0
        self.grip_handle = self.grip_result = None
        self.cancel_pending = False
        self.grip_future = self.node.gripper.send_goal_async(goal)
        self.closed = closed

    def cancel_gripper(self):
        self.cancel_pending = True
        if self.grip_handle is not None:
            self.grip_handle.cancel_goal_async()
            self.cancel_pending = False

    def read_gripper(self):
        if self.grip_future is not None and self.grip_future.done():
            future, self.grip_future = self.grip_future, None
            try:
                self.grip_handle = future.result()
                if not self.grip_handle.accepted:
                    self.error = "Gripper command rejected"
                else:
                    self.grip_result = self.grip_handle.get_result_async()
                    if self.cancel_pending:
                        self.cancel_gripper()
            except Exception as exc:
                self.error = f"Gripper communication failed: {exc}"

    def grip_ready(self, closed):
        self.read_gripper()
        if self.grip_result is None or not self.grip_result.done():
            return False
        try:
            result = self.grip_result.result()
        except Exception as exc:
            self.error = f"Gripper result failed: {exc}"
            return False
        if result.status != 4:  # action_msgs/GoalStatus.STATUS_SUCCEEDED
            self.error = "Gripper command did not complete; check overload or obstruction"
            return False
        if not closed:
            position = self.gripper_position()
            return position is not None and position >= OPEN_ENOUGH
        return bool(result.result.reached_goal or result.result.stalled)

    def holding(self):
        """Is the gripper holding something? Closing on a ball stops the fingers
        partway; empty, they close all the way."""
        position = self.node.joints.get("gripper_left_joint")
        return self.closed and position is not None and position > HOLDING

    def send(self, path, times):
        from gestures import JOINTS, point
        from trajectory_msgs.msg import JointTrajectory
        msg = JointTrajectory()
        msg.joint_names = JOINTS
        msg.points = [point(list(j), t) for j, t in zip(path, times)]
        self.node.jog_pub.publish(msg)

    def update(self):
        """Read new joint angles (call every frame)."""
        for _ in range(5):
            self.rclpy.spin_once(self.node, timeout_sec=0.0)
        self.read_gripper()
        from gestures import JOINTS
        if self.command is not None and any(time.monotonic() - self.node.joint_seen.get(j, 0) > 1.0 for j in JOINTS):
            self.error = "Arm joint feedback lost"
        self.run_steps()

    def joints(self):
        from gestures import JOINTS
        return [self.node.joints[j] for j in JOINTS]

    def stop(self):
        """Open the gripper (waiting for it, so the next start finds it open), back to the home
        pose, then disconnect."""
        try:
            if self.command is not None and not self.open_and_wait():
                print(f"Warning: the gripper didn't report open before stopping "
                      f"(position {self.gripper_position()}).")
            if self.command is not None:
                arm.move_joints_safely(self.node, [self.joints()[0], 0.0, 0.0, 0.0])
        finally:
            self.node.destroy_node()
            if self.rclpy.ok():
                self.rclpy.shutdown()
