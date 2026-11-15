"""Check the whole pick-up on the ROS arm software with fake hardware (no real arm moves).

    python3 tools/sim_check.py        (needs the simulated arm running: launch/test_all.sh starts it)

Uses the same code as the real arm (RealArm, GuidedControl) and checks, in order:
  the gripper starts closed, and start-up opens it before the arm moves;
  a mouth opening: open, lower to the pick-up height, close, lift back up;
  it holds (stays closed) with no mouth opening; the next mouth opening opens it;
  after stop() the gripper is open.
Prints one line per check and exits 0 if all passed.
"""

import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for folder in (ROOT, os.path.join(ROOT, "brain"), os.path.join(ROOT, "gestures")):
    sys.path.insert(0, folder)

from eyereach import eye_control  # noqa: E402
import arm  # noqa: E402
from gestures import GRIPPER_POSITION, connect, move_gripper  # noqa: E402

results = []


def check(name, ok, detail=""):
    results.append(ok)
    print(f"  {'ok  ' if ok else 'FAIL'}  {name}{f'  ({detail})' if detail else ''}")


def main():
    r = eye_control.arm_link.RealArm()
    r.hover = eye_control.HOVER_MM
    if not connect(r.node):
        print("  FAIL  the simulated arm isn't running")
        return 1
    move_gripper(r.node, GRIPPER_POSITION["close"], wait=True)
    for _ in range(20):
        r.rclpy.spin_once(r.node, timeout_sec=0.05)
    check("gripper closed before start (like a bad previous run)", r.gripper_position() <= -0.005,
          f"{r.gripper_position():+.3f}")

    shoulder_before = r.joints()[1]
    why = r.start()
    check("start-up opens the gripper", why is None and r.gripper_position() >= eye_control.arm_link.OPEN_ENOUGH,
          why or f"{r.gripper_position():+.3f}")
    check("... and the arm is at the travel height", abs(arm.forward(r.joints())[2] - eye_control.HOVER_MM) < 8,
          f"{arm.forward(r.joints())[2]:.0f} mm, wanted {eye_control.HOVER_MM:.0f}; shoulder moved "
          f"{abs(r.joints()[1] - shoulder_before):.2f} rad")

    ctl = eye_control.GuidedControl(r, movement="follow")
    t0, lowest, closed_low, open_going_down = time.monotonic(), 999.0, False, True
    ctl.step(None, mouth_open=True)
    while time.monotonic() - t0 < 15:
        ctl.step(None)
        z, g = arm.forward(r.joints())[2], r.gripper_position()
        if z < eye_control.HOVER_MM - 10 and lowest > eye_control.GRAB_MM + 5 and g < 0.012:
            open_going_down = False                    # closed before it reached the bottom
        lowest = min(lowest, z)
        if z < eye_control.GRAB_MM + 5 and g <= -0.005:
            closed_low = True
        if ctl.phase == "there" and time.monotonic() - t0 > 2:
            break
        time.sleep(0.02)
    z = arm.forward(r.joints())[2]
    check("mouth opening: goes down with the gripper open", open_going_down)
    check("... lowest point is the pick-up height", abs(lowest - eye_control.GRAB_MM) < 5,
          f"{lowest:.0f} mm, wanted {eye_control.GRAB_MM:.0f}")
    check("... closes at the bottom", closed_low)
    check("... lifts back up, still closed", z > eye_control.HOVER_MM - 15 and r.gripper_position() <= -0.005,
          f"{z:.0f} mm, gripper {r.gripper_position():+.3f}")

    end = time.monotonic() + 3
    while time.monotonic() < end:
        ctl.step(None)
        time.sleep(0.02)
    check("holds it for 3 s with no mouth opening", ctl.holding and r.gripper_position() <= -0.005)

    ctl.step(None, mouth_open=True)
    end = time.monotonic() + 4
    while time.monotonic() < end and ctl.phase != "waiting":
        ctl.step(None)
        time.sleep(0.02)
    for _ in range(20):
        r.rclpy.spin_once(r.node, timeout_sec=0.05)
    check("next mouth opening lets go (gripper open)", r.gripper_position() >= eye_control.arm_link.OPEN_ENOUGH,
          f"{r.gripper_position():+.3f}")

    move_gripper(r.node, GRIPPER_POSITION["close"], wait=True)   # close it, so stop() has to open it
    r.stop()
    import rclpy
    from rclpy.node import Node
    from sensor_msgs.msg import JointState
    rclpy.init()
    node, seen = Node("sim_check_peek"), {}
    node.create_subscription(JointState, "/joint_states", lambda m: seen.update(zip(m.name, m.position)), 10)
    end = time.monotonic() + 3
    while "gripper_left_joint" not in seen and time.monotonic() < end:
        rclpy.spin_once(node, timeout_sec=0.1)
    g = seen.get("gripper_left_joint")
    check("after stop() the gripper is open", g is not None and g >= eye_control.arm_link.OPEN_ENOUGH,
          "no reading" if g is None else f"{g:+.3f}")
    node.destroy_node()
    rclpy.shutdown()
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())
