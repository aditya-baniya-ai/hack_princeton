"""Check responsiveness and interruption without waiting on simulated motion."""
import unittest
from unittest.mock import Mock

from eyereach.eye_control import GuidedControl, Dwell


class LiveControlTest(unittest.TestCase):
    def setUp(self):
        self.robot = Mock()
        self.robot.busy.return_value = False
        self.robot.closed = False
        self.robot.gripper_area.return_value = (0.5, 0.5, 60)
        self.ctl = GuidedControl(self.robot, movement="follow", look=0.65)

    def test_follows_first_frame_and_new_targets_without_dwell(self):
        for t, spot in [(0, (0.2, 0.3)), (0.033, (0.8, 0.6))]:
            self.ctl.step(("map", spot), now=t)
            self.robot.follow.assert_called_with(*spot)
        self.assertEqual(self.robot.follow.call_count, 2)

    def test_leaving_map_or_losing_tracking_stops_commands(self):
        self.ctl.step(("map", (0.2, 0.3)), now=0)
        self.robot.follow.reset_mock()
        self.ctl.step(("button", "pick up"), now=0.03)
        self.ctl.step(None, now=0.06, suspended=True)
        self.robot.follow.assert_not_called()
        self.assertIsNone(self.ctl.target)

    def test_stop_stays_stopped_until_explicit_resume(self):
        self.ctl.press("stop", 0)
        self.ctl.step(("map", (0.2, 0.3)), now=0.1)
        self.robot.follow.assert_not_called()
        self.ctl.press("resume", 0.2)
        self.ctl.step(("map", (0.2, 0.3)), now=0.3)
        self.robot.follow.assert_called_once_with(0.2, 0.3)

    def test_double_blink_picks_up_without_following_during_sequence(self):
        self.robot.busy.return_value = True
        self.ctl.step(("map", (0.2, 0.3)), now=0)
        self.robot.follow.reset_mock()
        self.ctl.step(("map", (0.8, 0.6)), now=0.1, double_blink=True)
        self.robot.grab.assert_called_once()
        self.robot.follow.assert_not_called()
        self.assertEqual(self.ctl.phase, "picking")

    def test_pause_cancels_saved_route_and_ignores_blinks(self):
        self.ctl.go((0.1, 0.9), "dropoff", 0, then="let go")
        self.ctl.step(None, now=0.1, double_blink=True, suspended=True)
        self.robot.follow.assert_not_called()
        self.robot.grab.assert_not_called()
        self.assertIsNone(self.ctl.then)

    def test_button_requires_short_dwell_and_leaving_to_repeat(self):
        d = Dwell(0.65)
        button = ("button", "home")
        self.assertIsNone(d.update(button, 0)[1])
        self.assertEqual(d.update(button, 0.66)[1], button)
        self.assertIsNone(d.update(button, 2)[1])
        d.update(None, 3)
        d.update(button, 4)
        self.assertEqual(d.update(button, 4.66)[1], button)


class PickupSequenceTest(unittest.TestCase):
    def test_open_then_lower_then_close_then_lift_and_stay_closed(self):
        from eyereach import eye_control
        from unittest.mock import patch
        self.assertGreaterEqual(eye_control.HOVER_MM, 120.0)        # 12 cm: well clear of the table
        self.assertGreaterEqual(eye_control.GRAB_MM, 50.0)
        height = lambda joints: eye_control.arm_link.arm.forward(joints)[2]   # noqa: E731
        robot = eye_control.arm_link.VirtualArm()
        robot.hover = eye_control.HOVER_MM
        robot.start()
        self.assertAlmostEqual(robot.gripper_area()[2], eye_control.HOVER_MM, places=4)
        robot.closed = True                                          # left closed from before
        start_plan = robot.plan
        with patch.object(eye_control.arm_link.time, "monotonic", return_value=100.0):
            self.assertTrue(robot.grab(eye_control.GRAB_MM))
        self.assertFalse(robot.closed)                               # 1. opens first ...
        self.assertIs(robot.plan, start_plan)                        # ... and hasn't started down yet
        down_at, close_at, lift_at = [t for t, _ in robot.steps]
        self.assertGreaterEqual(down_at - 100.0, 1.0)                # waits for it to open
        with patch.object(eye_control.arm_link.time, "monotonic", return_value=down_at):
            robot.run_steps()
        self.assertAlmostEqual(height(robot.plan[0][1]), eye_control.GRAB_MM, places=4)   # 2. down, open
        self.assertFalse(robot.closed)
        with patch.object(eye_control.arm_link.time, "monotonic", return_value=close_at):
            robot.run_steps()
        self.assertTrue(robot.closed)                                # 3. closes at the bottom
        self.assertGreaterEqual(lift_at - close_at, 1.0)
        with patch.object(eye_control.arm_link.time, "monotonic", return_value=lift_at):
            robot.run_steps()
        self.assertAlmostEqual(height(robot.plan[0][1]), eye_control.HOVER_MM, places=4)  # 4. lifts
        self.assertTrue(robot.closed)                                # ... and stays closed

    def test_small_object_is_not_automatically_released(self):
        robot = Mock()
        robot.closed = True
        robot.holding.return_value = False
        robot.busy.return_value = False
        ctl = GuidedControl(robot, movement="follow")
        ctl.phase = "picking"
        ctl.step(None, now=1)
        self.assertTrue(ctl.holding)
        robot.open_here.assert_not_called()
        ctl.blink(2)
        robot.release_here.assert_called_once()


class HandButtonLatchTest(unittest.TestCase):
    def test_staying_on_pickup_does_not_activate_changed_release_button(self):
        robot = Mock()
        robot.closed = True
        robot.busy.return_value = False
        ctl = GuidedControl(robot, movement="follow", look=0.65)
        ctl.press("pick up", 0)
        ctl.step(("button", "pick up"), now=1)
        ctl.step(("button", "let go"), now=2)
        ctl.step(("button", "let go"), now=3)
        robot.release_here.assert_not_called()
        ctl.step(("map", (0.5, 0.5)), now=4)
        ctl.step(("button", "let go"), now=5)
        ctl.step(("button", "let go"), now=6)
        robot.release_here.assert_called_once()
