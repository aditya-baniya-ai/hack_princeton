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
        robot = eye_control.arm_link.VirtualArm()
        robot.hover = eye_control.HOVER_MM
        robot.start()
        robot.closed = True
        ready = [False]
        robot.grip_ready = lambda closed: ready[0]
        self.assertTrue(robot.grab(eye_control.GRAB_MM))
        self.assertEqual(robot.stage, "Opening")
        self.assertFalse(robot.closed)
        robot.run_steps()
        self.assertEqual(robot.plan, [])  # no lowering before open feedback
        ready[0] = True
        robot.run_steps()
        self.assertEqual(robot.stage, "Lowering")
        self.assertAlmostEqual(eye_control.arm_link.arm.forward(robot.plan[0][1])[2], eye_control.GRAB_MM)
        robot.run_steps()
        self.assertFalse(robot.closed)  # no closing before measured arrival
        robot.state = list(robot.plan[0][1])
        ready[0] = False
        robot.run_steps()
        self.assertEqual(robot.stage, "Closing")
        self.assertTrue(robot.closed)
        robot.run_steps()
        self.assertEqual(robot.stage, "Closing")
        ready[0] = True
        robot.run_steps()
        self.assertEqual(robot.stage, "Lifting")
        robot.state = list(robot.plan[0][1])
        robot.run_steps()
        self.assertFalse(robot.busy())
        self.assertTrue(robot.closed)
        self.assertAlmostEqual(eye_control.arm_link.arm.forward(robot.command)[2], robot.hover)

    def test_stays_closed_even_when_the_finger_check_says_nothing(self):
        for detected in (False, True):
            robot = Mock()
            robot.closed = True
            robot.holding.return_value = detected
            robot.busy.return_value = False
            robot.gripper_position.return_value = -0.008
            ctl = GuidedControl(robot, movement="follow")
            ctl.phase = "picking"
            ctl.step(None, now=1)
            self.assertTrue(ctl.holding)
            robot.open_here.assert_not_called()
            self.assertEqual(ctl.events[-1][1]["gripper"], -0.008)
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
