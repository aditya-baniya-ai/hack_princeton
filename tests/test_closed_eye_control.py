"""Closed-eye steering interlock: simulated faces and robot calls, no hardware."""
import unittest
from unittest.mock import Mock

import numpy as np

from eyereach.eye_control import ClosedEyes, GuidedControl
from eyereach.gaze_webcam import LEFT, RIGHT, eye_openings
from eyereach.head_tilt import HeadTilt
from eyereach.run_webcam import VISION_SPEEDS, tilt_step
from tests.test_gaze_webcam import face


class ClosedEyesTest(unittest.TestCase):
    def test_requires_sustained_closure_of_both_eyes(self):
        gate = ClosedEyes()
        self.assertFalse(gate.update((0.25, 0.25), 0))
        self.assertFalse(gate.update((0.04, 0.04), 0.1))
        self.assertFalse(gate.update((0.25, 0.25), 0.3))  # normal blink: never enables
        self.assertFalse(gate.update((0.04, 0.04), 1))
        self.assertTrue(gate.update((0.04, 0.04), 1.4))
        self.assertTrue(gate.update((0.04, 0.04), 2))
        self.assertFalse(gate.update((0.04, 0.25), 2.01))  # even one open eye stops
        self.assertFalse(gate.update((0.25, 0.04), 2.02))
        self.assertFalse(gate.update((0.04, 0.04), 2.1))
        self.assertTrue(gate.update((0.04, 0.04), 2.5))

    def test_face_loss_pause_and_invalid_readings_reset_the_hold(self):
        for reading, enabled in ((None, True), ((0.04, 0.04), False),
                                 ((float('nan'), 0.04), True), ((-1, 0.04), True)):
            gate = ClosedEyes()
            gate.update((0.04, 0.04), 0)
            self.assertTrue(gate.update((0.04, 0.04), 0.5))
            self.assertFalse(gate.update(reading, 0.6, enabled=enabled))
            self.assertFalse(gate.update((0.04, 0.04), 0.7))

    def test_landmarks_keep_eyes_separate_so_a_wink_never_enables(self):
        pts = face(opening=3)
        eye = LEFT
        pts[eye['upper']], pts[eye['lower']] = (370, 193), (370, 207)
        openings = eye_openings(pts)
        self.assertLess(openings[0], 0.12)
        self.assertGreater(openings[1], 0.12)
        gate = ClosedEyes()
        self.assertFalse(gate.update(openings, 0))
        self.assertFalse(gate.update(openings, 1))
        self.assertIsNone(eye_openings(np.zeros((478, 2))))


class VisionMovementTest(unittest.TestCase):
    def controller(self, movement='follow'):
        robot = Mock()
        robot.closed = False
        robot.busy.return_value = False
        robot.gripper_area.return_value = (0.5, 0.5, 120)
        robot.normal = (100.0, 1.0)
        robot.carry = (60.0, 0.6)
        robot.glide, robot.speed = robot.normal
        return robot, GuidedControl(robot, movement=movement, vision_speeds=VISION_SPEEDS)

    def test_left_and_right_tilts_move_only_with_closed_eyes_then_halt(self):
        for mode in ('follow', 'joystick'):
            for yaw in (-20, 20):
                with self.subTest(mode=mode, yaw=yaw):
                    robot, ctl = self.controller()
                    head = HeadTilt(mode=mode)
                    head.set_centre(0, 0)
                    head.update((yaw, 0), 0)
                    tilt = head.update((yaw, 0), 1)
                    tilt_step(ctl, tilt, True, False, 1, eyes_closed=False)
                    robot.follow.assert_not_called()
                    tilt_step(ctl, tilt, True, False, 1.5, eyes_closed=True)
                    robot.follow.assert_called_once()
                    robot.follow.reset_mock()
                    robot.reset_mock()
                    tilt_step(ctl, tilt, True, False, 1.6, eyes_closed=False)
                    robot.halt.assert_called_once()
                    robot.follow.assert_not_called()
                    # Holding replaces the trajectory before any robot update.
                    self.assertLess(robot.method_calls.index(('halt', (), {})),
                                    robot.method_calls.index(('update', (), {})))
                    self.assertIsNone(ctl.target)

    def test_opening_eyes_cancels_a_dwell_destination(self):
        robot, ctl = self.controller('dwell')
        ctl.phase, ctl.target, ctl.then = 'moving', (0.2, 0.5), 'let go'
        ctl.step(('map', (0.2, 0.5)), now=1, vision_movement=False)
        robot.halt.assert_called_once()
        self.assertIsNone(ctl.target)
        self.assertIsNone(ctl.then)
        self.assertEqual(ctl.phase, 'waiting')

    def test_mouth_action_remains_available_with_open_eyes(self):
        robot, ctl = self.controller()
        robot.busy.return_value = True
        ctl.step(('map', (0.1, 0.5)), now=1, mouth_open=True, vision_movement=False)
        robot.grab.assert_called_once()
        robot.follow.assert_not_called()

    def test_eye_gate_does_not_cancel_voice_movement(self):
        robot, ctl = self.controller()
        ctl.control_mode, ctl.phase = 'voice', 'jogging'
        robot.busy.return_value = True
        ctl.step(None, now=1, vision_movement=False)
        robot.halt.assert_not_called()

    def test_slow_speed_limits(self):
        for holding, expected in ((False, (10.0, 0.12)), (True, (6.0, 0.08))):
            robot, ctl = self.controller()
            ctl.holding = holding
            observed = []
            robot.follow.side_effect = lambda *_: observed.append((robot.glide, robot.speed))
            ctl.step(('map', (0.1, 0.5)), now=1, vision_movement=True)
            self.assertEqual(observed, [expected])
            self.assertEqual((robot.glide, robot.speed), (100.0, 1.0))
