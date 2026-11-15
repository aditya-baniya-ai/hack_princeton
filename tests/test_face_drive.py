"""Eyes-open, mouth-open steering and mouth-close pickup, without hardware."""
import unittest
from unittest.mock import Mock, patch

import numpy as np

from eyereach import eye_control as ec
from eyereach.gaze_webcam import LEFT, eye_openings
from eyereach.head_tilt import HeadTilt
from eyereach.run_webcam import VISION_SPEEDS, tilt_step
from tests.test_gaze_webcam import face

OPEN_EYES, CLOSED_EYES = (0.25, 0.25), (0.04, 0.04)


class FaceDriveTest(unittest.TestCase):
    def test_open_eyes_and_mouth_enable_movement_but_never_grab(self):
        gate = ec.FaceDrive()
        self.assertEqual(gate.update(OPEN_EYES, 0.05, 0), (False, False))
        self.assertEqual(gate.update(OPEN_EYES, 0.5, 1), (False, False))
        self.assertEqual(gate.update(OPEN_EYES, 0.5, 1.4), (True, False))
        self.assertEqual(gate.update(OPEN_EYES, 0.5, 2), (True, False))
        self.assertEqual(gate.update(CLOSED_EYES, 0.5, 3), (False, False))
        self.assertEqual(gate.update((0.25, 0.04), 0.5, 4), (False, False))

    def test_closure_fires_once_after_opening_and_holding_mouth(self):
        gate = ec.FaceDrive()
        gate.update(OPEN_EYES, 0.5, 0)
        gate.update(OPEN_EYES, 0.5, 0.4)
        self.assertEqual(gate.update(OPEN_EYES, 0.05, 1), (False, False))
        self.assertEqual(gate.update(OPEN_EYES, 0.05, 1.2), (False, True))
        self.assertEqual(gate.update(OPEN_EYES, 0.05, 5), (False, False))

    def test_invalid_readings_pause_and_face_loss_do_not_grab(self):
        for eyes, mouth, enabled in ((None, 0.05, True), (OPEN_EYES, None, True),
                                      (OPEN_EYES, float('nan'), True),
                                      (OPEN_EYES, 0.05, False)):
            gate = ec.FaceDrive()
            gate.update(OPEN_EYES, 0.5, 0)
            gate.update(OPEN_EYES, 0.5, 0.4)
            self.assertEqual(gate.update(eyes, mouth, 1, enabled=enabled), (False, False))
            self.assertEqual(gate.update(OPEN_EYES, 0.05, 2), (False, False))
            self.assertEqual(gate.update(OPEN_EYES, 0.05, 3), (False, False))

    def test_brief_mouth_opening_and_threshold_jitter_do_not_grab(self):
        gate = ec.FaceDrive()
        for mouth, now in ((0.05, 0), (0.5, 1), (0.25, 1.1), (0.05, 1.2), (0.05, 2)):
            self.assertEqual(gate.update(OPEN_EYES, mouth, now), (False, False))

    def test_separate_eye_landmarks_disable_movement_for_a_wink(self):
        pts = face(opening=3)
        pts[LEFT['upper']], pts[LEFT['lower']] = (370, 193), (370, 207)
        eyes = eye_openings(pts)
        gate = ec.FaceDrive()
        self.assertEqual(gate.update(eyes, 0.5, 0), (False, False))
        self.assertEqual(gate.update(eyes, 0.5, 1), (False, False))
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
        return robot, ec.GuidedControl(robot, movement=movement, vision_speeds=VISION_SPEEDS)

    def test_left_and_right_tilts_only_move_with_open_eyes_and_mouth(self):
        for mode in ('follow', 'joystick'):
            for yaw in (-20, 20):
                with self.subTest(mode=mode, yaw=yaw):
                    robot, ctl = self.controller()
                    head, gate = HeadTilt(mode=mode), ec.FaceDrive()
                    head.set_centre(0, 0)
                    head.update((yaw, 0), 0)
                    tilt = head.update((yaw, 0), 1)
                    for eyes, mouth, now in ((CLOSED_EYES, 0.5, 1), (OPEN_EYES, 0.05, 2)):
                        move, grab = gate.update(eyes, mouth, now)
                        tilt_step(ctl, tilt, True, False, now, grab_action=grab, vision_movement=move)
                    robot.follow.assert_not_called()
                    gate.update(OPEN_EYES, 0.5, 3)
                    move, grab = gate.update(OPEN_EYES, 0.5, 3.4)
                    tilt_step(ctl, tilt, True, False, 3.4, grab_action=grab, vision_movement=move)
                    robot.follow.assert_called_once()
                    robot.reset_mock()
                    move, grab = gate.update(OPEN_EYES, 0.05, 4)
                    tilt_step(ctl, tilt, True, False, 4, grab_action=grab, vision_movement=move)
                    robot.halt.assert_called_once()
                    robot.follow.assert_not_called()
                    self.assertEqual(robot.stage, 'Ready')
                    self.assertTrue(ctl.live_enabled)

    def test_mouth_closure_halts_dwell_motion_before_grabbing(self):
        robot, ctl = self.controller('dwell')
        ctl.phase, ctl.target = 'moving', (0.2, 0.5)
        robot.busy.return_value = True
        ctl.step(('map', (0.2, 0.5)), now=1, grab_action=True, vision_movement=False)
        robot.halt.assert_called_once()
        robot.grab.assert_called_once()
        self.assertEqual(ctl.phase, 'picking')

    def test_stopped_and_holding_states_cannot_trigger_another_grab_or_release(self):
        for holding in (False, True):
            robot, ctl = self.controller()
            ctl.holding = holding
            if not holding:
                ctl.press('stop', 0)
            ctl.step(None, now=1, grab_action=True, vision_movement=False)
            robot.grab.assert_not_called()
            robot.release_here.assert_not_called()

    def test_pickup_is_not_cancelled_when_mouth_stays_closed_or_tracking_is_lost(self):
        robot, ctl = self.controller()
        robot.busy.return_value = True
        ctl.step(None, now=0, grab_action=True, vision_movement=False)
        robot.reset_mock()
        ctl.step(None, now=1, suspended=True, vision_movement=False)
        robot.halt.assert_not_called()
        self.assertEqual(ctl.phase, 'picking')

    def test_voice_control_and_explicit_stop_still_work(self):
        robot, ctl = self.controller()
        ctl.control_mode, ctl.phase = 'voice', 'jogging'
        robot.busy.return_value = True
        ctl.step(None, now=1, vision_movement=False, grab_action=True)
        robot.halt.assert_not_called()
        robot.grab.assert_not_called()
        ctl.press('stop', 2)
        robot.halt.assert_called_once()
        self.assertFalse(ctl.live_enabled)

    def test_slow_speed_limits_are_unchanged(self):
        for holding, expected in ((False, (10.0, 0.12)), (True, (6.0, 0.08))):
            robot, ctl = self.controller()
            ctl.holding = holding
            observed = []
            robot.follow.side_effect = lambda *_: observed.append((robot.glide, robot.speed))
            ctl.step(('map', (0.1, 0.5)), now=1, vision_movement=True)
            self.assertEqual(observed, [expected])
            self.assertEqual((robot.glide, robot.speed), (100.0, 1.0))


class CompletePickupTest(unittest.TestCase):
    def test_mouth_closure_lowers_grabs_lifts_and_keeps_holding(self):
        from tests.test_fetch import arm_link, arm
        clock = [10.0]
        with patch.object(arm_link.time, 'monotonic', side_effect=lambda: clock[0]):
            robot = arm_link.VirtualArm()
            robot.hover = ec.HOVER_MM
            robot.start()
            robot.held = object()
            ctl = ec.GuidedControl(robot, movement='follow', vision_speeds=VISION_SPEEDS)
            gate, head = ec.FaceDrive(), HeadTilt()
            head.set_centre(0, 0)
            stages, grabs = [], []
            real_grab = robot.grab
            robot.grab = lambda height: grabs.append(height) or real_grab(height)

            def frame(mouth, eyes=OPEN_EYES):
                clock[0] += 0.05
                enabled = ctl.phase in ('waiting', 'there', 'following', 'moving') and ctl.live_enabled
                move, grab = gate.update(eyes, mouth, clock[0], enabled=enabled)
                tilt_step(ctl, head.update((0, 0), clock[0]), True, False, clock[0],
                          grab_action=grab, vision_movement=move)
                stages.append(robot.stage)

            for _ in range(12):
                frame(0.5)
            self.assertEqual(grabs, [])  # opening the mouth only steers
            for _ in range(500):
                frame(0.05)
                if ctl.holding:
                    break
            self.assertTrue(ctl.holding and robot.closed)
            self.assertEqual(len(grabs), 1)
            self.assertEqual([s for i, s in enumerate(stages) if i == 0 or s != stages[i-1]][-5:],
                             ['Opening', 'Lowering', 'Closing', 'Lifting', 'Holding'])
            self.assertAlmostEqual(arm.forward(robot.joints())[2], ec.HOVER_MM, delta=5)
            self.assertTrue(ctl.live_enabled)
            for _ in range(20):
                frame(0.05)
            self.assertEqual(robot.stage, 'Holding')
            for _ in range(12):
                frame(0.5)
            for _ in range(10):
                frame(0.05)
            self.assertEqual(len(grabs), 1)
            self.assertTrue(ctl.holding and robot.closed)
            self.assertNotEqual(robot.stage, 'Stopped')
