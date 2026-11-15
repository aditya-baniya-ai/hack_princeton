"""Head tilt control: directions, the hold before moving, and nods (simulated head angles, 20 frames a second)."""

import unittest

from eyereach.head_tilt import HeadTilt, HOLD


def run(tilt, poses, start=0.0, fps=20):
    """poses: [((yaw, pitch) or None, seconds)]. Returns every Tilt, and the last time."""
    out, t = [], start
    for pose, seconds in poses:
        for _ in range(max(1, round(seconds * fps))):
            t += 1 / fps
            out.append(tilt.update(pose, t))
    return out, t


def centred():
    h = HeadTilt(mode="joystick")
    h.set_centre(3.0, -6.0)                  # the person's straight ahead isn't exactly 0, 0
    return h


class DirectionTest(unittest.TestCase):
    def test_turn_left_moves_left_after_the_hold(self):
        h = centred()
        out, _ = run(h, [((3 + 20, -6), HOLD + 0.3)])          # yaw +20: face turned to the person's left
        self.assertIsNone(out[0].direction)                     # not straight away
        self.assertEqual(out[-1].label, "left")
        self.assertLess(out[-1].direction[0], -0.9)             # u goes down = the person's left
        self.assertGreater(out[-1].speed, 0.5)

    def test_right_up_down(self):
        for pose, label, axis, sign in (((3 - 20, -6), "right", 0, 1), ((3, -6 + 20), "away", 1, 1),
                                        ((3, -6 - 20), "toward you", 1, -1)):
            h = centred()
            out, _ = run(h, [(pose, 2.0)])
            self.assertEqual(out[-1].label, label)
            self.assertEqual(out[-1].direction[axis] > 0, sign > 0)

    def test_small_turns_and_the_centre_stop(self):
        h = centred()
        out, _ = run(h, [((3 + 5, -6 + 4), 2.0)])               # inside the dead zone
        self.assertTrue(all(o.direction is None for o in out))
        out, _ = run(h, [((3 + 20, -6), 2.0), ((3, -6), 0.1)])  # back to the centre: stops at once
        self.assertIsNone(out[-1].direction)

    def test_further_is_faster(self):
        slow, _ = run(centred(), [((3 + 12, -6), 2.0)])
        fast, _ = run(centred(), [((3 + 30, -6), 2.0)])
        self.assertLess(slow[-1].speed, fast[-1].speed)
        self.assertEqual(fast[-1].speed, 1.0)

    def test_no_face_means_no_movement(self):
        h = centred()
        out, _ = run(h, [((3 + 20, -6), 2.0), (None, 0.4)])   # lost for longer than the grace
        self.assertIsNone(out[-1].direction)
        self.assertEqual(out[-1].label, "no face")


class NoNodGestureTest(unittest.TestCase):
    def test_quick_nods_only_produce_position_updates(self):
        for mode in ("follow", "joystick"):
            h = HeadTilt(mode=mode)
            h.set_centre(0, 0)
            out, _ = run(h, [((0, 0), 1), ((0, -16), 0.25), ((0, 0), 0.5)] * 2)
            self.assertTrue(all(not hasattr(o, "nod") for o in out))


if __name__ == "__main__":
    unittest.main()


class TiltDrivesTheArmTest(unittest.TestCase):
    """The app's per-frame tilt step with the pretend arm: real time, simulated head angles."""

    def frames(self, ctl, head, pose, seconds):
        import time
        from eyereach import run_webcam
        end = time.monotonic() + seconds
        while time.monotonic() < end:
            now = time.monotonic()
            tilt = head.update(pose, now)
            run_webcam.tilt_step(ctl, tilt, pose is not None, False, now)
            time.sleep(0.03)

    def test_turn_left_moves_left_centre_stops_mouth_picks_and_lets_go(self):
        from eyereach import eye_control
        robot = eye_control.arm_link.VirtualArm()
        robot.start()
        robot.held = object()
        ctl = eye_control.GuidedControl(robot, movement="follow")
        ctl.tilt = True
        head = HeadTilt(mode="joystick")
        head.set_centre(0.0, 0.0)
        u0 = robot.gripper_area()[0]
        self.frames(ctl, head, (20.0, 0.0), 2.0)               # turn to the person's left
        u1 = robot.gripper_area()[0]
        self.assertLess(u1, u0 - 0.05)                          # the arm went left (u down)
        self.frames(ctl, head, (0.0, 0.0), 1.0)                # back to the centre
        u2 = robot.gripper_area()[0]
        self.frames(ctl, head, (0.0, 0.0), 1.0)
        self.assertAlmostEqual(robot.gripper_area()[0], u2, delta=0.01)   # stopped
        # A nod no longer grabs; only opening the mouth requests a pickup.
        for pitch, seconds in ((-16.0, 0.3), (0.0, 0.2)):
            self.frames(ctl, head, (0.0, pitch), seconds)
        self.assertFalse(ctl.holding or robot.closed)
        from eyereach.run_webcam import tilt_step
        import time
        now = time.monotonic()
        tilt_step(ctl, head.update((0, 0), now), True, False, now, mouth_action=True)
        self.assertEqual(ctl.phase, "picking")
        self.frames(ctl, head, (0.0, 0.0), 5.0)
        self.assertTrue(ctl.holding and robot.closed)           # holding after the mouth action
        self.frames(ctl, head, (0.0, 0.0), 1.5)
        now = time.monotonic()
        tilt_step(ctl, head.update((0, 0), now), True, False, now, mouth_action=True)
        self.frames(ctl, head, (0.0, 0.0), 3.0)
        self.assertFalse(ctl.holding or robot.closed)

    def test_no_face_stops_the_arm(self):
        from eyereach import eye_control
        robot = eye_control.arm_link.VirtualArm()
        robot.start()
        ctl = eye_control.GuidedControl(robot, movement="follow")
        ctl.tilt = True
        head = HeadTilt(mode="joystick")
        head.set_centre(0.0, 0.0)
        self.frames(ctl, head, (-20.0, 0.0), 1.5)              # moving right
        self.frames(ctl, head, None, 0.3)                       # face lost
        u = robot.gripper_area()[0]
        self.frames(ctl, head, None, 1.0)
        self.assertAlmostEqual(robot.gripper_area()[0], u, delta=0.01)


class DropOutTest(unittest.TestCase):
    def test_a_short_drop_out_keeps_going_a_long_one_stops(self):
        h = centred()
        out, t = run(h, [((3 + 20, -6), 1.5), (None, 0.15)])   # one or two lost frames
        self.assertEqual(out[-1].label, "left")
        out, _ = run(h, [(None, 0.5)], start=t)                 # lost for longer
        self.assertEqual(out[-1].label, "no face")
        self.assertIsNone(out[-1].direction)


class SidewaysTiltTest(unittest.TestCase):
    def test_tilting_the_head_right_or_left_moves_right_or_left(self):
        h = HeadTilt(mode="joystick")
        h.set_centre(0.0, 0.0, 0.0)
        out, _ = run(h, [((0.0, 0.0, 20.0), 1.5)])          # ear toward the right shoulder
        self.assertEqual(out[-1].label, "right")
        self.assertGreater(out[-1].direction[0], 0.9)       # u up = the person's right
        h = HeadTilt(mode="joystick")
        h.set_centre(0.0, 0.0, 0.0)
        out, _ = run(h, [((0.0, 0.0, -20.0), 1.5)])
        self.assertEqual(out[-1].label, "left")

    def test_looking_right_and_tilting_right_agree(self):
        h = HeadTilt(mode="joystick")
        h.set_centre(0.0, 0.0, 0.0)
        out, _ = run(h, [((-20.0, 0.0, 0.0), 1.5)])         # look right (yaw < 0)
        self.assertEqual(out[-1].label, "right")
        h = HeadTilt(mode="joystick")
        h.set_centre(0.0, 0.0, 0.0)
        out, _ = run(h, [((-12.0, 0.0, 25.0), 1.5)])        # both: the stronger (tilt) wins, same way
        self.assertEqual(out[-1].label, "right")


class ScreenAgreesWithTheHeadTest(unittest.TestCase):
    def test_tilt_up_goes_up_on_the_map_and_right_goes_right(self):
        from eyereach import run_webcam
        run_webcam.MAP["away_up"] = True
        try:
            h = HeadTilt(mode="joystick")
            h.set_centre(0.0, 0.0, 0.0)
            up, _ = run(h, [((0.0, 20.0, 0.0), 1.5)])               # tilt up = away (v up)
            h2 = HeadTilt(mode="joystick")
            h2.set_centre(0.0, 0.0, 0.0)
            right, _ = run(h2, [((-20.0, 0.0, 0.0), 1.5)])          # look right
            mid = run_webcam.to_screen(0.5, 0.5)
            go_up = run_webcam.to_screen(0.5, 0.5 + 0.2 * up[-1].direction[1])
            go_right = run_webcam.to_screen(0.5 + 0.2 * right[-1].direction[0], 0.5)
            self.assertLess(go_up[1], mid[1])                       # higher on the screen
            self.assertGreater(go_right[0], mid[0])                 # further right on the screen
            back = run_webcam.to_area(run_webcam.to_screen(0.3, 0.8))
            self.assertAlmostEqual(back[0], 0.3, places=2)
            self.assertAlmostEqual(back[1], 0.8, places=2)                # (whole pixels)
        finally:
            run_webcam.MAP["away_up"] = False


class NoseGuidesTest(unittest.TestCase):
    """The nose-between-the-eyes reading: right directions, and the same wherever the face is in the picture."""

    @staticmethod
    def face(yaw=0.0, pitch=0.0, x=300.0):
        import math
        import numpy as np
        from eyereach.gaze_webcam import LEFT, NOSE, RIGHT
        pts = np.zeros((478, 3))
        pts[LEFT["outer"]], pts[RIGHT["outer"]], pts[NOSE] = (45, 0, 0), (-45, 0, 0), (0, 55, -45)
        a, b = math.radians(yaw), math.radians(pitch)
        ry = np.array([[math.cos(a), 0, -math.sin(a)], [0, 1, 0], [math.sin(a), 0, math.cos(a)]])
        rx = np.array([[1, 0, 0], [0, math.cos(b), math.sin(b)], [0, -math.sin(b), math.cos(b)]])
        out = pts @ (rx @ ry).T
        out[:, 0] += x
        out[:, 1] += 200
        return out[:, :2]

    def angles(self, **kw):
        from eyereach.gaze_webcam import nose_offset
        from eyereach.head_tilt import nose_angles
        c, a = nose_angles(nose_offset(self.face())), nose_angles(nose_offset(self.face(**kw)))
        return a[0] - c[0], a[1] - c[1]

    def test_straight_reads_zero_anywhere_in_the_picture(self):
        for x in (80.0, 300.0, 560.0):
            yaw, pitch = self.angles(x=x)
            self.assertAlmostEqual(yaw, 0.0, delta=0.5)
            self.assertAlmostEqual(pitch, 0.0, delta=0.5)

    def test_turns_and_nods_read_the_right_way(self):
        yaw, pitch = self.angles(yaw=20.0)                    # turned to the person's left
        self.assertAlmostEqual(yaw, 20.0, delta=3.0)
        self.assertLess(abs(pitch), 2.0)                      # a turn doesn't leak into up/down
        self.assertLess(self.angles(yaw=-20.0)[0], -17.0)     # right
        self.assertGreater(self.angles(pitch=15.0)[1], 10.0)  # up
        self.assertLess(self.angles(pitch=-15.0)[1], -10.0)   # down


class FollowTest(unittest.TestCase):
    """The default: the arm follows the nose live. Straight = the middle; REACH degrees = the edge; no holding."""

    def test_straight_is_the_middle_and_turning_moves_the_spot_at_once(self):
        from eyereach.head_tilt import REACH
        h = HeadTilt()
        h.set_centre(0.0, 0.0, 0.0)
        out, _ = run(h, [((0.0, 0.0, 0.0), 0.5)])
        self.assertEqual(out[-1].target, (0.5, 0.5))
        self.assertEqual(out[-1].label, "centre")
        out, _ = run(h, [((-REACH / 2, 0.0, 0.0), 0.4)])         # look right a bit: no holding, already there
        self.assertAlmostEqual(out[-1].target[0], 0.75, delta=0.02)
        self.assertEqual(out[-1].label, "right")
        out, _ = run(h, [((0.0, REACH, 0.0), 0.6)])               # look up: away, at the edge
        self.assertAlmostEqual(out[-1].target[1], 1.0, delta=0.02)
        out, _ = run(h, [((0.0, 0.0, 0.0), 0.6)])                 # back to straight: back to the middle
        self.assertAlmostEqual(out[-1].target[0], 0.5, delta=0.02)
        self.assertAlmostEqual(out[-1].target[1], 0.5, delta=0.02)

    def test_small_jitter_stays_put(self):
        h = HeadTilt()
        h.set_centre(0.0, 0.0, 0.0)
        out, _ = run(h, [((0.8, -0.6, 0.0), 0.05), ((-0.7, 0.9, 0.0), 0.05)] * 10)
        self.assertTrue(all(o.target == (0.5, 0.5) for o in out))



class FollowDrivesTheArmTest(unittest.TestCase):
    def test_the_pretend_arm_follows_the_nose_right_and_back(self):
        import time
        from eyereach import eye_control, run_webcam
        robot = eye_control.arm_link.VirtualArm()
        robot.start()
        robot.set_speeds(*run_webcam.FOLLOW_SPEEDS)
        ctl = eye_control.GuidedControl(robot, movement="follow")
        ctl.tilt = True
        h = HeadTilt()
        h.set_centre(0.0, 0.0, 0.0)

        def frames(pose, seconds):
            end = time.monotonic() + seconds
            while time.monotonic() < end:
                run_webcam.tilt_step(ctl, h.update(pose, time.monotonic()), True, False, time.monotonic())
                time.sleep(0.03)
        frames((0.0, 0.0, 0.0), 1.5)
        u0 = robot.gripper_area()[0]
        frames((-10.0, 0.0, 0.0), 2.5)                            # nose a little to the right
        u1 = robot.gripper_area()[0]
        self.assertGreater(u1, u0 + 0.1)                           # the arm went right
        frames((0.0, 0.0, 0.0), 2.5)                               # nose back: the arm comes back
        self.assertAlmostEqual(robot.gripper_area()[0], u0, delta=0.05)


class SettleTest(unittest.TestCase):
    def test_a_badly_taken_straight_corrects_itself_but_a_real_turn_doesnt_drag_it(self):
        h = HeadTilt()
        h.set_centre(8.0, 0.0, 0.0)                   # taken while glancing 8 deg off
        out, t = run(h, [((0.0, 0.0, 0.0), 40.0)])     # then looking straight for 40 s
        self.assertLess(abs(h.centre[0]), 2.0)
        self.assertEqual(out[-1].target, (0.5, 0.5))
        h = HeadTilt()
        h.set_centre(0.0, 0.0, 0.0)
        run(h, [((-25.0, 0.0, 0.0), 20.0)])            # a long deliberate turn
        self.assertAlmostEqual(h.centre[0], 0.0, delta=0.5)
