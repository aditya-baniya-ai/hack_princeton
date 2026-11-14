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
    h = HeadTilt()
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


class NodTest(unittest.TestCase):
    def test_a_quick_nod_counts_once_and_never_moves(self):
        h = centred()
        out, _ = run(h, [((3, -6), 1.0), ((3, -6 - 8), 0.15), ((3, -6 - 16), 0.2), ((3, -6 - 8), 0.1), ((3, -6), 1.0)])
        self.assertEqual(sum(o.nod for o in out), 1)
        self.assertTrue(all(o.direction is None for o in out))

    def test_a_slow_tilt_down_is_a_move_not_a_nod(self):
        h = centred()
        out, _ = run(h, [((3, -6 - 18), 3.0)])
        self.assertEqual(sum(o.nod for o in out), 0)
        self.assertEqual(out[-1].label, "toward you")

    def test_a_small_dip_or_a_sideways_dip_is_not_a_nod(self):
        h = centred()
        out, _ = run(h, [((3, -6 - 7), 0.3), ((3, -6), 0.5)])                     # too shallow
        self.assertEqual(sum(o.nod for o in out), 0)
        out, _ = run(h, [((3 + 25, -6 - 16), 0.3), ((3 + 25, -6), 0.5)])          # turned sideways
        self.assertEqual(sum(o.nod for o in out), 0)

    def test_two_nods_need_a_pause(self):
        h = centred()
        nod = [((3, -6 - 16), 0.25), ((3, -6), 0.25)]
        out, _ = run(h, nod + nod)                               # straight after each other: one counts
        self.assertEqual(sum(o.nod for o in out), 1)
        out, _ = run(h, [((3, -6), 1.5)] + nod, start=out and 10.0)
        self.assertEqual(sum(o.nod for o in out), 1)


if __name__ == "__main__":
    unittest.main()


class TiltDrivesTheArmTest(unittest.TestCase):
    """The app's per-frame tilt step with the pretend arm: real time, simulated head angles."""

    def frames(self, ctl, head, pose, seconds, nod_ok=True):
        import time
        from eyereach import run_webcam
        end = time.monotonic() + seconds
        while time.monotonic() < end:
            now = time.monotonic()
            tilt = head.update(pose, now)
            if tilt.nod and nod_ok:
                ctl.blink(now)
            run_webcam.tilt_step(ctl, tilt, pose is not None, False, now)
            time.sleep(0.03)

    def test_turn_left_moves_left_centre_stops_nod_picks_and_lets_go(self):
        from eyereach import eye_control
        robot = eye_control.arm_link.VirtualArm()
        robot.start()
        robot.held = object()
        ctl = eye_control.GuidedControl(robot, movement="follow")
        ctl.tilt = True
        head = HeadTilt()
        head.set_centre(0.0, 0.0)
        u0 = robot.gripper_area()[0]
        self.frames(ctl, head, (20.0, 0.0), 2.0)               # turn to the person's left
        u1 = robot.gripper_area()[0]
        self.assertLess(u1, u0 - 0.05)                          # the arm went left (u down)
        self.frames(ctl, head, (0.0, 0.0), 1.0)                # back to the centre
        u2 = robot.gripper_area()[0]
        self.frames(ctl, head, (0.0, 0.0), 1.0)
        self.assertAlmostEqual(robot.gripper_area()[0], u2, delta=0.01)   # stopped
        self.frames(ctl, head, (0.0, 0.0), 1.3)                # (a pause before a nod counts)
        for pitch, seconds in ((-16.0, 0.3), (0.0, 0.2)):      # a nod
            self.frames(ctl, head, (0.0, pitch), seconds)
        self.assertEqual(ctl.phase, "picking")
        self.frames(ctl, head, (0.0, 0.0), 5.0)
        self.assertTrue(ctl.holding and robot.closed)           # holding after the nod
        self.frames(ctl, head, (0.0, 0.0), 1.5)
        for pitch, seconds in ((-16.0, 0.3), (0.0, 0.2)):      # nod again: let go
            self.frames(ctl, head, (0.0, pitch), seconds)
        self.frames(ctl, head, (0.0, 0.0), 3.0)
        self.assertFalse(ctl.holding or robot.closed)

    def test_no_face_stops_the_arm(self):
        from eyereach import eye_control
        robot = eye_control.arm_link.VirtualArm()
        robot.start()
        ctl = eye_control.GuidedControl(robot, movement="follow")
        ctl.tilt = True
        head = HeadTilt()
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
        h = HeadTilt()
        h.set_centre(0.0, 0.0, 0.0)
        out, _ = run(h, [((0.0, 0.0, 20.0), 1.5)])          # ear toward the right shoulder
        self.assertEqual(out[-1].label, "right")
        self.assertGreater(out[-1].direction[0], 0.9)       # u up = the person's right
        h = HeadTilt()
        h.set_centre(0.0, 0.0, 0.0)
        out, _ = run(h, [((0.0, 0.0, -20.0), 1.5)])
        self.assertEqual(out[-1].label, "left")

    def test_looking_right_and_tilting_right_agree(self):
        h = HeadTilt()
        h.set_centre(0.0, 0.0, 0.0)
        out, _ = run(h, [((-20.0, 0.0, 0.0), 1.5)])         # look right (yaw < 0)
        self.assertEqual(out[-1].label, "right")
        h = HeadTilt()
        h.set_centre(0.0, 0.0, 0.0)
        out, _ = run(h, [((-12.0, 0.0, 25.0), 1.5)])        # both: the stronger (tilt) wins, same way
        self.assertEqual(out[-1].label, "right")


class ScreenAgreesWithTheHeadTest(unittest.TestCase):
    def test_tilt_up_goes_up_on_the_map_and_right_goes_right(self):
        from eyereach import run_webcam
        run_webcam.MAP["away_up"] = True
        try:
            h = HeadTilt()
            h.set_centre(0.0, 0.0, 0.0)
            up, _ = run(h, [((0.0, 20.0, 0.0), 1.5)])               # tilt up = away (v up)
            h2 = HeadTilt()
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
