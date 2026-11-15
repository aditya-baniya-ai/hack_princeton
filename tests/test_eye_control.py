"""Runs without a robot or a HoloLens: the pretend arm from the driver stands in for the real one.
    python3 -m unittest discover -s tests
"""

import math
import sys
import time
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from eyereach import eye_control  # noqa: E402

CENTER, DROP_SPOT = (0.5, 0.5), (0.1, 0.9)


def run(ctl, gaze, until, seconds=12):
    """Feed the same gaze until the phase is `until` (or time runs out)."""
    end = time.monotonic() + seconds
    while ctl.phase != until and time.monotonic() < end:
        ctl.step(gaze)
        time.sleep(0.02)
    return ctl.phase


def new_arm():
    robot = eye_control.arm_link.VirtualArm()
    robot.start()
    return robot, eye_control.EyeControl(robot)


class EyeControlTest(unittest.TestCase):
    def test_grab_carry_and_drop(self):
        robot, ctl = new_arm()
        robot.held = object()                     # the pretend table: something is in the claw
        self.assertEqual(run(ctl, CENTER, "holding"), "holding")
        self.assertTrue(robot.closed)
        self.assertEqual(run(ctl, DROP_SPOT, "open"), "open")
        self.assertFalse(robot.closed)

    def test_empty_grab_is_not_held(self):
        robot, ctl = new_arm()                    # nothing in the claw
        run(ctl, CENTER, "grab")
        self.assertEqual(run(ctl, CENTER, "open"), "open")

    def test_no_grab_before_the_dwell(self):
        robot, ctl = new_arm()
        for _ in range(10):
            ctl.step(CENTER)
            time.sleep(0.02)
        self.assertEqual(ctl.phase, "open")

    def test_lost_gaze_holds_still(self):
        robot, ctl = new_arm()
        moves = []
        robot.follow = lambda u, v: moves.append((u, v))
        for _ in range(10):
            ctl.step(None)
        self.assertEqual((moves, ctl.phase), ([], "open"))


def guided(look=0.3):
    robot = eye_control.arm_link.VirtualArm()
    robot.start()
    return robot, eye_control.GuidedControl(robot, look=look)


def keep_looking(ctl, thing, until, seconds=20):
    """Look at the same thing until the phase is `until` (or time runs out)."""
    end = time.monotonic() + seconds
    while ctl.phase != until and time.monotonic() < end:
        ctl.step(thing)
        time.sleep(0.02)
    return ctl.phase


class GuidedControlTest(unittest.TestCase):
    def test_slow_by_default(self):
        robot, ctl = guided()
        self.assertLessEqual(robot.glide, 60.0)
        robot.carrying(True)
        self.assertLessEqual(robot.glide, 40.0)

    def test_nothing_moves_from_a_glance(self):
        robot, ctl = guided(look=1.5)
        moves = []
        robot.follow = lambda u, v: moves.append((u, v))
        for spot in ((0.2, 0.2), (0.8, 0.3), (0.5, 0.9), None, (0.3, 0.6)):   # eyes wandering around
            for _ in range(5):
                ctl.step(None if spot is None else ("map", spot))
                time.sleep(0.02)
        self.assertEqual((moves, ctl.phase, ctl.target), ([], "waiting", None))

    def test_look_go_pick_up_move_and_let_go(self):
        robot, ctl = guided()
        robot.held = object()                                    # something is on the table
        start = robot.gripper_area()[:2]
        here = (start[0], min(1.0, start[1] + 0.15))
        self.assertEqual(keep_looking(ctl, ("map", here), "there"), "there")
        ctl.step(None)                                           # look away, then at the button
        self.assertEqual(keep_looking(ctl, ("button", "pick up"), "picking"), "picking")
        self.assertEqual(keep_looking(ctl, None, "there"), "there")
        self.assertTrue(ctl.holding)
        self.assertEqual(ctl.buttons()[0][0], "let go")
        ctl.step(("map", robot.gripper_area()[:2]))  # look away before pressing the changed button
        ctl.step(None)
        self.assertEqual(keep_looking(ctl, ("button", "let go"), "letting go"), "letting go")
        self.assertEqual(keep_looking(ctl, None, "waiting"), "waiting")
        self.assertFalse(ctl.holding or robot.closed)

    def test_stop_while_moving(self):
        robot, ctl = guided()
        far = (0.9, 0.9)
        keep_looking(ctl, ("map", far), "moving")
        ctl.step(None)
        self.assertEqual(keep_looking(ctl, ("button", "stop"), "waiting", seconds=3), "waiting")
        self.assertIsNone(ctl.target)

    def test_greyed_out_buttons_do_nothing(self):
        robot, ctl = guided()
        self.assertFalse(ctl.can("voice"))                       # microphone not available
        for _ in range(30):
            ctl.step(("button", "voice"))
            time.sleep(0.02)
        self.assertEqual(ctl.phase, "waiting")

    def test_fix_my_aim_asks_for_calibration(self):
        robot, ctl = guided()
        keep_looking(ctl, ("button", "aim"), "never", seconds=0.6)
        self.assertTrue(ctl.wants_calibration)


class MouthOpenTest(unittest.TestCase):
    def test_hold_fires_once_until_closed_again(self):
        m = eye_control.MouthOpen()
        self.assertFalse(m.update(0.05, 0))
        self.assertFalse(m.update(0.5, 0.1))
        self.assertTrue(m.update(0.5, 0.5))
        self.assertFalse(m.update(0.5, 5))
        self.assertFalse(m.update(0.05, 5.1))
        self.assertFalse(m.update(0.5, 5.2))
        self.assertTrue(m.update(0.5, 5.6))

    def test_brief_opening_and_threshold_jitter_do_not_fire(self):
        m = eye_control.MouthOpen()
        for opening, now in ((0.05, 0), (0.5, 0.1), (0.25, 0.2),
                             (0.5, 0.3), (0.05, 0.4)):
            self.assertFalse(m.update(opening, now))

    def test_face_loss_disabled_and_initial_open_mouth_require_closing(self):
        for interrupt in (None, float('nan'), 0.05):
            m = eye_control.MouthOpen()
            self.assertFalse(m.update(0.5, 0))
            self.assertFalse(m.update(0.5, 1))
            m.update(0.05, 2)
            m.update(0.5, 2.1)
            m.update(interrupt, 2.2, enabled=interrupt != 0.05)
            self.assertFalse(m.update(0.5, 3))
            self.assertFalse(m.update(0.5, 4))
            m.update(0.05, 5)
            m.update(0.5, 5.1)
            self.assertTrue(m.update(0.5, 5.5))


class FreePlacementTest(unittest.TestCase):
    def test_mouth_pick_and_release_at_chosen_position(self):
        robot, ctl = guided()
        robot.held = object()                          # something is under the gripper
        ctl.step(None, mouth_open=True)
        self.assertEqual(ctl.phase, "picking")
        self.assertEqual(keep_looking(ctl, None, "there"), "there")
        self.assertTrue(ctl.holding and robot.closed)  # holding it: stays closed
        ctl.go((0.6, 0.5), "look", time.monotonic())
        self.assertEqual(keep_looking(ctl, None, "there"), "there")
        before = robot.gripper_area()
        ctl.step(None, mouth_open=True)
        self.assertEqual(keep_looking(ctl, None, "waiting"), "waiting")
        self.assertFalse(ctl.holding or robot.closed)
        self.assertLess(math.dist(before, robot.gripper_area()), 1.0)

    def test_stays_closed_after_a_pick_up_until_the_next_mouth_opening(self):
        robot, ctl = guided()                          # the finger check says "nothing held"
        ctl.step(None, mouth_open=True)
        self.assertEqual(keep_looking(ctl, None, "there", seconds=10), "there")
        for _ in range(50):                            # a while later, no mouth opening: still closed
            ctl.step(None)
            time.sleep(0.02)
        self.assertTrue(ctl.holding and robot.closed)
        self.assertEqual(ctl.buttons()[0][0], "let go")
        ctl.step(None, mouth_open=True)              # the next mouth opening opens it
        self.assertEqual(keep_looking(ctl, None, "waiting", seconds=10), "waiting")
        self.assertFalse(ctl.holding or robot.closed)

    def test_no_fixed_home_or_dropoff_controls(self):
        robot, ctl = guided()
        self.assertEqual([name for name, _ in ctl.buttons()], ["pick up", "stop", "voice", "aim"])
        for name in ("home", "dropoff"):
            self.assertFalse(ctl.can(name))
            ctl.press(name, time.monotonic())
        self.assertIsNone(ctl.target)


class GrabHeightTest(unittest.TestCase):
    def test_picks_up_at_a_safe_height(self):
        robot, ctl = guided()
        heights = []
        robot.grab = lambda height: heights.append(height) or False
        ctl.open_mouth(time.monotonic())
        self.assertEqual(heights, [eye_control.GRAB_MM])
        self.assertGreaterEqual(eye_control.GuidedControl(robot, grab_mm=5).grab_mm, 30.0)


class RestingEyesTest(unittest.TestCase):
    def test_looking_at_the_arm_itself_is_not_a_move(self):
        robot, ctl = guided()
        here = robot.gripper_area()[:2]
        for _ in range(60):                                      # stare at the claw for over a second
            ctl.step(("map", (here[0] + 0.02, here[1])))
            time.sleep(0.02)
        self.assertEqual((ctl.phase, [n for n, _ in ctl.events]), ("waiting", []))


class SessionLogTest(unittest.TestCase):
    def test_writes_and_sums_up(self):
        import tempfile
        from eyereach import session_log
        with tempfile.TemporaryDirectory() as d:
            log = session_log.SessionLog("tester", 1800, 34.0, 60.0, folder=Path(d))
            log.calibration(41.0, 90.0, [80, 100])
            log.event("pick", by="blink")
            log.event("picked", holding=True)
            log.event("task done", seconds=21.5)
            log.close()
            p = session_log.summary(Path(d))["tester"]
            self.assertEqual((p["picks"], p["blinks"], p["held"], p["tasks"]), (1, 1, 1, [21.5]))
            self.assertAlmostEqual(p["cal_deg"][0], 1.62, places=1)   # 90 px of 1800 on 34 cm, at 60 cm



if __name__ == "__main__":
    unittest.main()


class VoiceOrVisionTest(unittest.TestCase):
    def test_voice_mode_ignores_the_nose_and_vision_mode_ignores_voice_moves(self):
        import time
        robot = eye_control.arm_link.VirtualArm()
        robot.start()
        ctl = eye_control.GuidedControl(robot, movement="follow")
        now = time.monotonic()
        ctl.accept_after = now - 10
        self.assertFalse(ctl.voice(("jog", "left", 20.0), now))          # vision mode: "move left" ignored
        ctl.voice_available = True                                        # (the app sets it: microphone listening)
        ctl.press("voice", now)
        self.assertEqual(ctl.control_mode, "voice")
        ctl.accept_after = now - 10
        ctl.press("resume", now + 0.1)
        start = robot.gripper_area()[:2]
        for i in range(40):                                              # voice mode: the nose's spot ignored
            ctl.step(("map", (0.95, 0.5)), now + 0.2 + i * 0.03)
            time.sleep(0.03)
        self.assertLess(abs(robot.gripper_area()[0] - start[0]), 0.02)
        self.assertNotEqual(ctl.phase, "following")
