"""Saved park destinations and complete automatic return/drop, without hardware."""
import json
import math
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tests.test_fetch import arm, arm_link, fetch
from eyereach import eye_control as ec
from eyereach.run_webcam import VISION_SPEEDS
import park


class ParkPositionTest(unittest.TestCase):
    def test_save_and_load_round_trip(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / 'park.json'
            self.assertIsNone(park.load(path))
            saved = park.position(162, 0, 50)
            park.save(saved, path)
            self.assertEqual(park.load(path), saved)
            self.assertEqual(json.loads(path.read_text())['drop_mm'], 50)

    def test_bad_coordinates_and_heights_are_rejected(self):
        for x, y, z in ((400, 0, 50), (float('nan'), 0, 50), (162, 0, 20), (162, 0, 120)):
            with self.subTest(x=x, y=y, z=z), self.assertRaises(ValueError):
                park.position(x, y, z)
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / 'park.json'
            path.write_text('not json')
            with self.assertRaises(ValueError):
                park.load(path)


class AutomaticParkConfigTest(unittest.TestCase):
    def test_saved_destination_is_enabled_without_a_flag(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / 'park.json'
            saved = park.position(114.2721535778497, -118.93029864467701, 50)
            park.save(saved, path)
            self.assertEqual(park.configure(path=path), saved)
            self.assertEqual(park.configure(True, path=path), saved)
            self.assertIsNone(park.configure(False, path=path))

    def test_missing_destination_does_not_invent_coordinates(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / 'park.json'
            self.assertIsNone(park.configure(path=path))
            with self.assertRaises(ValueError):
                park.configure(True, path=path)

    def test_saved_release_height_can_be_overridden_without_moving_xy(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / 'park.json'
            park.save(park.position(162, 20, 50), path)
            self.assertEqual(park.configure(drop_mm=45, path=path), park.position(162, 20, 45))

    def test_invalid_saved_destination_fails_instead_of_using_another_drop_spot(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / 'park.json'
            path.write_text('{"x":400,"y":0,"drop_mm":50}')
            with self.assertRaises(ValueError):
                park.configure(path=path)
            self.assertIsNone(park.configure(False, path=path))
        with self.assertRaises(ValueError):
            park.configure(False, coordinates=(162, 20))


class ParkSequenceTest(unittest.TestCase):
    def simulate(self, enabled=True, pickup_area=(0.4, 0.45), stop_during=None, destination=None):
        clock = [10.0]
        stages = []
        with patch.object(arm_link.time, 'monotonic', side_effect=lambda: clock[0]):
            robot = arm_link.VirtualArm()
            robot.hover = ec.HOVER_MM
            robot.start()
            robot.held = object()
            initial = robot.hover_joints(*arm_link.area_to_point(*pickup_area))
            robot.state = robot.command = list(initial)
            ctl = ec.GuidedControl(robot, movement='follow', vision_speeds=VISION_SPEEDS)
            ctl.park_position = park.position(162, 20, 45) if destination is None else destination
            ctl.park_enabled = enabled
            releases, grabs = [], []
            release, grab = robot.release, robot.grab
            def record_release(height):
                releases.append((*arm.forward(robot.joints())[:3], height))
                return release(height)
            robot.release = record_release
            robot.grab = lambda height: grabs.append(height) or grab(height)
            ctl.step(None, now=clock[0], grab_action=True, vision_movement=False)
            saw_park = False
            for _ in range(2400):
                clock[0] += 0.05
                if ctl.parking:
                    saw_park = True
                if stop_during and ctl.parking and robot.stage == stop_during:
                    ctl.press('stop', clock[0])
                    for _ in range(20):
                        clock[0] += 0.05
                        ctl.step(None, now=clock[0], vision_movement=False)
                    break
                # The face/mouth gate stays disabled for the whole automatic return.
                ctl.step(None, now=clock[0], vision_movement=False)
                stages.append(robot.stage)
                if enabled and saw_park and not ctl.parking:
                    break
                if not enabled and ctl.holding:
                    break
            return robot, ctl, stages, releases, grabs, saw_park

    def test_pickup_carries_to_park_then_lowers_releases_and_lifts(self):
        for source in ((0.4, 0.45), (0.6, 0.6)):
            with self.subTest(source=source):
                robot, ctl, stages, releases, grabs, saw_park = self.simulate(pickup_area=source)
                self.assertTrue(saw_park)
                self.assertEqual(grabs, [ec.GRAB_MM])
                self.assertEqual(len(releases), 1)
                x, y, z, drop_mm = releases[0]
                self.assertLess(math.dist((x, y), (162, 20)), 4)
                self.assertGreater(z, 110)  # travels high; release() itself performs the lowering
                self.assertEqual(drop_mm, 45)
                gx, gy, gz, _ = arm.forward(robot.joints())
                self.assertLess(math.dist((gx, gy), (162, 20)), 4)
                self.assertAlmostEqual(gz, ec.HOVER_MM, delta=5)
                self.assertFalse(robot.closed or ctl.holding or ctl.parking)
                self.assertTrue(ctl.live_enabled)
                self.assertEqual(ctl.phase, 'waiting')
                self.assertEqual(stages.count('Stopped'), 0)
                self.assertTrue(any(name == 'parked' and not details['problem'] for name, details in ctl.events))

    def test_automatically_loaded_jetson_destination_receives_the_pickup(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / 'park.json'
            park.save(park.position(114.2721535778497, -118.93029864467701, 50), path)
            destination = park.configure(path=path)
            robot, ctl, _, releases, grabs, saw_park = self.simulate(destination=destination)
            self.assertTrue(saw_park)
            self.assertEqual(len(grabs), 1)
            self.assertEqual(len(releases), 1)
            self.assertLess(math.dist(releases[0][:2], (destination['x'], destination['y'])), 4)
            self.assertEqual(releases[0][3], 50)
            self.assertFalse(ctl.holding or robot.closed or ctl.parking)

    def test_mode_off_keeps_manual_pickup_behavior(self):
        robot, ctl, _, releases, grabs, saw_park = self.simulate(enabled=False)
        self.assertFalse(saw_park)
        self.assertEqual(grabs, [ec.GRAB_MM])
        self.assertTrue(ctl.holding and robot.closed)
        self.assertEqual(releases, [])

    def test_stop_cancels_park_before_dropping(self):
        robot, ctl, _, releases, _, saw_park = self.simulate(stop_during='Going to park')
        self.assertTrue(saw_park)
        self.assertFalse(ctl.live_enabled or ctl.parking)
        self.assertIsNone(ctl.job)
        self.assertTrue(robot.closed)
        self.assertEqual(releases, [])

    def test_stop_during_lowering_prevents_release(self):
        robot, ctl, _, releases, _, saw_park = self.simulate(stop_during='Lowering')
        self.assertTrue(saw_park)
        self.assertEqual(len(releases), 1)
        self.assertFalse(ctl.live_enabled or ctl.parking)
        self.assertTrue(robot.closed)
        self.assertIsNone(robot.operation)

    def test_park_timeout_holds_the_object_and_requires_resume(self):
        robot = arm_link.VirtualArm()
        robot.hover = ec.HOVER_MM
        robot.start()
        robot.closed = True
        ctl = ec.GuidedControl(robot, movement='follow')
        ctl.holding, ctl.parking, ctl.phase = True, True, 'fetching'
        ctl.job = park.ParkJob(robot, park.position(162, 20))
        ctl.job.GLIDE_SECONDS = -1
        ctl.step(None, vision_movement=False)
        self.assertFalse(ctl.live_enabled or ctl.parking)
        self.assertTrue(robot.closed and ctl.holding)
        self.assertIsNone(ctl.job)
        self.assertTrue(any(name == 'parked' and details['problem'] for name, details in ctl.events))

    def test_agent_fetch_uses_park_destination_and_separate_drop_height(self):
        robot = arm_link.VirtualArm()
        robot.hover = ec.HOVER_MM
        robot.start()
        ctl = ec.GuidedControl(robot, vision_speeds=VISION_SPEEDS)
        ctl.park_enabled, ctl.park_position = True, park.position(162, 20, 45)
        job = ctl.make_fetch_job(162, -20)
        self.assertIsInstance(job, park.ParkJob)
        self.assertEqual(job.plan[-1], ('down', 'release'))
        self.assertLess(math.dist(job.plan[-2][1], (162, 20)), 1e-6)
        self.assertEqual(job.grab_mm, ec.GRAB_MM)
        self.assertEqual(job.drop_mm, 45)
