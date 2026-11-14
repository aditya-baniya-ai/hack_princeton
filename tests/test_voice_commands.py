import json
import queue
import sys
import time
import types
import unittest
from unittest.mock import Mock, patch

from eyereach.voice_commands import BothInputs, PHRASES, SonicInput, VoiceInput, VoiceEvent, parse
from eyereach.eye_control import GuidedControl, arm_link


class ParserTest(unittest.TestCase):
    def test_exact_wake_phrase_and_bounded_distances(self):
        self.assertEqual(parse('Robot move left.'), ('jog', 'left', 20.0))
        self.assertEqual(parse('robot a little right'), ('jog', 'right', 5.0))
        self.assertEqual(parse('Robot grab the item'), 'pick up')
        self.assertEqual(parse('robot release the item'), 'let go')
        self.assertEqual(parse('Stop!'), 'stop')
        for text in ('move left', 'grab', 'do not robot grab', 'robot move left then right',
                     'robot move left [unk]', 'robot move up', 'robot go faster', 'hello'):
            self.assertIsNone(parse(text), text)

    def test_all_supported_phrases_are_parseable(self):
        for phrase, action in PHRASES.items():
            self.assertEqual(parse(phrase), action)


class VoiceRoutingTest(unittest.TestCase):
    def setUp(self):
        self.robot = Mock()
        self.robot.error = None
        self.robot.hover = 120.0
        self.robot.closed = False
        self.robot.busy.return_value = False
        self.robot.gripper_area.return_value = (0.5, 0.5, 120.0)
        self.ctl = GuidedControl(self.robot, movement='follow')
        self.ctl.voice_available = True

    def test_hands_free_mode_switch_and_no_competing_gaze_or_blink(self):
        self.ctl.voice('voice', 1)
        self.robot.halt.assert_called_once()
        self.assertFalse(self.ctl.live_enabled)
        self.ctl.voice('resume', 2)
        self.ctl.step(('map', (0.9, 0.2)), now=3, double_blink=True)
        self.robot.follow.assert_not_called()
        self.robot.grab.assert_not_called()
        self.assertTrue(self.ctl.voice(('jog', 'left', 20.0), 4))
        self.robot.jog.assert_called_once_with('left', 20.0)

    def test_voice_grab_and_release_also_work_in_vision_mode(self):
        self.assertTrue(self.ctl.voice('pick up', 1))
        self.robot.grab.assert_called_once()
        self.robot.closed = True
        self.ctl.step(None, now=2)
        self.assertTrue(self.ctl.voice('let go', 3))
        self.robot.release_here.assert_called_once()

    def test_busy_stale_duplicate_and_pre_switch_utterances_do_nothing(self):
        self.ctl.voice('voice', 1)
        self.ctl.voice('resume', 2)
        self.assertFalse(self.ctl.voice(('jog', 'left', 20.0), 3, started=0.5, heard=3))
        self.assertFalse(self.ctl.voice(('jog', 'left', 20.0), 4, heard=2))
        self.ctl.voice(('jog', 'left', 20.0), 5)
        self.assertFalse(self.ctl.voice(('jog', 'right', 20.0), 5.1))
        self.ctl.step(None, now=5.2)  # operation completed
        self.assertFalse(self.ctl.voice(('jog', 'right', 20.0), 6, started=5.1))
        self.assertFalse(self.ctl.voice(('jog', 'left', 20.0), 5.3))
        self.assertEqual(self.robot.jog.call_count, 1)

    def test_stop_during_pickup_is_always_available_and_does_not_open(self):
        self.ctl.phase = 'picking'
        self.robot.closed = True
        self.assertTrue(self.ctl.can('stop'))
        self.ctl.voice('stop', 2)
        self.robot.halt.assert_called_once()
        self.robot.open_here.assert_not_called()
        self.assertTrue(self.ctl.holding)
        self.assertFalse(self.ctl.live_enabled)

    def test_low_arm_recovers_before_following_after_stop(self):
        self.robot.gripper_area.return_value = (0.5, 0.5, 50.0)
        self.robot.busy.return_value = True
        self.ctl.press('resume', 1)
        self.ctl.step(('map', (0.9, 0.2)), now=2)
        self.robot.recover.assert_called_once()
        self.robot.follow.assert_not_called()

    def test_no_voice_mode_when_microphone_unavailable(self):
        self.ctl.voice_available = False
        self.assertFalse(self.ctl.voice('voice', 1))
        self.assertEqual(self.ctl.control_mode, 'vision')


class DriverTest(unittest.TestCase):
    def setUp(self):
        self.robot = arm_link.VirtualArm()
        self.robot.hover = 120
        self.robot.start()

    def test_jogs_are_exact_bounded_and_follow_user_perspective(self):
        start = arm_link.arm.forward(self.robot.joints())[:2]
        self.assertTrue(self.robot.jog('right', 20.0))
        end = arm_link.arm.forward(self.robot.operation['goal'])[:2]
        self.assertAlmostEqual(end[0], start[0], places=4)
        self.assertAlmostEqual(end[1] - start[1], 20, places=4)
        self.assertTrue(self.robot.busy())
        self.assertFalse(self.robot.jog('left', 20.0))
        self.robot.halt()
        self.assertFalse(self.robot.busy())
        self.assertIsNone(self.robot.operation)

    def test_workspace_boundary_rejected_before_publishing(self):
        self.robot.state = self.robot.hover_joints(*arm_link.area_to_point(.5, 0))
        self.robot.send = Mock()
        self.assertFalse(self.robot.jog('toward me', 20.0))
        self.robot.send.assert_not_called()
        self.assertIn('workspace', self.robot.error)

    def test_stop_clears_pending_pickup_without_opening(self):
        self.robot.grab(50)
        self.robot.run_steps()
        self.robot.closed = True
        self.robot.halt()
        self.robot.run_steps()
        self.assertTrue(self.robot.closed)
        self.assertIsNone(self.robot.operation)
        self.assertEqual(self.robot.plan[0][1], self.robot.state)

    def test_timeout_holds_instead_of_advancing(self):
        self.robot.grab(50)
        self.robot.grip_ready = lambda closed: False
        with patch.object(arm_link.time, 'monotonic', return_value=self.robot.operation['deadline'] + 1):
            self.robot.run_steps()
        self.assertIn('timed out', self.robot.error)
        self.assertIsNone(self.robot.operation)


class SonicInputTest(unittest.TestCase):
    def test_commands_go_through_the_whitelist(self):
        sonic = SonicInput()
        sonic.read(['{"status": "Listening"}\n', 'a warning, not JSON\n',
                    '{"command": "a little left", "heard": "move it a bit to the left"}\n',
                    '{"command": "throw it at the wall"}\n', '{"command": "stop", "heard": "hold on"}\n'])
        events = sonic.poll()
        self.assertEqual([e.action for e in events], [('jog', 'left', 5.0), 'stop'])
        self.assertEqual(events[0].text, 'move it a bit to the left (Sonic)')
        self.assertEqual(sonic.status, 'Unavailable')                # the program ended
        self.assertEqual(sonic.error, 'Sonic stopped')

    def test_only_the_offline_listener_owns_errors(self):
        offline = Mock(status='Listening', error=None, poll=Mock(return_value=['a']))
        sonic = Mock(status='Unavailable', error='Sonic stopped', poll=Mock(return_value=['b']))
        both = BothInputs(offline, sonic)
        self.assertEqual((both.status, both.error, both.poll()), ('Listening', None, ['a', 'b']))
        both.close()
        sonic.close.assert_called_once()
        offline.close.assert_called_once()


class MicrophoneWorkerTest(unittest.TestCase):
    def run_recognition(self, results):
        listener = VoiceInput('.')
        for _ in results:
            listener.audio.put((time.monotonic(), b'\0\0'))
        iterator = iter(results)
        current = [None]

        class Recognizer:
            def __init__(self, *args):
                pass
            def SetWords(self, value):
                pass
            def AcceptWaveform(self, data):
                current[0] = next(iterator)
                return current[0][0]
            def Result(self):
                if current[0][2]:
                    listener.done.set()
                return json.dumps(current[0][1])
            def PartialResult(self):
                return json.dumps(current[0][1])

        stream = Mock(active=True)
        stream.__enter__ = Mock(return_value=stream)
        stream.__exit__ = Mock(return_value=False)
        modules = {
            'sounddevice': types.SimpleNamespace(RawInputStream=lambda **kw: stream),
            'vosk': types.SimpleNamespace(Model=lambda path: None, KaldiRecognizer=Recognizer),
        }
        with patch.dict(sys.modules, modules):
            listener._run()
        self.assertIsNone(listener.error)
        return listener.poll()

    def test_only_confident_final_movement_executes(self):
        events = self.run_recognition([
            (False, {'partial': 'robot move right'}, False),
            (True, {'text': 'robot move right', 'result': [{'conf': .99}]}, True),
        ])
        self.assertEqual([e.action for e in events], [('jog', 'right', 20.0)])
        self.assertEqual(self.run_recognition([
            (True, {'text': 'robot grab', 'result': [{'conf': .5}]}, True)]), [])

    def test_partial_stop_preempts_and_suppresses_later_action(self):
        events = self.run_recognition([
            (False, {'partial': 'robot move left stop'}, False),
            (True, {'text': 'robot grab', 'result': [{'conf': 1}]}, True),
        ])
        self.assertEqual([e.action for e in events], ['stop'])

    def test_audio_overflow_fails_closed(self):
        listener = VoiceInput('.')
        for _ in range(17):
            listener._audio(b'\0\0', 1, None, None)
        self.assertTrue(listener.done.is_set())
        self.assertIsNotNone(listener.error)


class GazeSwitchTest(unittest.TestCase):
    def setUp(self):
        self.robot = Mock()
        self.robot.error = None
        self.robot.hover = 120
        self.robot.closed = False
        self.robot.gripper_area.return_value = (.5, .5, 120)
        self.ctl = GuidedControl(self.robot, look=.5, movement='follow')
        self.ctl.voice_available = True

    def test_looking_at_mode_switch_does_not_switch_back(self):
        self.ctl.step(('button', 'voice'), now=0)
        self.ctl.step(('button', 'voice'), now=.6)
        self.assertEqual(self.ctl.control_mode, 'voice')
        self.ctl.step(('button', 'vision'), now=1)
        self.ctl.step(('button', 'vision'), now=2)
        self.assertEqual(self.ctl.control_mode, 'voice')
        self.ctl.step(('map', (.5, .5)), now=3)
        self.ctl.step(('button', 'vision'), now=4)
        self.ctl.step(('button', 'vision'), now=5)
        self.assertEqual(self.ctl.control_mode, 'vision')

    def test_looking_at_stop_cannot_automatically_resume(self):
        self.ctl.step(('button', 'stop'), now=0)
        self.ctl.step(('button', 'stop'), now=1)
        self.ctl.step(('button', 'resume'), now=2)
        self.ctl.step(('button', 'resume'), now=3)
        self.assertFalse(self.ctl.live_enabled)
        self.ctl.step(('map', (.5, .5)), now=4)
        self.ctl.step(('button', 'resume'), now=5)
        self.ctl.step(('button', 'resume'), now=6)
        self.assertTrue(self.ctl.live_enabled)


class GripperFeedbackTest(unittest.TestCase):
    def robot(self):
        robot = arm_link.RealArm.__new__(arm_link.RealArm)
        arm_link.ArmBase.__init__(robot, 20)
        robot.grip_future = robot.grip_handle = robot.grip_result = None
        robot.cancel_pending = False
        robot.node = Mock()
        robot.node.joints = {'gripper_left_joint': .019}
        return robot

    def test_rejected_command_reports_error(self):
        from concurrent.futures import Future
        robot = self.robot()
        robot.grip_future = Future()
        robot.grip_future.set_result(types.SimpleNamespace(accepted=False))
        self.assertFalse(robot.grip_ready(True))
        self.assertIn('rejected', robot.error)

    def test_waits_for_result_and_accepts_confirmed_contact(self):
        from concurrent.futures import Future
        robot = self.robot()
        robot.grip_result = Future()
        self.assertFalse(robot.grip_ready(True))
        robot.grip_result.set_result(types.SimpleNamespace(status=4,
                                    result=types.SimpleNamespace(reached_goal=False, stalled=True)))
        self.assertTrue(robot.grip_ready(True))

    def test_aborted_gripper_does_not_advance_pickup(self):
        from concurrent.futures import Future
        robot = self.robot()
        robot.grip_result = Future()
        robot.grip_result.set_result(types.SimpleNamespace(status=6))
        self.assertFalse(robot.grip_ready(True))
        self.assertIn('did not complete', robot.error)

    def test_stop_before_goal_acceptance_cancels_when_accepted(self):
        from concurrent.futures import Future
        robot = self.robot()
        robot.cancel_gripper()
        handle = Mock(accepted=True)
        robot.grip_future = Future()
        robot.grip_future.set_result(handle)
        robot.read_gripper()
        handle.cancel_goal_async.assert_called_once()
