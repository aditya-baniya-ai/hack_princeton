"""Quiet speech policy and real audio cutoffs, using fake speakers and no network."""
import asyncio
import io
import subprocess
import unittest
from unittest.mock import AsyncMock, Mock, patch

from agent import voice_sonic as sonic
from eyereach import run_webcam as app


class SpeechPolicyTest(unittest.TestCase):
    def test_necessary_mode_suppresses_routine_speech_before_synthesis(self):
        with patch.object(app.shutil, 'which', return_value='spd-say'), \
             patch.object(app.subprocess, 'Popen') as popen, patch.object(app.threading, 'Thread'):
            voice = app.Voice()
            self.assertFalse(voice.say('Left.'))
            self.assertFalse(voice.say('Hello. Let me explain everything.'))
            popen.assert_not_called()
            self.assertTrue(voice.say('Is it the orange?', necessary=True))
            self.assertFalse(voice.say('Is it the orange?', necessary=True))
            self.assertEqual(popen.call_count, 2)  # cancel previous dispatcher message, then speak

    def test_all_mode_still_truncates_text_and_limits_frequency(self):
        with patch.object(app.shutil, 'which', return_value='spd-say'), \
             patch.object(app.subprocess, 'Popen') as popen, patch.object(app.threading, 'Thread'):
            voice = app.Voice(speech='all')
            self.assertTrue(voice.say('Left. Now let me explain every movement in detail.'))
            spoken = popen.call_args.args[0][-1]
            self.assertEqual(spoken, 'Left.')
            self.assertFalse(voice.say('Right.'))
            self.assertTrue(voice.say('Stopped.', necessary=True))

    def test_off_mode_does_not_start_sonic_or_speak_essential_prompts(self):
        with patch.object(app.shutil, 'which', return_value='spd-say'), \
             patch.object(app.subprocess, 'Popen') as popen:
            voice = app.Voice(sonic=True, speech='off')
            self.assertFalse(voice.say('Stopped.', necessary=True))
            popen.assert_not_called()

    def test_fallback_audio_is_cancelled_at_cutoff_and_microphone_unmutes(self):
        process = Mock()
        process.wait.side_effect = subprocess.TimeoutExpired('spd-say', 3)
        with patch.object(app.shutil, 'which', return_value='spd-say'), \
             patch.object(app.subprocess, 'Popen', side_effect=[Mock(), process, Mock()]), \
             patch.object(app.threading, 'Thread') as thread:
            voice = app.Voice(max_seconds=3)
            voice.listener = Mock()
            voice.say('Is it the orange?', necessary=True)
            thread.call_args.kwargs['target']()
            process.wait.assert_called_once_with(timeout=3)
            process.terminate.assert_called_once()
            self.assertLessEqual(voice.listener.mute_until, app.time.monotonic() + 0.31)

    def test_sonic_receives_cutoff_and_only_short_prompts(self):
        proc = Mock()
        proc.poll.return_value = None
        proc.stdin = io.StringIO()
        with patch.object(app.AGENT_PYTHON.__class__, 'exists', return_value=True), \
             patch.object(app.subprocess, 'Popen', return_value=proc) as popen:
            voice = app.Voice(sonic=True, max_seconds=2)
            self.assertIn('--speech-max-seconds', popen.call_args.args[0])
            self.assertEqual(popen.call_args.args[0][-1], '2')
            voice.say('Is it the orange? Please give me a long answer.', necessary=True)
            self.assertEqual(proc.stdin.getvalue(), 'Is it the orange?\n')


class SonicAudioCutoffTest(unittest.TestCase):
    def test_speaker_writes_no_more_than_the_audio_budget(self):
        speaker = Mock()
        fake_sd = Mock()
        fake_sd.RawOutputStream.return_value.__enter__ = Mock(return_value=speaker)
        fake_sd.RawOutputStream.return_value.__exit__ = Mock(return_value=False)
        with patch.dict('sys.modules', sounddevice=fake_sd), \
             patch.object(sonic, 'reply_pcm', new=AsyncMock(return_value=b'\0' * sonic.OUT_RATE * 2 * 20)) as reply:
            sonic.speak_lines(lines=['Is it the orange?'], max_seconds=2)
            self.assertEqual(len(speaker.write.call_args.args[0]), 2 * sonic.OUT_RATE * 2)
            self.assertEqual(reply.call_args.kwargs['max_audio_seconds'], 2)

    def test_generation_stops_at_audio_budget_and_closes_session(self):
        async def run():
            session = Mock()
            session.active = False
            session.start = AsyncMock()
            session.text = AsyncMock()
            session.start_audio = AsyncMock()
            session.end = AsyncMock()
            session.finished = asyncio.Event()
            session.audio_out = asyncio.Queue()
            session.audio_out.put_nowait(b'\0' * sonic.OUT_RATE * 2 * 20)
            with patch.object(sonic, 'SonicSession', return_value=session):
                pcm = await sonic.reply_pcm('Hi.', max_audio_seconds=3)
            self.assertEqual(len(pcm), 3 * sonic.OUT_RATE * 2)
            session.end.assert_awaited_once_with(audio_open=True)
        asyncio.run(run())
