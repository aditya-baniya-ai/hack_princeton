"""The Nova Sonic voice module, without the network or audio (its events are plain JSON)."""

import asyncio
import json
import unittest

from agent import voice_sonic


class FakeStream:
    def __init__(self):
        self.sent = []

    async def send(self, chunk):
        self.sent.append(json.loads(chunk.value.bytes_.decode("utf-8"))["event"])


class EventTest(unittest.TestCase):
    def test_typed_turn_and_audio_events_have_the_right_shape(self):
        try:
            import aws_sdk_bedrock_runtime  # noqa: F401
        except ImportError:
            self.skipTest("the Sonic SDK lives in ~/eyereach_agent_venv (Python 3.12)")
        s = voice_sonic.SonicSession()
        s.stream = type("S", (), {"input_stream": FakeStream()})()
        asyncio.run(s.text('Hello "robot"'))
        asyncio.run(s.start_audio())
        asyncio.run(s.audio(b"\x00\x01" * 4))
        sent = s.stream.input_stream.sent
        self.assertEqual([next(iter(e)) for e in sent], ["contentStart", "textInput", "contentEnd", "contentStart", "audioInput"])
        self.assertEqual(sent[1]["textInput"]["content"], 'Hello "robot"')    # quotes survive (JSON, not f-strings)
        self.assertEqual(sent[3]["contentStart"]["audioInputConfiguration"]["sampleRateHertz"], 16000)
        self.assertEqual(sent[4]["audioInput"]["content"], "AAEAAQABAAE=")

    def test_the_prompt_never_claims_to_move_the_arm(self):
        self.assertIn("cannot move the arm yourself", voice_sonic.SYSTEM_PROMPT)


if __name__ == "__main__":
    unittest.main()
