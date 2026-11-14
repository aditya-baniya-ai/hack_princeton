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

    def test_tool_result_and_tools_in_the_prompt(self):
        s = voice_sonic.SonicSession(tools=[voice_sonic.COMMAND_TOOL])
        s.stream = type("S", (), {"input_stream": FakeStream()})()
        try:
            asyncio.run(s.tool_result("use-1", {"status": "sent"}))
        except ImportError:
            self.skipTest("the Sonic SDK lives in ~/eyereach_agent_venv (Python 3.12)")
        start, result, end = s.stream.input_stream.sent
        self.assertEqual(start["contentStart"]["toolResultInputConfiguration"]["toolUseId"], "use-1")
        self.assertEqual(start["contentStart"]["role"], "TOOL")
        self.assertEqual(json.loads(result["toolResult"]["content"]), {"status": "sent"})
        self.assertIn("contentEnd", end)

    def test_every_sonic_command_is_a_whitelisted_phrase(self):
        from eyereach.voice_commands import parse
        for command in voice_sonic.COMMANDS:
            self.assertIsNotNone(parse("robot " + command), command)

    def test_command_out_prints_only_known_commands(self):
        import contextlib
        import io
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            on_tool = voice_sonic.command_out(["move it a bit left"])
            on_tool("robot_command", {"command": "a little left"})
            self.assertEqual(on_tool("robot_command", {"command": "throw it"}), {"status": "unknown command"})
            on_tool("other_tool", {"command": "grab"})
        self.assertEqual([json.loads(line) for line in out.getvalue().splitlines()],
                         [{"command": "a little left", "heard": "move it a bit left"}])
        self.assertIn("Never say the word robot", voice_sonic.COMMAND_PROMPT)   # the offline listener needs "robot ..."

    def test_the_prompt_never_claims_to_move_the_arm(self):
        self.assertIn("cannot move the arm yourself", voice_sonic.SYSTEM_PROMPT)


if __name__ == "__main__":
    unittest.main()
