"""Talk to EyeReach with Amazon Nova 2 Sonic: it listens on the microphone and answers out loud.

    python -m agent.voice_sonic                     talk: microphone in, speaker out; Ctrl+C ends
    python -m agent.voice_sonic --text "Hello"      one typed sentence; the spoken reply is saved to a WAV
                                                    file (no microphone or speaker needed)
    python -m agent.voice_sonic --list              list the audio devices

Runs in ~/eyereach_agent_venv: the Sonic SDK (aws_sdk_bedrock_runtime) needs Python 3.12.
Speech-to-speech in one model, on Amazon Bedrock (event format from Amazon's Nova 2 Sonic sample).

Half duplex: while Sonic is speaking, silence is sent instead of the microphone, so it never hears
itself through the speaker. It only talks: it does not move the arm. STOP for the arm stays offline
(the gaze STOP button or the offline voice "Stop" in the eye app), so it works without the network.

Environment variables (from .env):
    AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY   Bedrock access
    AWS_REGION                                 default us-east-1 (Sonic is listed there)
    SONIC_MODEL_ID                             default amazon.nova-2-sonic-v1:0
    SONIC_VOICE                                default matthew
"""

import argparse
import asyncio
import base64
import json
import os
import sys
import time
import uuid
import wave

MODEL = os.environ.get("SONIC_MODEL_ID", "amazon.nova-2-sonic-v1:0")
REGION = os.environ.get("AWS_REGION", "us-east-1")
VOICE = os.environ.get("SONIC_VOICE", "matthew")
IN_RATE, OUT_RATE = 16000, 24000          # microphone in, speech out (16-bit mono both)
CHUNK = 1024                              # microphone frames per send
SYSTEM_PROMPT = (
    "You are EyeReach, the friendly voice of a robot arm that helps a person who cannot move their arms. "
    "They move the arm by pointing with their head at a map on the screen, and blink twice to pick something "
    "up or let it go. Answer in one or two short, warm sentences. You cannot move the arm yourself, so never "
    "say that you moved, grabbed or dropped anything. If they want the arm to stop, tell them to say Stop or "
    "look at the STOP button.")


def event(name, **body):
    return {"event": {name: body}}


class SonicSession:
    """One conversation with Nova Sonic over the bidirectional stream."""

    def __init__(self, model=MODEL, region=REGION, voice=VOICE, system=SYSTEM_PROMPT, on_text=print):
        self.model, self.region, self.voice, self.system, self.on_text = model, region, voice, system, on_text
        self.prompt = str(uuid.uuid4())
        self.audio_name = str(uuid.uuid4())
        self.stream = None
        self.active = False
        self.audio_out = asyncio.Queue()       # speech bytes from Sonic, to play or save
        self.finished = asyncio.Event()        # set when Sonic finishes speaking a reply
        self.role, self.show = None, False
        self.reader = None

    async def send(self, body):
        from aws_sdk_bedrock_runtime.models import BidirectionalInputPayloadPart, InvokeModelWithBidirectionalStreamInputChunk
        await self.stream.input_stream.send(InvokeModelWithBidirectionalStreamInputChunk(
            value=BidirectionalInputPayloadPart(bytes_=json.dumps(body).encode("utf-8"))))

    async def start(self):
        from aws_sdk_bedrock_runtime.client import BedrockRuntimeClient, InvokeModelWithBidirectionalStreamOperationInput
        from aws_sdk_bedrock_runtime.config import Config
        from smithy_aws_core.identity.environment import EnvironmentCredentialsResolver
        client = BedrockRuntimeClient(config=Config(
            endpoint_uri=f"https://bedrock-runtime.{self.region}.amazonaws.com", region=self.region,
            aws_credentials_identity_resolver=EnvironmentCredentialsResolver()))
        self.stream = await client.invoke_model_with_bidirectional_stream(
            InvokeModelWithBidirectionalStreamOperationInput(model_id=self.model))
        self.active = True
        self.reader = asyncio.create_task(self._read())
        await self.send(event("sessionStart", inferenceConfiguration={"maxTokens": 1024, "topP": 0.9, "temperature": 0.7}))
        await self.send(event("promptStart", promptName=self.prompt,
                              textOutputConfiguration={"mediaType": "text/plain"},
                              audioOutputConfiguration={"mediaType": "audio/lpcm", "sampleRateHertz": OUT_RATE,
                                                        "sampleSizeBits": 16, "channelCount": 1, "voiceId": self.voice,
                                                        "encoding": "base64", "audioType": "SPEECH"}))
        await self.text(self.system, role="SYSTEM", interactive=False)

    async def text(self, words, role="USER", interactive=True):
        name = str(uuid.uuid4())
        await self.send(event("contentStart", promptName=self.prompt, contentName=name, type="TEXT",
                              interactive=interactive, role=role, textInputConfiguration={"mediaType": "text/plain"}))
        await self.send(event("textInput", promptName=self.prompt, contentName=name, content=words))
        await self.send(event("contentEnd", promptName=self.prompt, contentName=name))

    async def start_audio(self):
        await self.send(event("contentStart", promptName=self.prompt, contentName=self.audio_name, type="AUDIO",
                              interactive=True, role="USER",
                              audioInputConfiguration={"mediaType": "audio/lpcm", "sampleRateHertz": IN_RATE,
                                                       "sampleSizeBits": 16, "channelCount": 1,
                                                       "audioType": "SPEECH", "encoding": "base64"}))

    async def audio(self, pcm):
        await self.send(event("audioInput", promptName=self.prompt, contentName=self.audio_name,
                              content=base64.b64encode(pcm).decode("ascii")))

    async def end(self, audio_open=False):
        if not self.active:
            return
        try:
            if audio_open:
                await self.send(event("contentEnd", promptName=self.prompt, contentName=self.audio_name))
            await self.send(event("promptEnd", promptName=self.prompt))
            await self.send(event("sessionEnd"))
            await self.stream.input_stream.close()
        finally:
            self.active = False
            if self.reader:
                self.reader.cancel()

    async def _read(self):
        try:
            while self.active:
                output = await self.stream.await_output()
                result = await output[1].receive()
                if not (result and result.value and result.value.bytes_):
                    continue
                ev = json.loads(result.value.bytes_.decode("utf-8")).get("event", {})
                if "contentStart" in ev:
                    start = ev["contentStart"]
                    self.role = start.get("role")
                    fields = json.loads(start.get("additionalModelFields", "{}") or "{}")
                    # Sonic may send a reply's words twice (SPECULATIVE, then FINAL): show them once
                    self.show = fields.get("generationStage", "SPECULATIVE") == "SPECULATIVE"
                elif "textOutput" in ev:
                    words = ev["textOutput"]["content"]
                    if self.role == "USER":
                        self.on_text(f"You:      {words}")
                    elif self.role == "ASSISTANT" and self.show:
                        self.on_text(f"EyeReach: {words}")
                elif "audioOutput" in ev:
                    await self.audio_out.put(base64.b64decode(ev["audioOutput"]["content"]))
                elif "contentEnd" in ev:
                    end = ev["contentEnd"]
                    if end.get("type") == "AUDIO" and end.get("stopReason") == "END_TURN":
                        self.finished.set()                  # the reply's speech is complete
                elif "completionEnd" in ev:
                    self.finished.set()
        except asyncio.CancelledError:
            pass
        except Exception as error:                        # the stream closed or AWS refused: say why
            self.on_text(f"(Sonic stream ended: {type(error).__name__}: {error})")
            self.active = False
            self.finished.set()


async def say_text(words, out_path, timeout=30.0):
    """Send one typed sentence and save Sonic's spoken reply as a WAV file. Returns seconds of speech."""
    session = SonicSession()
    await session.start()
    await session.text(words)
    await session.start_audio()                # Sonic only answers while an audio input is open

    async def silence():
        while session.active:
            await session.audio(bytes(2 * CHUNK))
            await asyncio.sleep(CHUNK / IN_RATE)
    quiet = asyncio.create_task(silence())
    chunks, end = [], time.monotonic() + timeout
    while time.monotonic() < end and not (session.finished.is_set() and session.audio_out.empty()):
        try:
            chunks.append(await asyncio.wait_for(session.audio_out.get(), 0.5))
        except asyncio.TimeoutError:
            if chunks and session.audio_out.empty() and session.finished.is_set():
                break
    quiet.cancel()
    await session.end(audio_open=True)
    pcm = b"".join(chunks)
    with wave.open(out_path, "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(OUT_RATE)
        f.writeframes(pcm)
    return len(pcm) / (2 * OUT_RATE)


async def talk(microphone=None, speaker=None):
    """Microphone to Sonic, Sonic to the speaker, until Ctrl+C."""
    import sounddevice as sd
    loop = asyncio.get_running_loop()
    mic = asyncio.Queue(maxsize=200)
    speaking_until = [0.0]                                 # the speaker is busy until then: send silence

    def heard(data, frames, when, status):
        loop.call_soon_threadsafe(lambda: mic.full() or mic.put_nowait(bytes(data)))

    session = SonicSession()
    await session.start()
    await session.start_audio()
    print("Listening. Talk to EyeReach; Ctrl+C ends.")

    async def send_microphone():
        silence = bytes(2 * CHUNK)
        while session.active:
            pcm = await mic.get()
            await session.audio(silence if time.monotonic() < speaking_until[0] else pcm)

    async def play():
        with sd.RawOutputStream(samplerate=OUT_RATE, channels=1, dtype="int16", device=speaker) as out:
            while session.active:
                pcm = await session.audio_out.get()
                speaking_until[0] = max(speaking_until[0], time.monotonic()) + len(pcm) / (2 * OUT_RATE) + 0.3
                await loop.run_in_executor(None, out.write, pcm)

    try:
        listening = sd.RawInputStream(samplerate=IN_RATE, blocksize=CHUNK, channels=1, dtype="int16",
                                      device=microphone, callback=heard)
    except Exception as error:                     # busy (another program has it) or unplugged: the webcam
        print(f"Microphone {microphone} didn't open ({error}); using the webcam (system default) instead.")
        listening = sd.RawInputStream(samplerate=IN_RATE, blocksize=CHUNK, channels=1, dtype="int16",
                                      device=None, callback=heard)
    with listening:
        tasks = [asyncio.create_task(send_microphone()), asyncio.create_task(play())]
        try:
            while session.active:
                await asyncio.sleep(0.2)
        finally:
            for task in tasks:
                task.cancel()
            await session.end(audio_open=True)


def device(value):
    return int(value) if value is not None and str(value).isdigit() else value


def main():
    parser = argparse.ArgumentParser(description="Talk to EyeReach with Amazon Nova 2 Sonic.")
    parser.add_argument("--text", help="send this sentence instead of the microphone; save the reply as a WAV file")
    parser.add_argument("--out", default="/tmp/eyereach_sonic_reply.wav", help="where --text saves the reply")
    parser.add_argument("--microphone", help="input device number or name (default: the speaker's microphone, "
                                             "else the webcam's; see eyereach/audio_devices.py)")
    parser.add_argument("--speaker", help="output device number or name (default: the system's)")
    parser.add_argument("--list", action="store_true", help="list audio devices and exit")
    args = parser.parse_args()
    if args.list:
        import sounddevice as sd
        print(sd.query_devices())
        return 0
    if not (os.environ.get("AWS_ACCESS_KEY_ID") and os.environ.get("AWS_SECRET_ACCESS_KEY")):
        print("No AWS keys in the environment: load .env first (set -a; . ./.env; set +a).")
        return 1
    if args.text:
        seconds = asyncio.run(say_text(args.text, args.out))
        print(f"Reply: {seconds:.1f} s of speech saved to {args.out}" if seconds else "No spoken reply came back.")
        return 0 if seconds else 1
    microphone = device(args.microphone)
    if microphone is None:                             # the speaker's microphone first, the webcam's as fallback
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from eyereach.audio_devices import pick_microphone
        microphone = pick_microphone()
    try:
        asyncio.run(talk(microphone, device(args.speaker)))
    except KeyboardInterrupt:
        print("\nBye.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
