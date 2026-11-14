"""Optional offline speech input. Importing this module needs no audio packages.

Only complete, exact commands execute. All actions except STOP need the word
'robot'. The microphone worker never touches the robot or the OpenCV window.
"""
import argparse
from dataclasses import dataclass
import json
import os
import subprocess
import sys
from pathlib import Path
import queue
import re
import threading
import time

MIN_CONFIDENCE = 0.70   # every word of a command must reach this (was 0.80: too many real commands dropped)
MODEL = Path.home() / '.eyereach' / 'models' / 'vosk-model-small-en-us-0.15'
REPO = Path(__file__).resolve().parent.parent
# Nova 2 Sonic's SDK needs Python 3.12: it runs as its own program in the agent's environment
AGENT_PYTHON = Path(os.environ.get('EYEREACH_AGENT_PYTHON', Path.home() / 'eyereach_agent_venv' / 'bin' / 'python'))
PHRASES = {
    'stop': 'stop', 'robot stop': 'stop',
    'robot use voice': 'voice', 'robot use vision': 'vision',
    'robot grab': 'pick up', 'robot grab the item': 'pick up',
    'robot pick up': 'pick up', 'robot release': 'let go',
    'robot release the item': 'let go', 'robot let go': 'let go',
    'robot resume': 'resume', 'robot fix my aim': 'aim',
    'robot yes': 'yes', 'robot no': 'no',            # answers when the task agent asks "is it the banana?"
}
for direction in ('left', 'right', 'toward me', 'away from me'):
    PHRASES[f'robot move {direction}'] = ('jog', direction, 20.0)
    PHRASES[f'robot a little {direction}'] = ('jog', direction, 5.0)
# The way people actually say it: away/forward = away from me; closer/back = toward me
for word, direction in (('away', 'away from me'), ('forward', 'away from me'), ('closer', 'toward me'),
                        ('back', 'toward me')):
    PHRASES[f'robot move {word}'] = ('jog', direction, 20.0)
    PHRASES[f'robot a little {word}'] = ('jog', direction, 5.0)
PHRASES['robot come closer'] = ('jog', 'toward me', 20.0)
# Short forms without 'robot' (the app only accepts these in voice mode, which is chosen on purpose); switching
# mode and fixing the aim always need 'robot', so ordinary talk can't change mode.
for _phrase, _action in list(PHRASES.items()):
    if _phrase.startswith('robot ') and _action not in ('voice', 'vision', 'aim', 'stop'):
        PHRASES[_phrase[len('robot '):]] = _action


def needs_voice_mode(text):
    """A short form (no 'robot'), other than Stop: only acted on in voice mode."""
    words = re.sub(r'[^a-z ]', '', text.lower()).split()
    return bool(words) and words[0] != 'robot' and words != ['stop']


def parse(text):
    """Return a whitelisted action, never a substring or an inferred instruction."""
    return PHRASES.get(re.sub(r'[^a-z ]', '', text.lower()).strip())


@dataclass(frozen=True)
class VoiceEvent:
    action: object
    text: str
    started: float
    heard: float


class VoiceInput:
    def __init__(self, model=MODEL, device=None):
        self.model, self.device = Path(model).expanduser(), device
        self.status = 'Starting microphone'
        self.error = None
        self.events = queue.Queue(maxsize=16)
        self.audio = queue.Queue(maxsize=16)
        self.done = threading.Event()
        self.thread = None
        self.last_heard = ''
        self.dropped = queue.Queue(maxsize=16)    # (text, why, lowest word confidence) heard but not acted on
        self.mute_until = 0.0                     # the app is talking until then: hear silence, not itself

    def start(self):
        self.thread = threading.Thread(target=self._run, name='eyereach-speech', daemon=True)
        self.thread.start()
        return self

    def _audio(self, data, frames, timing, status):
        if status:
            self.error = f'Microphone stream error: {status}'
            self.done.set()
            return
        try:
            now = time.monotonic()
            # while the app speaks its own prompts, pass silence so it can't take its voice for a command
            self.audio.put_nowait((now, bytes(len(data)) if now < self.mute_until else bytes(data)))
        except queue.Full:
            self.error = 'Microphone fell behind; restart voice input'
            self.done.set()

    def _emit(self, event):
        try:
            self.events.put_nowait(event)
        except queue.Full:
            self.error = 'Speech queue overflow; restart voice input'
            self.done.set()

    def _run(self):
        try:
            import sounddevice as sd
            from vosk import Model, KaldiRecognizer
            if not self.model.is_dir():
                raise RuntimeError(f'Voice model missing: {self.model}; run launch/setup_voice.sh')
            recognizer = KaldiRecognizer(Model(str(self.model)), 16000,
                                        json.dumps([*PHRASES, '[unk]']))
            recognizer.SetWords(True)
            if self.done.is_set():
                return
            with sd.RawInputStream(samplerate=16000, blocksize=1600, device=self.device,
                                   channels=1, dtype='int16', callback=self._audio) as stream:
                self.status = 'Listening'
                started, stopped = None, False
                while not self.done.is_set():
                    if not stream.active:
                        raise RuntimeError('Microphone disconnected')
                    try:
                        stamp, data = self.audio.get(timeout=0.2)
                    except queue.Empty:
                        continue
                    if time.monotonic() - stamp > 0.5:
                        raise RuntimeError('Microphone audio is stale')
                    final = recognizer.AcceptWaveform(data)
                    result = json.loads(recognizer.Result() if final else recognizer.PartialResult())
                    text = result.get('text' if final else 'partial', '')
                    if text and started is None:
                        started = stamp
                    action = parse(text)
                    # STOP may interrupt before the speaker finishes. Never use partial
                    # recognition for movement, gripping, release, or mode changes.
                    if 'stop' in text.split() and not stopped:
                        self._emit(VoiceEvent('stop', text, started or stamp, stamp))
                        stopped = True
                    if final:
                        words = result.get('result', [])
                        lowest = min((w.get('conf', 0) for w in words), default=0.0)
                        confident = bool(words) and lowest >= MIN_CONFIDENCE
                        self.last_heard = text
                        if confident and action is not None and action != 'stop' and not stopped:
                            self._emit(VoiceEvent(action, text, started or stamp, stamp))
                        elif text and text != '[unk]' and not stopped:
                            # Heard but not acted on: say why, so the person isn't left guessing
                            why = 'not a command' if action is None else f'not sure ({lowest:.2f}, needs {MIN_CONFIDENCE:.2f})'
                            try:
                                self.dropped.put_nowait((text, why, round(lowest, 2)))
                            except queue.Full:
                                pass
                        started, stopped = None, False
        except Exception as exc:
            self.error = str(exc)
        finally:
            self.status = 'Unavailable'

    def poll_dropped(self):
        """Phrases heard since the last call that were not acted on: [(text, why, lowest confidence)]."""
        items = []
        while True:
            try:
                items.append(self.dropped.get_nowait())
            except queue.Empty:
                return items

    def poll(self):
        items = []
        while True:
            try:
                items.append(self.events.get_nowait())
            except queue.Empty:
                return items

    def close(self):
        self.done.set()
        if self.thread:
            self.thread.join(timeout=2)


class SonicInput:
    """Free speech through Nova 2 Sonic ("move it a bit to the left"), next to VoiceInput.

    agent/voice_sonic.py --commands prints one JSON line per command; each command goes through the
    same whitelist (parse) and the same checks in the app as an exact spoken phrase. It needs the
    network, so STOP never depends on it: the offline VoiceInput and the gaze STOP button stay on."""

    def __init__(self, python=AGENT_PYTHON):
        self.python = python
        self.status = 'Starting Sonic'
        self.error = None
        self.events = queue.Queue(maxsize=16)
        self.process = None
        self.last_heard = ''

    def start(self):
        try:
            self.process = subprocess.Popen([str(self.python), '-u', '-m', 'agent.voice_sonic', '--commands'],
                                            cwd=REPO, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                                            stderr=sys.stderr, text=True)
        except OSError as exc:
            self.error, self.status = f'Sonic did not start: {exc}', 'Unavailable'
            return self
        threading.Thread(target=self.read, args=(self.process.stdout,), name='eyereach-sonic', daemon=True).start()
        return self

    def read(self, lines):
        for line in lines:
            try:
                message = json.loads(line)
            except ValueError:
                continue                                  # not for us (a warning, say)
            if 'status' in message:
                self.status = message['status']
            if 'error' in message:
                self.error = message['error']
            action = parse('robot ' + str(message.get('command', '')))
            if str(message.get('task', '')).strip():      # "bring me the banana": for the task agent
                action = ('task', str(message['task']).strip()[:200])
                message.setdefault('command', action[1])
            if action is not None:
                now = time.monotonic()
                self.last_heard = message.get('heard') or message['command']
                try:
                    self.events.put_nowait(VoiceEvent(action, f"{self.last_heard} (Sonic)", now, now))
                except queue.Full:
                    pass                                  # old commands are dropped anyway
        self.status = 'Unavailable'
        self.error = self.error or 'Sonic stopped'

    def poll(self):
        items = []
        while True:
            try:
                items.append(self.events.get_nowait())
            except queue.Empty:
                return items

    def close(self):
        if self.process and self.process.poll() is None:
            self.process.terminate()


class BothInputs:
    """The offline listener (it owns STOP and the microphone status) plus Sonic's free speech."""

    def __init__(self, offline, sonic):
        self.offline, self.sonic = offline, sonic

    status = property(lambda self: self.offline.status)
    error = property(lambda self: self.offline.error)    # only the offline one stops the arm when it fails

    @property
    def mute_until(self):
        return self.offline.mute_until

    @mute_until.setter
    def mute_until(self, until):
        # the offline microphone hears silence; Sonic (its own program on the microphone) can't be muted from
        # here, so what it hears while the app talks, and just after, is dropped in poll()
        self.offline.mute_until = until

    def poll_dropped(self):
        return self.offline.poll_dropped()

    def poll(self):
        sonic = self.sonic.poll()
        until = getattr(self.offline, 'mute_until', 0.0)
        if isinstance(until, (int, float)) and time.monotonic() < until + 2.0:
            sonic = []                                    # likely the app's own voice coming back
        return self.offline.poll() + sonic

    def close(self):
        self.sonic.close()
        self.offline.close()


def main():
    parser = argparse.ArgumentParser(description='Microphone check only: never connects to the robot')
    parser.add_argument('--list-devices', action='store_true')
    parser.add_argument('--device', help='microphone name or index')
    parser.add_argument('--model', type=Path, default=MODEL)
    args = parser.parse_args()
    if args.list_devices:
        import sounddevice as sd
        print(sd.query_devices())
        return
    device = int(args.device) if args.device and args.device.isdigit() else args.device
    if device is None:
        from eyereach.audio_devices import pick_microphone
        device = pick_microphone()
    listener = VoiceInput(args.model, device).start()
    print('Say Robot move left, Robot grab the item, or Stop. Ctrl+C exits. No robot is connected.')
    try:
        while not listener.error:
            for event in listener.poll():
                print(f'{event.text} -> {event.action}', flush=True)
            for text, why, lowest in listener.poll_dropped():
                print(f'{text} -> ignored: {why}', flush=True)
            time.sleep(0.05)
        raise SystemExit(listener.error)
    except KeyboardInterrupt:
        pass
    finally:
        listener.close()


if __name__ == '__main__':
    main()
