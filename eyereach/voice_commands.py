"""Optional offline speech input. Importing this module needs no audio packages.

Only complete, exact commands execute. All actions except STOP need the word
'robot'. The microphone worker never touches the robot or the OpenCV window.
"""
import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import queue
import re
import threading
import time

MIN_CONFIDENCE = 0.70   # every word of a command must reach this (was 0.80: too many real commands dropped)
MODEL = Path.home() / '.eyereach' / 'models' / 'vosk-model-small-en-us-0.15'
PHRASES = {
    'stop': 'stop', 'robot stop': 'stop',
    'robot use voice': 'voice', 'robot use vision': 'vision',
    'robot grab': 'pick up', 'robot grab the item': 'pick up',
    'robot pick up': 'pick up', 'robot release': 'let go',
    'robot release the item': 'let go', 'robot let go': 'let go',
    'robot resume': 'resume', 'robot fix my aim': 'aim',
}
for direction in ('left', 'right', 'toward me', 'away from me'):
    PHRASES[f'robot move {direction}'] = ('jog', direction, 20.0)
    PHRASES[f'robot a little {direction}'] = ('jog', direction, 5.0)


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
            self.audio.put_nowait((time.monotonic(), bytes(data)))
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
