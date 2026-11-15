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

MIN_CONFIDENCE = 0.70   # (exact phrases, kept for parse() callers)
HEARD_CONFIDENCE = 0.5  # free speech: the sentence's average word confidence must reach this
MODEL = Path.home() / '.eyereach' / 'models' / 'vosk-model-small-en-us-0.15'
REPO = Path(__file__).resolve().parent.parent
# Nova 2 Sonic's SDK needs Python 3.12: it runs as its own program in the agent's environment
AGENT_PYTHON = Path(os.environ.get('EYEREACH_AGENT_PYTHON', Path.home() / 'eyereach_agent_venv' / 'bin' / 'python'))
MOVE_MM = {'left': 50.0, 'right': 50.0, 'toward me': 20.0, 'away from me': 20.0}
SMALL_MOVE_MM = 5.0
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
    PHRASES[f'robot move {direction}'] = ('jog', direction, MOVE_MM[direction])
    PHRASES[f'robot a little {direction}'] = ('jog', direction, SMALL_MOVE_MM)
# The way people actually say it: away/forward = away from me; closer/back = toward me
for word, direction in (('away', 'away from me'), ('forward', 'away from me'), ('closer', 'toward me'),
                        ('back', 'toward me')):
    PHRASES[f'robot move {word}'] = ('jog', direction, MOVE_MM[direction])
    PHRASES[f'robot a little {word}'] = ('jog', direction, SMALL_MOVE_MM)
PHRASES['robot come closer'] = ('jog', 'toward me', MOVE_MM['toward me'])
# Short forms without 'robot' (the app only accepts these in voice mode, which is chosen on purpose); switching
# mode and fixing the aim always need 'robot', so ordinary talk can't change mode.
for _phrase, _action in list(PHRASES.items()):
    if _phrase.startswith('robot ') and _action not in ('voice', 'vision', 'aim', 'stop'):
        PHRASES[_phrase[len('robot '):]] = _action


for _phrase in ('use my voice', 'voice mode', 'switch to voice', 'use voice', 'voice control'):
    PHRASES[_phrase] = 'voice'
for _phrase in ('use my face', 'use my head', 'use my eyes', 'face mode', 'switch to face', 'use vision',
                'face control'):
    PHRASES[_phrase] = 'vision'


def needs_voice_mode(text):
    """A short form (no 'robot'), other than Stop and the mode phrases: only acted on in voice mode."""
    words = re.sub(r'[^a-z ]', '', text.lower()).split()
    return bool(words) and words[0] != 'robot' and words != ['stop'] and parse(text) not in ('voice', 'vision')


# ---- understanding ordinary sentences ------------------------------------------------------------------
# "could you move it a bit to the left please" -> ('jog', 'left', 5.0): the key word anywhere in the sentence.
# STOP wins over everything; then the mode words; then fetching a named thing; then grab / let go; then a direction.

STOP_WORDS = {'stop', 'halt', 'freeze'}
VOICE_WORDS = {'voice'}
VISION_WORDS = {'vision', 'visual'}
GRAB_WORDS = {'grab', 'pick', 'take', 'catch', 'grip', 'hold', 'close'}
DROP_WORDS = {'drop', 'release', 'open', 'place'}
FETCH_WORDS = {'bring', 'fetch', 'find', 'get'}
SMALL_WORDS = {'little', 'bit', 'tiny', 'slightly', 'slight', 'small', 'touch'}
DIRECTIONS = {'left': 'left', 'right': 'right',
              'closer': 'toward me', 'toward': 'toward me', 'towards': 'toward me', 'back': 'toward me',
              'backward': 'toward me', 'backwards': 'toward me', 'near': 'toward me', 'nearer': 'toward me',
              'away': 'away from me', 'forward': 'away from me', 'forwards': 'away from me',
              'further': 'away from me', 'farther': 'away from me', 'front': 'away from me'}
ARM_PARTS = {'gripper', 'hand', 'claw', 'fingers', 'grip', 'on', 'tight'}   # "close the gripper": not a thing to fetch
NOT_WORDS = {'dont', 'not', 'never', 'no'}
FILLER = {'the', 'a', 'an', 'me', 'my', 'please', 'it', 'that', 'this', 'up', 'some', 'to', 'for', 'can', 'you',
          'could', 'would', 'go', 'and', 'of', 'on', 'from', 'table'}


KEYWORDS = sorted(STOP_WORDS | VOICE_WORDS | VISION_WORDS | GRAB_WORDS | DROP_WORDS | FETCH_WORDS | SMALL_WORDS |
                  set(DIRECTIONS) | {'move', 'go', 'let', 'put', 'down', 'up', 'use', 'my', 'face', 'head', 'eyes',
                                     'mode', 'switch', 'to', 'a', 'the', 'me', 'it', 'please', 'yes', 'no', 'not',
                                     "don't", 'robot', 'that', 'this', 'bit', 'little', 'more', 'again',
                                     'start', 'continue', 'resume', 'on'})


SHORT_ACTIONS = ('stop', 'voice', 'vision', 'resume', 'yes', 'no')   # short words: the offline recogniser's job too


def understand(text):
    """The action meant by an ordinary sentence, or None. Only one action per sentence."""
    words = re.sub(r"[^a-z ]", "", str(text).lower().replace("'", "")).split()
    if not words:
        return None
    present = set(words)
    if present & STOP_WORDS:
        return 'stop'
    if present & VOICE_WORDS or {'use', 'mouth'} <= present:
        return 'voice'
    if present & VISION_WORDS or ('face' in present and present & {'use', 'mode', 'switch', 'my'}) or \
            ('head' in present and 'use' in present) or ('eyes' in present and 'use' in present):
        return 'vision'
    if words in (['yes'], ['yeah'], ['yep'], ['correct']) or 'thats it' in ' '.join(words):
        return 'yes'                                      # answers to "Is it the banana?"
    if words in (['no'], ['nope']) or 'not that' in ' '.join(words):
        return 'no'
    if present & NOT_WORDS:
        return None                                       # "don't move left": nothing
    if present & {'start', 'continue', 'resume'} or {'go', 'on'} <= present and len(words) <= 3:
        return 'resume'                                   # carry on after a stop
    for i, word in enumerate(words):                      # "bring me the banana", "please pick up the orange"
        if word in FETCH_WORDS or word in GRAB_WORDS:
            thing = [w for w in words[i + 1:] if w not in FILLER and w not in DIRECTIONS and w not in ARM_PARTS
                     and w not in SMALL_WORDS]
            if thing:
                return ('task', 'the ' + ' '.join(thing[:3]))
    if 'let' in present and 'go' in present or {'put', 'down'} <= present or present & DROP_WORDS:
        return 'let go'
    if present & GRAB_WORDS:
        return 'pick up'
    for word in words:                                    # the first direction said
        if word in DIRECTIONS:
            return ('jog', DIRECTIONS[word], SMALL_MOVE_MM if present & SMALL_WORDS else MOVE_MM[DIRECTIONS[word]])
    return None


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
            # a keyword vocabulary in any order (not fixed phrases): everything else is [unk], so it spots "left",
            # "grab", "stop" inside any sentence without inventing words; understand() then finds the command
            recognizer = KaldiRecognizer(Model(str(self.model)), 16000, json.dumps(KEYWORDS + ['[unk]']))
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
                    action = understand(text) if final else None
                    # STOP may interrupt before the speaker finishes. Never use partial
                    # recognition for movement, gripping, release, or mode changes.
                    if set(text.split()) & STOP_WORDS and not stopped:
                        self._emit(VoiceEvent('stop', text, started or stamp, stamp))
                        stopped = True
                    if final:
                        words = result.get('result', [])
                        lowest = min((w.get('conf', 0) for w in words), default=0.0)
                        average = sum(w.get('conf', 0) for w in words) / len(words) if words else 0.0
                        confident = bool(words) and average >= HEARD_CONFIDENCE
                        self.last_heard = text
                        if confident and action is not None and action != 'stop' and not stopped:
                            self._emit(VoiceEvent(action, text, started or stamp, stamp))
                        elif text and text != '[unk]' and not stopped:
                            # Heard but not acted on: say why, so the person isn't left guessing
                            why = 'no command word' if action is None else f'not sure ({average:.2f})'
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


def die_with_parent():
    """preexec_fn for helper programs (Sonic, the agent): Linux ends them when the app ends, however it ends,
    so no Sonic stream keeps the microphone and AWS credit after the app is gone."""
    try:
        import ctypes
        import signal
        ctypes.CDLL("libc.so.6", use_errno=True).prctl(1, signal.SIGTERM)   # 1 = PR_SET_PDEATHSIG
    except (OSError, AttributeError):
        pass                                                              # not Linux: close() still ends them


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
                                            preexec_fn=die_with_parent,
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
            if str(message.get('heard', '')).strip() and 'command' not in message and 'task' not in message:
                action = understand(message['heard'])        # what Sonic heard, word for word: find the command
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
        self.recent = []                                  # (action, when) acted on, so the other one's copy is dropped

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
        if isinstance(until, (int, float)) and time.monotonic() < until + 0.8:
            sonic = []                                    # likely the app's own (short) voice coming back
        offline = self.offline.poll()
        if self.sonic.status == 'Listening' and not self.sonic.error:
            # Sonic writes sentences down better; the offline one keeps the short words Sonic tends to drop
            # ("voice", "vision", "start", "stop", "yes", "no"); moves and grabs come from Sonic
            offline = [event for event in offline if getattr(event, 'action', None) in SHORT_ACTIONS]
        now = time.monotonic()
        self.recent = [(action, at) for action, at in self.recent if now - at < 3.0]
        fresh = []
        for event in offline + sonic:                   # both heard the same thing: act once
            action = getattr(event, 'action', event)
            if any(action == seen for seen, _ in self.recent):
                continue
            self.recent.append((action, now))
            fresh.append(event)
        return fresh

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
