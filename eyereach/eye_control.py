"""EyeReach: drive the arm with gaze instead of a hand.

The arm driver (brain/arm_link.py) already follows a
table point (u, v) and can grab and release. Here that point comes from where the user looks.
Hold your gaze on the claw's spot to grab; hold it on the drop zone to let go.
"""

import math
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "brain"))
import arm_link  # noqa: E402

DWELL = 0.8                    # seconds of steady gaze that count as "yes"
STEADY = 0.04                  # the gaze may drift this far (play-area units) and still be steady
NEAR = 0.06                    # the claw must be this close to the gaze before a grab or a drop
DROP = (0.0, 0.8, 0.2, 1.0)    # (u0, v0, u1, v1) drop zone; shortcut: fixed corner, set it to the tray


def in_drop(gaze, zone=DROP):
    return zone[0] <= gaze[0] <= zone[2] and zone[1] <= gaze[1] <= zone[3]


class EyeControl:
    """Same job as the hand-tracking loop, with (u, v) from the gaze and dwell instead of a fist."""

    def __init__(self, robot, dwell=DWELL):
        self.robot, self.dwell = robot, dwell
        self.phase = "open"        # open -> grab -> holding -> (open again)
        self.rest = None           # (u, v, since): where the gaze has been resting

    def steady(self, gaze, now):
        """True once the gaze has rested on one spot for the dwell time."""
        if gaze is None:
            self.rest = None
            return False
        if self.rest is None or math.dist(gaze, self.rest[:2]) > STEADY:
            self.rest = (*gaze, now)
            return False
        return now - self.rest[2] >= self.dwell

    def near(self, gaze):
        u, v, _ = self.robot.gripper_area()
        return math.dist(gaze, (u, v)) <= NEAR

    def let_go(self):
        self.robot.open_here()
        self.robot.carrying(False)
        self.phase, self.rest = "open", None

    def step(self, gaze, now=None):
        """One frame. gaze is (u, v) in the play area, or None when tracking is lost
        (then the arm just holds where it is)."""
        now = time.monotonic() if now is None else now
        r = self.robot
        r.update()
        if gaze is not None:
            r.follow(*gaze)                                # ignored while grabbing
        yes = self.steady(gaze, now) and self.near(gaze)   # never act where the claw isn't
        if self.phase == "open":
            if yes and not in_drop(gaze) and r.grab():
                self.phase, self.rest = "grab", None
        elif self.phase == "grab":
            if not r.busy():
                if r.holding():
                    self.phase = "holding"
                    r.carrying(True)                       # slower while holding something
                else:
                    self.let_go()                          # missed: don't pretend it's held
        elif self.phase == "holding" and yes and in_drop(gaze):
            self.let_go()
        return self.phase


# ---------------------------------------------------------------------------------------------
# Guided control: for someone who can only move their eyes. Nothing moves until they choose it.
#   look at a spot on the map and keep looking  -> the arm slowly goes there
#   eyes and mouth open                        -> steer slowly
#   close the mouth after steering              -> pick up if empty, release if holding
#   look at a button and keep looking           -> PICK UP / LET GO, STOP, FIX MY AIM
#   look away, close the eyes, or lose the face -> nothing new happens, the arm finishes and waits
# ---------------------------------------------------------------------------------------------

VOLUME_STEP = 10               # % per VOLUME - / VOLUME + press
LOOK = 1.5                     # seconds of steady looking that press a button or choose a spot
SPOT = 0.06                    # the gaze may wander this far (map units) and still be on one spot
ARRIVED = 0.04                 # the claw is "there" this close to the chosen spot (map units)
HERE = 0.08                    # looking this close to where the claw already is doesn't count as a new move
EYE_SPEEDS = (60.0, 0.6, 40.0, 0.4)   # glide mm/s, joint rad/s; then the same while carrying
HOVER_MM = 120.0               # fingertips 12 cm above the table at startup and while moving or carrying
GRAB_MM = 50.0                 # fingertips this high above the table when picking up (2 and 4 cm were too close)
LOWEST_GRAB_MM = 30.0          # --grab-mm can't go below this
VOICE_IDLE = 8.0               # auto mode: seconds without a spoken command before the face takes over again


class Dwell:
    """How long the gaze has rested on one thing: a button (by name) or a map spot."""

    def __init__(self, seconds=LOOK):
        self.seconds, self.thing, self.since = seconds, None, None
        self.fired = False

    def update(self, thing, now):
        """thing: None, ("button", name) or ("map", (u, v)). Returns (progress 0..1, fired thing)."""
        same = (thing is not None and self.thing is not None and thing[0] == self.thing[0] and
                (thing[1] == self.thing[1] if thing[0] == "button" else
                 math.dist(thing[1], self.thing[1]) <= SPOT))
        if not same:
            self.fired = False
            self.thing, self.since = thing, (now if thing is not None else None)
            return 0.0, None
        progress = (now - self.since) / self.seconds
        if progress >= 1.0 and not self.fired:
            self.fired = True
            return 1.0, self.thing
        return min(1.0, progress), None


class FaceDrive:
    """Eyes open + mouth open steers; a deliberate mouth closure requests one pickup."""

    def __init__(self, mouth_open=0.35, mouth_closed=0.20, eye_open=0.12,
                 open_hold=0.35, close_hold=0.15):
        self.opened, self.closed, self.eye_open = mouth_open, mouth_closed, eye_open
        self.open_hold, self.close_hold = open_hold, close_hold
        self.reset()

    def reset(self):
        self.open_since = self.close_since = None
        self.armed = False

    def update(self, eyes, mouth, now, enabled=True):
        if (not enabled or eyes is None or len(eyes) != 2 or mouth is None or
                not math.isfinite(mouth) or mouth < 0 or
                any(not math.isfinite(x) or x < 0 for x in eyes)):
            self.reset()
            return False, False
        eyes_open = all(x >= self.eye_open for x in eyes)
        if mouth >= self.opened:
            self.close_since = None
            if not eyes_open:
                self.open_since = None
                return False, False
            if self.open_since is None:
                self.open_since = now
            if now - self.open_since >= self.open_hold:
                self.armed = True
            return self.armed, False
        self.open_since = None
        if mouth <= self.closed and self.armed and eyes_open:
            if self.close_since is None:
                self.close_since = now
            if now - self.close_since >= self.close_hold:
                self.reset()
                return False, True
        else:
            self.close_since = None
        return False, False


class GuidedControl:
    """Continuous or dwell-based control of the arm with dwell buttons and mouth-controlled steering. step() takes what the person
    is looking at; self.say holds sentences for the voice; self.events holds (name, details) for the
    session log; self.wants_calibration asks the app to re-calibrate."""

    def __init__(self, robot, look=LOOK, speeds=EYE_SPEEDS, grab_mm=GRAB_MM, movement="dwell", vision_speeds=None):
        self.robot, self.dwell = robot, Dwell(look)
        if movement not in ("follow", "dwell"):
            raise ValueError("movement must be follow or dwell")
        self.movement, self.live_enabled = movement, True
        self.vision_speeds = vision_speeds  # optional limits for steering, independent of voice/task movements
        self.grab_mm = max(grab_mm, LOWEST_GRAB_MM)        # never closer to the table than 3 cm
        robot.set_speeds(*speeds)
        self.target = None             # the chosen map spot (u, v), or None: the arm waits
        self.then = None               # what to do on arriving: None or "let go" (the drop-off move)
        self.phase = "waiting"         # waiting, moving, there, picking, letting go
        self.holding = False
        self.control_mode = "vision"
        self.voice_available = False
        self.accept_after = 0.0
        self.last_voice = None
        self.gaze_latches = set()
        self.hand_gaze_latched = False
        self.progress, self.looking = 0.0, None   # for drawing the filling ring
        self.closest = (math.inf, 0.0)    # (closest the claw got to the target, when): spots out of reach
        self.say, self.events, self.wants_calibration = [], [], False
        self.wants_volume = 0          # % for the app to change the speaker volume by (VOLUME - / +)
        self.volume = None             # the speaker volume in %, for the screen (the app keeps it current)
        self.task_started = None       # when the person first moved the arm for this pick-and-place
        self.moved_at = None
        # auto mode (the app turns it on): a spoken arm command switches to voice, which stops the face from
        # moving the arm; after VOICE_IDLE quiet seconds the face controls the arm again
        self.auto_mode = False
        self.voice_chosen = False      # the person chose voice ("use my voice", USE VOICE): no automatic return to the
                                       # face, and the app stops tracking the face, until they say "use my face"
        self.reanchor = False          # back to the face: the app re-takes "straight" where the arm is (no jump)
        self.voice_since = 0.0         # when voice mode began, or the last spoken command in it
        # the task agent ("bring me the banana", run by the app): it asks, the arm fetches
        self.wants_task = None         # a request for the app to hand to the agent
        self.task_running = False      # the agent is working on one (face control waits)
        self.question = None           # the object the agent asks the person to confirm
        self.answers = []              # the person's yes/no, for the app to hand back
        self.job = None                # the arm's fetch (brain/fetch.py FetchJob) while phase is "fetching"
        self.fetched = []              # how each fetch ended (None: brought), for the app to hand back

    def buttons(self):
        """The buttons on screen now, top to bottom: (name, label)."""
        hand = ("let go", "LET GO") if self.holding else ("pick up", "PICK UP")
        toggle = ("resume", "START") if not self.live_enabled else ("stop", "STOP")
        mode = ("vision", "USE VISION") if self.control_mode == "voice" else ("voice", "USE VOICE")
        return [hand, toggle, mode, ("aim", "FIX MY AIM")]

    def can(self, name):
        """Can this button do something right now? (Greyed out on screen when not.)"""
        busy = self.phase in ("picking", "letting go", "jogging", "recovering", "fetching")
        if name == "pick up":
            return not busy and not self.holding and self.phase != "moving"
        if name == "let go":
            return not busy and self.holding and self.phase != "moving"
        if name in ("home", "dropoff"):
            return False
        if name == "stop":
            return True
        if name == "voice":
            return not busy and self.voice_available
        if name == "vision":
            return not busy
        if name in ("volume down", "volume up"):
            return True                                    # harmless, so always usable
        return not busy and name in ("aim", "resume")

    def instruction(self):
        """One short sentence: what to do now."""
        if self.question:
            return f"Is it the {self.question}? Say yes or no."
        if self.phase in ("picking", "letting go", "jogging", "recovering", "fetching"):
            return getattr(self.robot, "stage", "Working") + ". Look at STOP or say Stop."
        if self.task_running:
            return "Working on it. Look at STOP or say Stop."
        if self.phase == "fault":
            return "Arm stopped: " + str(self.robot.error)
        if not self.live_enabled:
            return "Stopped. Say vision or voice to go on."
        if self.control_mode == "voice":
            return "Say left, right, closer, away, grab, drop, or vision."
        if self.phase == "moving":
            return "Moving. Look at STOP or say Stop."
        if getattr(self, "tilt", False):
            return ("Eyes + mouth open to steer. Close mouth to release." if self.holding else
                    "Eyes + mouth open to steer. Close mouth to pick up.")
        if self.movement == "follow":
            return "Eyes + mouth open to move. Close mouth to release." if self.holding else "Eyes + mouth open to move. Close mouth to pick up."
        if self.holding:
            return ("Close your mouth to release here." if self.phase == "there" else
                    "Move to where you want to put it, then close your mouth.")
        return "Open your eyes and mouth to steer; close your mouth to pick up."

    def go(self, target, why, now, then=None):
        self.target, self.phase, self.then, self.closest = target, "moving", then, (math.inf, now)
        self.moved_at = now
        if self.task_started is None:
            self.task_started = now
        self.events.append(("move", {"to": [round(target[0], 3), round(target[1], 3)], "by": why}))

    def press(self, name, now, how="button"):
        r = self.robot
        if not self.can(name):
            return
        if name in ("voice", "vision"):
            # chosen on purpose (a spoken mode command or the button) sticks; speaking an arm command in auto
            # mode ("speech") or the quiet-time return ("auto") doesn't
            self.voice_chosen = name == "voice" and how not in ("speech", "auto")
            self.reanchor = name == "vision"
            self.gaze_latches.add("mode")
            r.halt()
            self.control_mode = name
            self.target, self.then, self.phase = None, None, "waiting"
            # voice mode is ready at once (the arm only moves on a spoken command); back to vision waits
            # for MOVE AGAIN, so the nose can't make the arm jump to where it points
            self.live_enabled = name == "voice" or getattr(self, "tilt", False)   # nose: re-anchored, no jump
            self.voice_since = now
            self.holding = bool(r.closed)
            r.carrying(self.holding)
            self.accept_after = now
            self.dwell = Dwell(self.dwell.seconds)
            if name == "voice":
                self.say.append("Voice mode. Face tracking is off. Say move left, move right, closer or away. "
                                "Say use my face to go back." if self.voice_chosen else
                                "Voice mode. Say move left, move right, move closer or move away.")
            else:
                self.say.append("Face control. The arm follows your nose." if getattr(self, "tilt", False) else
                                "Vision mode. Say start, or press START, to follow your nose.")
            self.events.append(("control mode", {"mode": name, "by": how}))
        elif name == "pick up" and r.grab(self.grab_mm):
            self.target = None
            self.phase = "picking"
            self.hand_gaze_latched = True
            self.say.append("Picking up.")
            self.events.append(("pick", {"by": how}))
        elif name == "let go" and r.release_here():
            self.phase, self.then = "letting go", None
            self.hand_gaze_latched = True
            self.say.append("Letting go.")
            self.events.append(("let go", {"by": how}))
        elif name == "resume":
            self.gaze_latches.add("motion")
            r.error = None
            self.phase = "waiting"
            area = r.gripper_area()
            if isinstance(r.hover, (int, float)) and area[2] < r.hover - 5:
                if not r.recover():
                    self.say.append("Cannot lift here. Ask the helper to check the arm.")
                    return
                self.phase = "recovering"
            self.live_enabled = True
            self.say.append("Ready to move.")
        elif name == "stop":
            self.gaze_latches.add("motion")
            r.halt()
            if self.job is not None:                       # the agent's fetch ends here
                self.fetched.append("the person said stop")
                self.job = None
            if self.question:
                self.answers.append(False)                 # STOP is a no
                self.question = None
            self.holding = bool(r.closed)
            r.carrying(self.holding)
            self.accept_after = now
            self.live_enabled = False
            self.dwell = Dwell(self.dwell.seconds)
            self.target, self.then, self.phase = None, None, "waiting"
            self.say.append("Stopped.")
            self.events.append(("stop", {}))
        elif name in ("volume down", "volume up"):
            self.wants_volume = VOLUME_STEP if name == "volume up" else -VOLUME_STEP
            self.dwell = Dwell(self.dwell.seconds)         # keep looking at it: it steps again
            self.events.append(("volume", {"change": self.wants_volume, "by": how}))
        elif name == "aim":
            r.halt()
            self.accept_after = now
            self.target, self.then, self.phase = None, None, "waiting"
            self.wants_calibration = True
            self.events.append(("fix aim", {}))

    def voice(self, action, now, started=None, heard=None):
        """Execute on the UI thread. Never queue a command through a busy operation."""
        if action == "stop":
            self.press("stop", now, how="voice")
            return True
        started = now if started is None else started
        heard = now if heard is None else heard
        if now - heard > 1.0 or started < self.accept_after:
            return False
        if self.last_voice is not None and action == self.last_voice[0] and now - self.last_voice[1] < 0.8:
            return False
        if action in ("yes", "no"):
            if not self.question:
                return False
            self.answers.append(action == "yes")
            self.question = None
            self.last_voice = (action, now)
            return True
        if self.phase in ("picking", "letting go", "jogging", "recovering", "fetching"):
            return False
        task = isinstance(action, tuple) and len(action) == 2 and action[0] == "task"
        jog = isinstance(action, tuple) and len(action) == 3 and action[0] == "jog"
        if self.auto_mode and self.control_mode == "vision" and (task or jog or action in ("pick up", "let go")):
            if not self.can("voice"):
                return False
            self.press("voice", now, how="speech")         # speaking stops the face from moving the arm
        if self.control_mode == "voice":
            self.voice_since = now
        if task:
            if self.task_running or self.wants_task:
                self.say.append("One thing at a time.")
                return False
            self.wants_task = action[1]
            self.say.append(f"Looking for {action[1]}.")
            self.events.append(("task", {"request": action[1]}))
        elif jog:
            if self.control_mode != "voice" or not self.live_enabled:
                self.say.append("Say voice first.")
                return False
            if not self.robot.jog(action[1], action[2]):
                self.say.append("That movement is not available here.")
                return False
            self.phase, self.target = "jogging", None
            self.events.append(("move", {"by": "voice", "direction": action[1], "mm": action[2]}))
        else:
            if action not in ("voice", "vision", "pick up", "let go", "resume", "aim") or not self.can(action):
                return False
            self.press(action, now, how="voice")
        self.last_voice = (action, now)
        return True

    def task_done(self, reply, now):
        """The agent finished: say its answer; the quiet time before face control starts again from now."""
        self.task_running, self.question = False, None
        self.voice_since = now
        self.say.append(reply)
        self.events.append(("task done", {"reply": reply}))

    def ask(self, name):
        """The agent asks: is it this object?"""
        self.question = name
        self.say.append("I found it.")                     # the app adds the question (instruction())

    def start_fetch(self, job):
        """The agent picks: the arm runs this FetchJob, one step a frame. -> None, or why it can't."""
        if job.why:
            return job.why
        if self.phase in ("picking", "letting go", "jogging", "recovering", "fetching", "fault"):
            return "the arm is busy"
        self.job, self.phase, self.target = job, "fetching", None
        self.events.append(("fetch", {}))
        return None

    def follow_vision(self, target):
        """Limit this steering step without changing the voice/task/gripper speed profiles."""
        r = self.robot
        if self.vision_speeds is None:
            r.follow(*target)
            return
        glide, speed = r.glide, r.speed
        limits = self.vision_speeds[2:] if self.holding else self.vision_speeds[:2]
        try:
            r.glide, r.speed = min(glide, limits[0]), min(speed, limits[1])
            r.follow(*target)
        finally:
            r.glide, r.speed = glide, speed

    def close_mouth(self, now):
        """One action after steering: pick up when empty, release when holding."""
        if self.live_enabled and not (self.task_running or self.question):
            self.press("let go" if self.holding else "pick up", now, how="mouth close")

    def hold_steering(self):
        """Hold the measured position without declaring an explicit STOP or interrupting pickup."""
        self.robot.halt()
        self.robot.stage = "Holding" if self.holding else "Ready"
        self.target, self.then, self.phase = None, None, "waiting"

    def step(self, looking, now=None, grab_action=False, suspended=False, vision_movement=True):
        """One frame. looking: None, ("button", name) or ("map", (u, v)). grab_action: a deliberate mouth closure just fired.
        vision_movement: eyes and mouth are open; gates steering, not an active pickup."""
        now = time.monotonic() if now is None else now
        r = self.robot
        if self.control_mode == "vision" and not vision_movement:
            # Cancel the active trajectory before updating the robot, including dwell destinations.
            if self.phase in ("moving", "following"):
                self.hold_steering()
            if looking is not None and looking[0] == "map":
                looking = None
        r.update()
        error = getattr(r, "error", None)
        if isinstance(error, str) and error and self.phase != "fault":
            self.press("stop", now)
            self.phase = "fault"
            self.say.append(error)
            self.events.append(("arm fault", {"message": error}))
        if (self.auto_mode and self.control_mode == "voice" and self.live_enabled and self.phase in ("waiting", "there")
                and not (self.task_running or self.wants_task or self.question) and not self.voice_chosen
                and now - self.voice_since > VOICE_IDLE):
            self.press("vision", now, how="auto")
            self.live_enabled = True                       # what the person asked for: the face takes over again
            self.say[-1:] = ["Face control again."]
        if self.control_mode == "voice":
            grab_action = False
        if suspended:
            looking, grab_action = None, False
            if self.phase in ("moving", "following", "jogging"):
                if self.control_mode == "vision" and self.phase in ("moving", "following"):
                    self.hold_steering()
                else:
                    r.halt()
                    self.target, self.then, self.phase = None, None, "waiting"
        if grab_action:
            self.close_mouth(now)
        if looking is not None:
            for group, names in (("mode", ("voice", "vision")), ("motion", ("stop", "resume"))):
                if looking[0] == "button" and looking[1] in names:
                    if group in self.gaze_latches:
                        looking = None
                        break
                else:
                    self.gaze_latches.discard(group)
        hand_look = looking is not None and looking[0] == "button" and looking[1] in ("pick up", "let go")
        if looking is not None and not hand_look:
            self.hand_gaze_latched = False
        if hand_look and self.hand_gaze_latched:
            looking = None  # changing PICK UP to LET GO must not press it again
        if self.control_mode == "voice" and looking is not None and looking[0] == "map":
            looking = None
        if looking is not None and looking[0] == "button" and not self.can(looking[1]):
            looking = None                                 # a greyed-out button does nothing
        self.looking = looking
        self.progress, fired = self.dwell.update(
            None if self.movement == "follow" and looking is not None and looking[0] == "map" else looking, now)
        if fired is not None and fired[0] == "button":
            self.press(fired[1], now)
        elif fired is not None and self.live_enabled and self.control_mode == "vision" and self.phase not in ("picking", "letting go", "jogging", "recovering", "fetching", "fault"):
            claw = r.gripper_area()[:2]
            if self.phase == "moving" or math.dist(fired[1], claw) > HERE:   # resting the eyes on the arm isn't a move
                self.go(fired[1], "look", now)
                self.say.append("Moving there.")
        if self.control_mode == "vision" and self.movement == "follow" and self.phase not in ("moving", "picking", "letting go", "jogging", "recovering", "fetching", "fault"):
            if not suspended and self.live_enabled and looking is not None and looking[0] == "map":
                if self.phase != "following":
                    self.events.append(("move", {"to": list(looking[1]), "by": "follow"}))
                self.target, self.phase = looking[1], "following"
                if self.task_started is None:
                    self.task_started = now
                self.follow_vision(self.target)
            elif self.phase == "following":
                self.hold_steering()
        if self.phase == "moving":
            self.follow_vision(self.target)
            u, v, _ = r.gripper_area()
            gap = math.dist((u, v), self.target)
            if gap < self.closest[0] - 0.005:
                self.closest = (gap, now)
            arrived = not r.busy() and gap <= ARRIVED
            if arrived or now - self.closest[1] > 2.5:     # no closer for a while: as near as it can get
                self.phase = "there"
                self.events.append(("arrived", {"seconds": round(now - self.moved_at, 2), "reached": arrived}))
                if self.then == "let go":
                    self.press("let go", now, how="dropoff")
                else:
                    self.say.append("Arrived." if arrived else "That is as far as the arm can reach.")
        elif self.phase == "fetching":
            if self.job is None:                           # STOP ended it
                self.phase = "waiting"
            elif self.job.step():
                self.fetched.append(self.job.why)
                self.events.append(("fetched", {"problem": self.job.why}))
                self.job, self.phase = None, "there"
                self.holding = bool(r.closed)
                r.carrying(self.holding)
                self.accept_after = now
        elif self.phase in ("jogging", "recovering") and not r.busy():
            self.phase = "there"
            self.accept_after = now
            self.say.append("Ready.")
        elif self.phase == "picking" and not r.busy():
            self.accept_after = now
            # Keep the gripper closed until the next mouth-close action, LET GO or voice drop. The position-based holding guess
            # is only logged: on the real arm it said "nothing" even when it had the object.
            detected = r.holding()
            position = getattr(r, "gripper_position", lambda: None)()
            self.holding = r.closed
            r.carrying(self.holding)
            self.phase = "there" if self.holding else "waiting"
            self.events.append(("picked", {"holding": detected, "closed": r.closed,
                                           "gripper": round(position, 4) if isinstance(position, (int, float)) else None}))
            r.stage = "Holding" if self.holding else "Ready"
            self.say.append("Holding. Open your eyes and mouth to move; close your mouth to release.")
        elif self.phase == "letting go" and not r.busy():
            self.accept_after = now
            self.holding = False
            r.carrying(False)
            self.phase = "waiting"
            if self.task_started is not None:              # one pick-and-place done
                self.events.append(("task done", {"seconds": round(now - self.task_started, 2)}))
            self.task_started = None
            self.say.append("Done.")
        return self.phase
