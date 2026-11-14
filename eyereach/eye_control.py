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
#   blink twice quickly                         -> pick up here, or let go if holding something
#   look at a button and keep looking           -> PICK UP / LET GO, DROP-OFF, HOME, STOP, FIX MY AIM
#   look away, close the eyes, or lose the face -> nothing new happens, the arm finishes and waits
# ---------------------------------------------------------------------------------------------

LOOK = 1.5                     # seconds of steady looking that press a button or choose a spot
SPOT = 0.06                    # the gaze may wander this far (map units) and still be on one spot
ARRIVED = 0.04                 # the claw is "there" this close to the chosen spot (map units)
HERE = 0.08                    # looking this close to where the claw already is doesn't count as a new move
EYE_SPEEDS = (60.0, 0.6, 40.0, 0.4)   # glide mm/s, joint rad/s; then the same while carrying
GRAB_MM = 40.0                 # fingertips this high above the table when picking up (the driver's 15 is too low)
RELEASE_MM = 50.0              # ... and when letting go
HOME = (0.5, 0.5)              # map spot the arm starts at
DROPOFF = (0.1, 0.9)           # map spot of the drop-off tray (near the arm, on the person's left)


class Dwell:
    """How long the gaze has rested on one thing: a button (by name) or a map spot."""

    def __init__(self, seconds=LOOK):
        self.seconds, self.thing, self.since = seconds, None, None

    def update(self, thing, now):
        """thing: None, ("button", name) or ("map", (u, v)). Returns (progress 0..1, fired thing)."""
        same = (thing is not None and self.thing is not None and thing[0] == self.thing[0] and
                (thing[1] == self.thing[1] if thing[0] == "button" else
                 math.dist(thing[1], self.thing[1]) <= SPOT))
        if not same:
            self.thing, self.since = thing, (now if thing is not None else None)
            return 0.0, None
        progress = (now - self.since) / self.seconds
        if progress >= 1.0:
            fired, self.thing, self.since = self.thing, None, None   # look away and back to press again
            return 1.0, fired
        return progress, None


class DoubleBlink:
    """Two quick blinks in a row. Measured against how open this person's eyes usually are, so it works
    for narrow and wide eyes. A long close (resting the eyes) never counts, and resets the count."""

    def __init__(self, shut=0.45, longest=0.5, within=1.0):
        self.shut = shut               # closed below this part of the usual opening
        self.longest = longest         # seconds: a blink is shorter than this
        self.within = within           # seconds: both blinks end within this
        self.usual = None              # usual opening, a slow average while open
        self.closed_since = None
        self.blinks = []               # times recent quick blinks ended

    def update(self, opening, now):
        """opening: eye_opening() this frame, or None with no face. True on the frame a double blink ends."""
        if opening is None:
            self.closed_since, self.blinks = None, []
            return False
        if self.usual is not None and opening < self.shut * self.usual:
            if self.closed_since is None:
                self.closed_since = now
            return False
        self.usual = opening if self.usual is None else 0.97 * self.usual + 0.03 * opening
        if self.closed_since is None:
            return False
        closed, self.closed_since = now - self.closed_since, None
        if closed > self.longest:
            self.blinks = []
            return False
        self.blinks = [t for t in self.blinks if now - t <= self.within] + [now]
        if len(self.blinks) >= 2:
            self.blinks = []
            return True
        return False


class GuidedControl:
    """Eyes-only control of the arm with dwell buttons and a double blink. step() takes what the person
    is looking at; self.say holds sentences for the voice; self.events holds (name, details) for the
    session log; self.wants_calibration asks the app to re-calibrate."""

    def __init__(self, robot, look=LOOK, speeds=EYE_SPEEDS, home=HOME, dropoff=DROPOFF, grab_mm=GRAB_MM):
        self.robot, self.dwell = robot, Dwell(look)
        self.grab_mm = max(grab_mm, 20.0)                  # never closer to the table than 20 mm
        self.release_mm = max(RELEASE_MM, self.grab_mm + 10)
        robot.set_speeds(*speeds)
        self.home, self.dropoff = home, dropoff
        self.target = None             # the chosen map spot (u, v), or None: the arm waits
        self.then = None               # what to do on arriving: None or "let go" (the drop-off move)
        self.phase = "waiting"         # waiting, moving, there, picking, letting go
        self.holding = False
        self.progress, self.looking = 0.0, None   # for drawing the filling ring
        self.closest = (math.inf, 0.0)    # (closest the claw got to the target, when): spots out of reach
        self.say, self.events, self.wants_calibration = [], [], False
        self.task_started = None       # when the person first moved the arm for this pick-and-place
        self.moved_at = None

    def buttons(self):
        """The buttons on screen now, top to bottom: (name, label)."""
        hand = ("let go", "LET GO") if self.holding else ("pick up", "PICK UP")
        return [hand, ("dropoff", "DROP-OFF"), ("home", "HOME"), ("stop", "STOP"), ("aim", "FIX MY AIM")]

    def can(self, name):
        """Can this button do something right now? (Greyed out on screen when not.)"""
        busy = self.phase in ("picking", "letting go")
        if name == "pick up":
            return not busy and not self.holding and self.phase != "moving"
        if name == "let go":
            return not busy and self.holding and self.phase != "moving"
        if name == "dropoff":
            return not busy and self.holding and self.then is None
        if name == "stop":
            return self.phase == "moving"
        return not busy

    def instruction(self):
        """One short sentence: what to do now."""
        if self.phase == "moving":
            return "Moving to the drop-off. Look at STOP to stop." if self.then else "Moving. Look at STOP to stop."
        if self.phase in ("picking", "letting go"):
            return {"picking": "Picking up ...", "letting go": "Letting go ..."}[self.phase]
        if self.holding:
            return ("Blink twice to let go here, or look at DROP-OFF." if self.phase == "there" else
                    "Look where to put it and keep looking, or look at DROP-OFF.")
        return ("Blink twice to pick it up here, or choose another spot." if self.phase == "there" else
                "Look at a spot on the table and keep looking. The arm goes there.")

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
        if name == "pick up" and r.grab(self.grab_mm):
            self.phase = "picking"
            self.say.append("Picking up.")
            self.events.append(("pick", {"by": how}))
        elif name == "let go" and r.release(self.release_mm):
            self.phase, self.then = "letting go", None
            self.say.append("Letting go.")
            self.events.append(("let go", {"by": how}))
        elif name == "dropoff":
            self.go(self.dropoff, "dropoff", now, then="let go")
            self.say.append("Going to the drop-off.")
        elif name == "home":
            self.go(self.home, "home", now)
            self.say.append("Going home.")
        elif name == "stop":
            self.target, self.then, self.phase = None, None, "waiting"
            self.say.append("Stopped.")
            self.events.append(("stop", {}))
        elif name == "aim":
            self.target, self.then, self.phase = None, None, "waiting"
            self.wants_calibration = True
            self.events.append(("fix aim", {}))

    def blink(self, now):
        """A double blink: pick up here, or let go if holding something."""
        self.press("let go" if self.holding else "pick up", now, how="blink")

    def step(self, looking, now=None, double_blink=False):
        """One frame. looking: None, ("button", name) or ("map", (u, v)). double_blink: one just ended."""
        now = time.monotonic() if now is None else now
        r = self.robot
        r.update()
        if double_blink:
            self.blink(now)
        if looking is not None and looking[0] == "button" and not self.can(looking[1]):
            looking = None                                 # a greyed-out button does nothing
        self.looking = looking
        self.progress, fired = self.dwell.update(looking, now)
        if fired is not None and fired[0] == "button":
            self.press(fired[1], now)
        elif fired is not None and self.phase not in ("picking", "letting go"):
            claw = r.gripper_area()[:2]
            if self.phase == "moving" or math.dist(fired[1], claw) > HERE:   # resting the eyes on the arm isn't a move
                self.go(fired[1], "look", now)
                self.say.append("Moving there.")
        if self.phase == "moving":
            r.follow(*self.target)
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
        elif self.phase == "picking" and not r.busy():
            self.holding = r.holding()
            r.carrying(self.holding)
            self.phase = "there" if self.holding else "waiting"
            self.events.append(("picked", {"holding": self.holding}))
            self.say.append("Got it." if self.holding else "Nothing there. Try another spot.")
            if not self.holding:
                r.open_here()
        elif self.phase == "letting go" and not r.busy():
            self.holding = False
            r.carrying(False)
            self.phase = "waiting"
            if self.task_started is not None:              # one pick-and-place done
                self.events.append(("task done", {"seconds": round(now - self.task_started, 2)}))
            self.task_started = None
            self.say.append("Done.")
        return self.phase
