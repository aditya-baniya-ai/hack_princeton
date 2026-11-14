"""EyeReach: drive the arm with gaze instead of a hand.

The arm driver (game/arm_link.py) already follows a
table point (u, v) and can grab and release. Here that point comes from where the user looks.
Hold your gaze on the claw's spot to grab; hold it on the drop zone to let go.
"""

import math
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "game"))   # arm_link adds ../brain itself
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
