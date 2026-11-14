"""Head tilt control: turn or tilt the head to move the arm that way; nod to pick up or let go.

No screen calibration: only "which way is the face turned, compared with when you looked straight ahead".

How the turn is measured: the nose guides it. The point between the eyes is the reference; turning moves the nose
tip sideways from it and looking up or down moves it up or down (gaze_webcam.nose_offset, in eye-widths). That
2D measure stays steady wherever the face is in the picture and whatever the camera's angle (the 3D head angle
read 41 deg "straight" with the webcam off to the side, and drifted). nose_angles() turns it into degrees.

    python3 -m eyereach.head_tilt          check it with your own head (no arm): prints each move and nod

    look left / right, or tilt the head left / right (ear toward the shoulder)
                        -> the arm moves to your left / right (the stronger of the two counts)
    look up / down      -> the arm moves away from you / toward you
    back to the centre  -> the arm stops
    a quick nod         -> pick up here, or let go if holding something

Safety and false triggers:
- A small dead zone around the centre, so holding still doesn't drift the arm.
- A tilt has to be held for HOLD seconds before the arm moves, so a nod (quicker than that) never moves it.
- A nod is a quick dip and return (down past NOD_DEG and back near the centre within NOD_MAX seconds) with the
  head not turned sideways; then a NOD_GAP pause before another nod counts.
- No face (or no head direction) gives no movement.
Directions come from gaze_webcam.head_pose: yaw > 0 is the face turned toward the picture's right, which is the
person's own left; pitch > 0 is up; roll > 0 is the head tilted to the person's own right (checked by turning a
photo: +15 deg counter-clockwise gave roll +14.8). FLIP_VERTICAL swaps up/down if that suits someone better.
"""

import math
from dataclasses import dataclass

DEAD = 8.0           # degrees from the centre that count as "still"
FULL = 25.0          # degrees for full speed
HOLD = 0.6           # seconds a tilt must be held before the arm moves (a nod is quicker)
NOD_DEG = 10.0       # a nod dips at least this far down ...
NOD_BACK = 5.0       # ... and comes back within this of the centre ...
NOD_MAX = 0.9        # ... within this many seconds,
NOD_SIDE = 10.0      # with the head turned less than this sideways
NOD_GAP = 1.2        # seconds after a nod before another one counts
FLIP_VERTICAL = False
NOSE_DEPTH = 0.5     # how far the nose tip stands out from the eyes, in eye-widths: a turn of a degrees moves it
                     # about NOSE_DEPTH * sin(a) eye-widths across (so 20 deg is about 0.17)
GRACE = 0.25         # seconds a lost face keeps the last direction (MediaPipe drops a frame now and then)


@dataclass
class Tilt:
    direction: tuple = None      # (du, dv) on the table map, unit length, or None: don't move
    speed: float = 0.0           # 0..1 of the speed limit
    label: str = "centre"        # "left", "right", "away", "toward", "left and away", ... or "centre" / "no face"
    nod: bool = False            # a nod just finished
    held: float = 0.0            # 0..1: how far the current tilt is toward HOLD (for the screen)
    yaw: float = 0.0             # degrees from the centre (person's left is positive)
    pitch: float = 0.0           # degrees from the centre (up is positive)
    roll: float = 0.0            # degrees from the centre (tilted to the person's right is positive)


def nose_angles(nose, roll_rad=0.0):
    """(sideways, down) from gaze_webcam.nose_offset -> (yaw, pitch, roll) in degrees for HeadTilt.update:
    yaw > 0 is the person's left, pitch > 0 is up. Linear (offset / NOSE_DEPTH radians): only the change from
    the centre counts, so the nose's resting place below the eyes doesn't matter."""
    if nose is None:
        return None
    side, down = nose
    yaw = side / NOSE_DEPTH
    # turning narrows the eye width the camera sees, which makes the nose look further down: take that back out
    down *= math.cos(max(-1.2, min(1.2, yaw)))
    return math.degrees(yaw), -math.degrees(down / NOSE_DEPTH), math.degrees(roll_rad)


class HeadTilt:
    def __init__(self, dead=DEAD, full=FULL, hold=HOLD, flip_vertical=FLIP_VERTICAL):
        self.dead, self.full, self.hold, self.flip = dead, full, hold, flip_vertical
        self.centre = None           # (yaw, pitch) in degrees when looking straight ahead
        self.tilt_since = None       # when the current (outside the dead zone) tilt began
        self.tilt_side = None        # its rough direction, so changing direction restarts the hold
        self.dip = None              # (start time, deepest pitch) of a possible nod
        self.dip_done = False        # a dip turned out to be a tilt: no new nod until back near the centre
        self.last_nod = -math.inf
        self.last_seen = None        # ((yaw, pitch), time) of the last face, for GRACE

    def set_centre(self, yaw, pitch, roll=0.0):
        """Degrees; usually the median of a couple of seconds of looking straight ahead."""
        self.centre = (yaw, pitch, roll)
        self.tilt_since = self.tilt_side = self.dip = None
        self.dip_done = False

    def update(self, pose_deg, now):
        """pose_deg: (yaw, pitch) or (yaw, pitch, roll) in degrees, or None with no face. Returns a Tilt."""
        if pose_deg is None and self.last_seen is not None and now - self.last_seen[1] <= GRACE:
            pose_deg = self.last_seen[0]               # a short drop-out: carry on as before
        elif pose_deg is not None:
            self.last_seen = (pose_deg, now)
        if pose_deg is None or self.centre is None:
            self.tilt_since = self.tilt_side = self.dip = None
            self.dip_done = False
            return Tilt(label="no face" if pose_deg is None else "centre")
        yaw, pitch = pose_deg[0] - self.centre[0], pose_deg[1] - self.centre[1]
        roll = (pose_deg[2] - self.centre[2]) if len(pose_deg) > 2 else 0.0
        nod = self._nod(yaw, pitch, now)
        out = Tilt(nod=nod, yaw=yaw, pitch=pitch, roll=roll)
        # left/right: looking (yaw) or tilting the head (roll), whichever is stronger; as "to the left" degrees
        lefty = yaw if abs(yaw) >= abs(roll) else -roll
        side = (abs(lefty) > self.dead and (1 if lefty > 0 else -1), abs(pitch) > self.dead and (1 if pitch > 0 else -1))
        if side == (False, False) or nod or self.dip is not None:
            self.tilt_since = self.tilt_side = None   # centred, or in the middle of a nod: don't move
            return out
        if side != self.tilt_side:
            self.tilt_since, self.tilt_side = now, side
        out.held = min(1.0, (now - self.tilt_since) / self.hold)
        if out.held < 1.0:
            out.label = "hold"
            return out
        # yaw > 0 (person's left) -> u down (the map's u = 0 is the person's left); up -> away (v up)
        du = -max(0.0, abs(lefty) - self.dead) * (1 if lefty > 0 else -1)
        dv = max(0.0, abs(pitch) - self.dead) * (1 if pitch > 0 else -1) * (-1 if self.flip else 1)
        size = math.hypot(du, dv)
        out.direction = (du / size, dv / size)
        out.speed = min(1.0, size / (self.full - self.dead))
        parts = [p for p, on in (("left", du < 0), ("right", du > 0), ("away", dv > 0), ("toward you", dv < 0))
                 if on and abs(du if p in ("left", "right") else dv) > 0.35 * size]
        out.label = " and ".join(parts) or "centre"
        return out

    def _nod(self, yaw, pitch, now):
        """True on the frame a quick down-and-back nod finishes."""
        if self.dip is None:
            if self.dip_done:
                self.dip_done = pitch < -NOD_BACK        # still tilted down: it's a move, not a new nod
                return False
            if pitch < -NOD_DEG / 2 and abs(yaw) < NOD_SIDE and now - self.last_nod > NOD_GAP:  # (roll: see update)
                self.dip = (now, pitch)                # going down: maybe a nod
            return False
        start, deepest = self.dip
        deepest = min(deepest, pitch)
        self.dip = (start, deepest)
        if abs(yaw) >= NOD_SIDE or now - start > NOD_MAX:
            self.dip, self.dip_done = None, True       # turned sideways, or too slow: a tilt, not a nod
            return False
        if pitch > -NOD_BACK:                          # back near the centre
            self.dip = None
            if deepest <= -NOD_DEG:
                self.last_nod = now
                return True
        return False


def live(seconds=60.0, camera="/dev/video0"):
    """Check it with your own head, no arm: sets the centre, then prints what each movement does."""
    import time
    import numpy as np
    from eyereach.gaze_webcam import Camera, IrisGaze
    cam, gaze, head = Camera(camera), IrisGaze("head"), HeadTilt()
    print("Look straight at the screen and keep still...")
    poses, start = [], time.monotonic()
    while len(poses) < 30 and time.monotonic() - start < 15:
        gaze.features(cam.read())
        if gaze.pose is not None:
            if gaze.nose is not None and gaze.pose is not None:
                poses.append(nose_angles(gaze.nose, gaze.pose[2]))
    if len(poses) < 30:
        print("No face found: move the camera so the whole face is in the picture.")
        return
    head.set_centre(*np.median(np.array(poses), axis=0))
    print("Centre set. Turn left/right, tilt up/down, nod. Ctrl+C ends.\n")
    last, end = None, time.monotonic() + seconds
    try:
        while time.monotonic() < end:
            gaze.features(cam.read())
            now = time.monotonic()
            t = head.update(None if gaze.nose is None or gaze.pose is None else nose_angles(gaze.nose, gaze.pose[2]), now)
            line = f"yaw {t.yaw:+5.1f}  pitch {t.pitch:+5.1f}  roll {t.roll:+5.1f}   {t.label:20s}" + ("   NOD -> pick up / let go" if t.nod else "")
            if t.nod or t.label != last:
                print(line, flush=True)
                last = t.label
    except KeyboardInterrupt:
        pass
    finally:
        gaze.close()
        cam.close()


if __name__ == "__main__":
    import sys
    live(float(sys.argv[1]) if len(sys.argv) > 1 else 60.0)
