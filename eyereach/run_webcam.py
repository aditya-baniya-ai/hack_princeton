"""EyeReach with a webcam: move the arm by pointing your face (or with your eyes only). No keyboard, mouse or helper needed once
it is running; the screen (and a voice) always says what to do next.

    python3 -m eyereach.run_webcam              pretend arm: try this first
    python3 -m eyereach.run_webcam --real       the real arm (on the Jetson; hold the arm first)

1. It finds the face, then shows a yellow dot that glides around the screen. Pointing the nose at it
   (or, with --pointer eyes, following it with the eyes) teaches the computer where you point (about
   40 seconds). Your face is shown with a ray coming out of the nose, so you can see where you point.
2. Point at the table map: the arm follows immediately, within its speed limits.
3. Blink twice quickly to pick up where the arm is (or to let go when holding something).
4. Look at a big button and keep looking (it fills up): PICK UP / LET GO, STOP, FIX MY AIM.
Looking away or closing the eyes never moves anything. Every session is logged (eyereach/session_log.py).

Full screen by default (--windowed for a 1280 x 720 window).
Movement follows the pointer immediately; --movement dwell restores destination dwell.
Helper keys: Esc close, C calibrate again, Space pause (the arm holds still), V show/hide the camera view.
"""

import argparse
import shutil
import subprocess
import time
from pathlib import Path

import cv2
import numpy as np

from eyereach import eye_control, speaker_volume
from eyereach.head_tilt import HeadTilt
from eyereach.session_log import SessionLog
from eyereach.voice_commands import VoiceInput, MODEL
from eyereach.gaze_webcam import Camera, GazeMap, IrisGaze, OneEuro, camera_view, in_view

W, H = 1280, 720                                # the picture size; set_size() changes it to the screen's
WINDOW = "EyeReach"
PLAY_PART = (0.32, 0.20, 0.73, 0.91)            # the table map, as parts of the picture: left top right bottom
TILT_PLAY_PART = (0.03, 0.24, 0.29, 0.62)       # tilt control: a smaller map on the left; the face is in the middle
MAP = {"away_up": False}       # tilt control: face in the middle, "away from you" at the top of the map
BUTTONS_PART = (0.77, 0.20, 0.98, 0.91)         # the column of buttons
VOLUME_PART = (0.025, 0.60, 0.30, 0.95)         # the speaker volume, under the face picture (top: volume_area)
PLAY = (51, 122, 870, 684)                      # the table map in pixels (set by set_size)
COLUMNS, ROWS = (0.06, 0.35, 0.65, 0.94), (0.10, 0.50, 0.90)
DOT_SECONDS = 2.5                               # seconds of looking at each calibration dot
GLIDE = 0.9                                     # seconds the dot takes to glide to the next spot
SETTLE = 0.6                                    # seconds after it stops before the eyes count
RETRY_ERROR = 0.07                              # calibration worse than this part of the width: try once more
BG = (34, 26, 26)
DIM, WHITE, CYAN, PINK = (150, 140, 140), (235, 235, 235), (230, 220, 90), (180, 110, 255)
GOLD, GREEN, RED, GREY = (90, 200, 255), (120, 220, 120), (90, 90, 255), (90, 85, 85)
RAY_COLOR = (255, 230, 120)
TILT_LEAD = 0.2                # tilt control: the arm heads this far (map units) ahead of where it is
STILL_DEG_S = 25.0             # the head turning faster than this (degrees per second) isn't blinking on purpose
TELL = {   # what to do during calibration, per pointer mode: (on screen, spoken)
    "head": ("Point your nose at the yellow dot. Turn your head, not just your eyes.",
             "Point your nose at the yellow dot, and follow it by turning your head."),
    "eyes": ("Follow the yellow dot with your eyes. Keep your head still.",
             "Follow the yellow dot with your eyes, and keep your head still."),
    "both": ("Look at the yellow dot and turn your face toward it.",
             "Look at the yellow dot, and turn your face toward it."),
}


def rect(part):
    return tuple(int(p * (W if i % 2 == 0 else H)) for i, p in enumerate(part))


def set_size(width, height):
    """Draw everything at this size (the screen's, in full screen)."""
    global W, H, PLAY
    W, H = int(width), int(height)
    PLAY = rect(TILT_PLAY_PART if MAP["away_up"] else PLAY_PART)


def face_rect():
    """Where the face picture goes: (left, top, width, height). Tilt control: big, in the middle of the window."""
    if MAP["away_up"]:
        fw = int(min(0.40 * W, 0.6 * H * 4 / 3))
        return (W - fw) // 2, int(0.22 * H), fw, int(fw * 3 / 4)
    fw = int(min(0.27 * W, 0.55 * H * 4 / 3))
    return int(0.025 * W), int(0.25 * H), fw, int(fw * 3 / 4)


set_size(W, H)


def button_rects(n):
    """n buttons stacked in the button column, with gaps: [(left, top, right, bottom)]."""
    left, top, right, bottom = rect(BUTTONS_PART)
    gap = int(0.03 * H)
    height = (bottom - top - gap * (n - 1)) // n
    return [(left, top + i * (height + gap), right, top + i * (height + gap) + height) for i in range(n)]


def screen_size():
    """The screen's size in pixels, or None if it can't be read."""
    try:
        import tkinter
        root = tkinter.Tk()
        size = root.winfo_screenwidth(), root.winfo_screenheight()
        root.destroy()
        return size
    except Exception:                                  # no display, or no tkinter
        return None


def saved_path(mode="head"):
    """The calibration is in screen pixels and per pointer mode, so each screen size and mode keeps its own."""
    return Path.home() / ".eyereach" / f"calibration_{mode}_{W}x{H}.json"


def on_screen(point, slack=0.15):
    """Keep a predicted gaze point on the screen. A little outside (the fit overshoots at the edges)
    is pinned to the edge; far outside (looking away, head moved, half a blink) gives None."""
    if point is None:
        return None
    x, y = point
    if not (-slack * W <= x <= (1 + slack) * W and -slack * H <= y <= (1 + slack) * H):
        return None
    return min(max(x, 0.0), W - 1.0), min(max(y, 0.0), H - 1.0)




def to_area(point):
    """Screen pixel -> (u, v) on the table map, or None outside the map."""
    left, top, right, bottom = PLAY
    u, v = (point[0] - left) / (right - left), (point[1] - top) / (bottom - top)
    if MAP["away_up"]:
        v = 1 - v
    return (u, v) if 0 <= u <= 1 and 0 <= v <= 1 else None


def to_screen(u, v):
    left, top, right, bottom = PLAY
    if MAP["away_up"]:
        v = 1 - v
    return int(left + u * (right - left)), int(top + v * (bottom - top))


def volume_area():
    """The volume control's area: under the face picture and its label (whose size depends on the screen)."""
    left, _, right, bottom = rect(VOLUME_PART)
    if MAP["away_up"]:                             # tilt control: under the map, on the left
        left, right, top = PLAY[0], PLAY[2], PLAY[3] + int(0.07 * H)
    else:
        _, face_top, _, face_height = face_rect()
        top = face_top + face_height + int(0.075 * H)
    return left, min(top, bottom - int(0.2 * H)), right, bottom


def volume_rects():
    """The QUIETER and LOUDER buttons, side by side at the bottom of the volume area: [(name, label, rect)]."""
    left, top, right, bottom = volume_area()
    gap = int(0.015 * W)
    top = top + int(0.095 * H)                     # under the title and the bar
    half = (right - left - gap) // 2
    return [("volume down", "QUIETER", (left, top, left + half, bottom)),
            ("volume up", "LOUDER", (left + half + gap, top, right, bottom))]


def looking_at(ctl, point):
    """What a screen point is on: ("button", name), ("map", (u, v)) or None."""
    if point is None:
        return None
    for name, _, (left, top, right, bottom) in volume_rects():
        if left <= point[0] <= right and top <= point[1] <= bottom:
            return ("button", name)
    buttons = ctl.buttons()
    for (name, _), (left, top, right, bottom) in zip(buttons, button_rects(len(buttons))):
        if left <= point[0] <= right and top <= point[1] <= bottom:
            return ("button", name)
    area = to_area(point)
    return None if area is None else ("map", area)


# ---- drawing -------------------------------------------------------------------------------------

def scale(size):
    return size * H / 720


def thickness(thick):
    return max(1, int(round(thick * H / 720)))


def text(canvas, s, x, y, size=0.6, color=DIM, thick=1):
    cv2.putText(canvas, s, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, scale(size), color, thickness(thick),
                cv2.LINE_AA)


def centered(canvas, s, cx, y, size=0.6, color=DIM, thick=1, max_width=None):
    """Text centred on cx, made smaller if it would be wider than the screen."""
    width = cv2.getTextSize(s, cv2.FONT_HERSHEY_SIMPLEX, scale(size), thickness(thick))[0][0]
    limit = 0.94 * W if max_width is None else max_width
    if width > limit:
        size *= limit / width
        width = cv2.getTextSize(s, cv2.FONT_HERSHEY_SIMPLEX, scale(size), thickness(thick))[0][0]
    text(canvas, s, cx - width / 2, y, size, color, thick)


def paste(canvas, view, left, top):
    h, w = view.shape[:2]
    canvas[top:top + h, left:left + w] = view


def blank():
    return np.full((H, W, 3), BG, np.uint8)


_CROP = {"centre": None}       # the face's smoothed (x, y, size) in the camera picture, for face_crop


def face_crop(frame, pts):
    """The camera picture cut around the face, so the face is always in the middle (black past the picture's
    edges), following it smoothly. Returns (cropped picture, the face points moved into it, its (w, h))."""
    fh, fw = frame.shape[:2]
    if pts is not None:
        (x0, y0), (x1, y1) = pts.min(0), pts.max(0)
        target = np.array([(x0 + x1) / 2, (y0 + y1) / 2, max(y1 - y0, (x1 - x0) * 0.75)])
        last = _CROP["centre"]
        _CROP["centre"] = target if last is None else 0.8 * last + 0.2 * target
    centre = _CROP["centre"]
    if centre is None:
        return frame, pts, (fw, fh)
    ch = int(min(max(centre[2] * 2.6, 140), fh * 1.5))     # the face fills about 40% of the height
    cw = int(ch * 4 / 3)
    pad = max(cw, ch)
    big = cv2.copyMakeBorder(frame, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=(30, 24, 24))
    x0, y0 = int(centre[0] - cw / 2), int(centre[1] - ch / 2)
    crop = big[y0 + pad:y0 + pad + ch, x0 + pad:x0 + pad + cw]
    return crop, (None if pts is None else pts - np.array([x0, y0], float)), (cw, ch)


def face_panel(frame, gaze, ok, size):
    """The mirrored face, centred, with its key points, and where the nose is on it (or None)."""
    pts, frame_size = gaze.pts, gaze.size
    if frame is not None:
        frame, pts, frame_size = face_crop(frame, gaze.pts)
    view = camera_view(frame, pts, size, ok, None, frame_size)
    nose = None if pts is None else in_view(pts[1], frame_size, size)
    return view, nose


def draw_ray(canvas, start, end, now):
    """An animated ray from the nose on the face picture to the pointer: dashes moving outward."""
    start, end = np.asarray(start, float), np.asarray(end, float)
    length = float(np.linalg.norm(end - start))
    if length < 5:
        return
    unit = (end - start) / length
    cv2.line(canvas, tuple(start.astype(int)), tuple(end.astype(int)), (120, 100, 60), 1, cv2.LINE_AA)
    cv2.arrowedLine(canvas, tuple(start.astype(int)), tuple(end.astype(int)),
                    RAY_COLOR, thickness(2), cv2.LINE_AA, tipLength=min(0.08, 16 / length))


def tilt_status(ctl):
    """The short line under the face picture: what the head is doing (tilt control) or how to move (pointing)."""
    label = getattr(ctl, "tilt_label", None)
    if not getattr(ctl, "tilt", False) or label is None:
        return "Point to move"
    return {"centre": "Centre: still", "hold": "Hold the tilt...", "no face": "No face"}.get(label, "Moving " + label)


def draw(ctl, point, tracking, paused, fps, face=None):
    """The control screen, with a dedicated face panel and a calibrated aim line."""
    canvas = blank()
    if paused:
        centered(canvas, "Paused by the helper.", W // 2, 0.09 * H, 1.1, GOLD, 2)
    elif not tracking and ctl.control_mode == "vision":
        centered(canvas, "I can't see your face. Please look at the screen.", W // 2, 0.09 * H, 1.1, RED, 2)
    else:
        centered(canvas, ctl.instruction(), W // 2, 0.09 * H, 1.1, GOLD, 2)
    robot = ctl.robot
    text(canvas, f"{robot.name}  |  {fps:.0f} fps   |  helper keys: Esc quit, C fix aim, Space pause, V face picture, +/- volume",
         0.01 * W, 0.025 * H, 0.4)

    mode = "VOICE" if ctl.control_mode == "voice" else "VISION"
    status = getattr(ctl, "microphone_status", "Voice input off")
    centered(canvas, f"{mode}  |  {status}", W // 2, 0.135 * H, 0.55, CYAN)

    # the table map
    left, top, right, bottom = PLAY
    cv2.rectangle(canvas, (left, top), (right, bottom), (150, 110, 130), 2)
    near, far = ("near you", "away from you")
    centered(canvas, far if MAP["away_up"] else near, (left + right) // 2, top - 0.012 * H, 0.55)
    centered(canvas, near if MAP["away_up"] else "near the arm", (left + right) // 2, bottom + 0.035 * H, 0.55)
    ring = int(scale(26))
    if ctl.target is not None:                             # the chosen spot
        t = to_screen(*ctl.target)
        cv2.circle(canvas, t, ring, GREEN, 2, cv2.LINE_AA)
        cv2.drawMarker(canvas, t, GREEN, cv2.MARKER_CROSS, ring, 2, cv2.LINE_AA)
    if robot.command is not None:                          # the claw: a ring, pink while holding
        u, v, _ = robot.gripper_area()
        claw = to_screen(min(1, max(0, u)), min(1, max(0, v)))
        color = PINK if ctl.holding else CYAN
        cv2.circle(canvas, claw, ring, color, 3, cv2.LINE_AA)
        centered(canvas, "arm", claw[0], claw[1] + ring + scale(22), 0.55, color)

    # the speaker volume: a bar, and QUIETER / LOUDER buttons (gaze dwell or mouse, like the others)
    left, top, right, _ = volume_area()
    volume = getattr(ctl, "volume", None)
    centered(canvas, "SPEAKER VOLUME" + ("" if volume is None else f"  {volume}%"), (left + right) // 2,
             top + scale(22), 0.7, WHITE, 2, max_width=0.95 * (right - left))
    bar_top, bar_height = top + int(0.045 * H), int(0.03 * H)
    cv2.rectangle(canvas, (left, bar_top), (right, bar_top + bar_height), (60, 48, 48), -1)
    if volume is not None:
        cv2.rectangle(canvas, (left, bar_top), (left + int((right - left) * volume / 100), bar_top + bar_height), GOLD, -1)
    cv2.rectangle(canvas, (left, bar_top), (right, bar_top + bar_height), DIM, 1)
    for name, label, (bl, bt, br, bb) in volume_rects():
        on = ctl.looking == ("button", name)
        color = GOLD if on else WHITE
        cv2.rectangle(canvas, (bl, bt), (br, bb), (60, 48, 48), -1)
        if on:
            cv2.rectangle(canvas, (bl, bt), (bl + int((br - bl) * ctl.progress), bb), (40, 110, 140), -1)
        cv2.rectangle(canvas, (bl, bt), (br, bb), color, 3 if on else 2)
        centered(canvas, label, (bl + br) // 2, (bt + bb) // 2 + scale(12), 0.95, color, 2, max_width=0.85 * (br - bl))

    # the buttons
    buttons = ctl.buttons()
    for (name, label), (bl, bt, br, bb) in zip(buttons, button_rects(len(buttons))):
        active = ctl.can(name)
        on = ctl.looking == ("button", name)
        color = (GOLD if on else WHITE) if active else GREY
        cv2.rectangle(canvas, (bl, bt), (br, bb), (60, 48, 48) if active else (44, 36, 36), -1)
        if on:                                             # fills from the left while looked at
            cv2.rectangle(canvas, (bl, bt), (bl + int((br - bl) * ctl.progress), bb), (40, 110, 140), -1)
        cv2.rectangle(canvas, (bl, bt), (br, bb), color, 3 if on else 2)
        centered(canvas, label, (bl + br) // 2, (bt + bb) // 2 + scale(14), 1.2, color, 2, max_width=0.88 * (br - bl))

    # the face, and the ray out of the nose to where it points
    if face is not None:
        view, nose = face
        corner = face_rect()[:2]
        paste(canvas, view, *corner)
        centered(canvas, "YOU", corner[0] + view.shape[1] // 2, corner[1] - scale(18), 0.8, WHITE, 2)
        centered(canvas, "Say Robot move left" if ctl.control_mode == "voice" else tilt_status(ctl), corner[0] + view.shape[1] // 2,
                 corner[1] + view.shape[0] + scale(35), 0.65, GOLD, 2)
        direction = getattr(ctl, "tilt_dir", None)
        if getattr(ctl, "tilt", False) and direction is not None and not paused:
            vw, vh = view.shape[1], view.shape[0]      # a big arrow from the middle of the face, the way it moves
            mid = (corner[0] + vw // 2, corner[1] + vh // 2)
            tip = (int(mid[0] + direction[0] * 0.27 * vw), int(mid[1] - direction[1] * 0.27 * vh))
            cv2.arrowedLine(canvas, mid, tip, (30, 24, 24), int(scale(14)), cv2.LINE_AA, tipLength=0.3)
            cv2.arrowedLine(canvas, mid, tip, GOLD, int(scale(8)), cv2.LINE_AA, tipLength=0.3)
        if getattr(ctl, "tilt", False):                # which way is which, lit up while moving that way
            label = getattr(ctl, "tilt_label", "") or ""
            vw, vh = view.shape[1], view.shape[0]
            for word, (x, y) in (("LEFT", (0.1, 0.5)), ("RIGHT", (0.9, 0.5)), ("AWAY", (0.5, 0.09)),
                                 ("TOWARD YOU", (0.5, 0.86))):
                on = word.lower() in label
                size, thick = (0.6, 2) if on else (0.5, 1)
                cx, cy = int(corner[0] + x * vw), int(corner[1] + y * vh)
                tw, th = cv2.getTextSize(word, cv2.FONT_HERSHEY_SIMPLEX, scale(size), thickness(thick))[0]
                cv2.rectangle(canvas, (cx - tw // 2 - 6, cy - th - 6), (cx + tw // 2 + 6, cy + 6), (30, 24, 24), -1)
                centered(canvas, word, cx, cy, size, GOLD if on else WHITE, thick)
        if not getattr(ctl, "tilt", False) and nose is not None and point is not None and not paused:
            draw_ray(canvas, (corner[0] + nose[0], corner[1] + nose[1]), point, time.monotonic())

    # where the pointer is, with a ring that fills while it rests on the map
    if point is not None and not paused:
        p = (int(point[0]), int(point[1]))
        cv2.circle(canvas, p, int(scale(9)), GOLD, -1, cv2.LINE_AA)
        if ctl.movement == "dwell" and ctl.looking is not None and ctl.looking[0] == "map":
            cv2.ellipse(canvas, p, (ring, ring), 0, -90, -90 + 360 * ctl.progress, GREEN, 3, cv2.LINE_AA)
    heard = getattr(ctl, "heard_command", "")
    if heard:
        centered(canvas, "Heard: " + heard, W // 2, 0.985 * H, 0.5, WHITE)
    return canvas


# ---- talking -------------------------------------------------------------------------------------

class Voice:
    """Says sentences out loud (speech-dispatcher's spd-say), without waiting. Quiet if it's missing."""

    def __init__(self, on=True):
        self.on = on and shutil.which("spd-say") is not None
        self.last = None

    def say(self, sentence):
        if not self.on or not sentence or sentence == self.last:
            return
        self.last = sentence
        quiet = dict(stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        subprocess.Popen(["spd-say", "-C"], **quiet).wait()                # stop the last sentence
        subprocess.Popen(["spd-say", "-r", "-25", sentence], **quiet)


# ---- calibration ---------------------------------------------------------------------------------

def look(camera, gaze):
    """One camera frame and its eye features (None: no face or a blink)."""
    frame = camera.read()
    if frame is None:
        gaze.pts, gaze.opening, gaze.pose = None, None, None
        return None, None
    return frame, gaze.features(frame)


def wait(seconds):
    """Keep the window alive for a while. True if Esc was pressed."""
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        if cv2.waitKey(30) & 0xFF == 27:
            return True
    return False


def welcome(camera, gaze, shown, voice, again=False):
    tell_screen, tell_voice = TELL[gaze.mode]
    """Explain, then wait until the face has been steady in view for 3 seconds. False: closed."""
    size = (int(0.42 * W), int(0.315 * W))
    if size[1] > 0.5 * H:
        size = (int(0.5 * H * 4 / 3), int(0.5 * H))
    seen_since, lost_since = None, time.monotonic()
    title = "Let's fix your aim." if again else "Hello!"
    voice.say(("Let's fix your aim. " if again else "Hello. First I need to learn where you are pointing. ") +
              tell_voice)
    while True:
        frame, f = look(camera, gaze)
        now = time.monotonic()
        if gaze.pts is not None:
            seen_since, lost_since = seen_since or now, None
        else:
            seen_since, lost_since = None, lost_since or now
        canvas = blank()
        centered(canvas, title, W // 2, 0.09 * H, 1.4, GOLD, 2)
        centered(canvas, "First I need to learn where you are pointing.", W // 2, 0.16 * H, 0.9, WHITE, 2)
        centered(canvas, tell_screen, W // 2, 0.22 * H, 0.9, WHITE, 2)
        top = int(0.27 * H)
        paste(canvas, face_panel(frame, gaze, f is not None, size)[0], (W - size[0]) // 2, top)
        below = top + size[1] + scale(45)
        if seen_since is None:
            centered(canvas, "I can't see your face.", W // 2, below, 1.0, RED, 2)
            centered(canvas, "Helper: move the camera so the face is in the middle of the picture.",
                     W // 2, below + scale(36), 0.65)
            if now - lost_since > 4:
                voice.say("I can't see your face. Please look at the screen.")
        else:
            left = 3 - (now - seen_since)
            if left <= 0:
                return True
            centered(canvas, f"Starting in {int(left) + 1} ...", W // 2, below, 1.0, GREEN, 2)
        shown(canvas)
        if cv2.waitKey(1) & 0xFF == 27:
            return False


def tilt_step(ctl, tilt, tracking, paused, now, voice_acted=False):
    """One frame of tilt control: head for a spot TILT_LEAD ahead of the arm in the tilt's direction, at the
    tilt's speed; no tilt (or no face, or paused) and the arm stops. A nod is handled before this (ctl.blink)."""
    moving = tracking and tilt is not None and tilt.direction is not None and not voice_acted and not paused
    base = (ctl.robot.carry if ctl.holding else ctl.robot.normal)[0]
    ctl.robot.glide = base * max(0.25, tilt.speed) if moving else base
    target = None
    if moving:
        u, v, _ = ctl.robot.gripper_area()
        target = (min(1.0, max(0.0, u + tilt.direction[0] * TILT_LEAD)),
                  min(1.0, max(0.0, v + tilt.direction[1] * TILT_LEAD)))
    ctl.step(("map", target) if target else None, now, False,
             suspended=paused or (ctl.control_mode == "vision" and not tracking))
    return target


def framing_hint(gaze):
    """A tip for the helper when the face is near an edge of the picture (turning or nodding then loses it)."""
    if gaze.pts is None:
        return ""
    w, h = gaze.size
    (x0, y0), (x1, y1) = gaze.pts.min(0), gaze.pts.max(0)
    if h - y1 < 0.15 * h:
        return "Helper: the face is low in the picture. Tilt the camera down a little."
    if y0 < 0.1 * h:
        return "Helper: the face is high in the picture. Tilt the camera up a little."
    if x0 < 0.12 * w or w - x1 < 0.12 * w:
        return "Helper: the face is near the side. Turn the camera toward the face."
    return ""


def find_centre(camera, gaze, shown, voice, again=False):
    """Tilt control: the head's direction when looking straight ahead (degrees: yaw, pitch), from 1.5 s of
    looking at the screen once the face has been in view for a second. None if Esc was pressed."""
    size = (int(0.42 * W), int(0.315 * W))
    if size[1] > 0.5 * H:
        size = (int(0.5 * H * 4 / 3), int(0.5 * H))
    voice.say("Let's set your centre. Look straight at the screen and keep still." if again else
              "Hello. Look straight at the screen and keep still for two seconds. "
              "Then turn your head to move the arm, and nod to pick something up.")
    seen_since, poses = None, []
    while True:
        frame, f = look(camera, gaze)
        now = time.monotonic()
        if gaze.pose is None:
            seen_since, poses = None, []
        else:
            seen_since = seen_since or now
            if now - seen_since > 1.0:
                poses.append(np.degrees(gaze.pose[:3]))
        canvas = blank()
        centered(canvas, "Set your centre" if again else "Hello!", W // 2, 0.09 * H, 1.4, GOLD, 2)
        centered(canvas, "Look straight at the screen and keep still.", W // 2, 0.16 * H, 0.9, WHITE, 2)
        centered(canvas, "Then: turn your head to move the arm. Nod to pick up or let go.", W // 2, 0.22 * H,
                 0.8, WHITE, 2)
        top = int(0.27 * H)
        paste(canvas, face_panel(frame, gaze, f is not None, size)[0], (W - size[0]) // 2, top)
        below = top + size[1] + scale(45)
        edge = framing_hint(gaze)
        if edge:
            centered(canvas, edge, W // 2, top - scale(12), 0.7, GOLD, 2)
        if seen_since is None:
            centered(canvas, "I can't see your face.", W // 2, below, 1.0, RED, 2)
            centered(canvas, "Helper: move the camera so the whole face is in the picture.", W // 2,
                     below + scale(36), 0.65)
        else:
            done = min(1.0, (now - seen_since) / 2.5)
            bar = int(0.3 * W)
            x0 = (W - bar) // 2
            cv2.rectangle(canvas, (x0, int(below)), (x0 + bar, int(below + scale(10))), (60, 48, 48), -1)
            cv2.rectangle(canvas, (x0, int(below)), (x0 + int(bar * done), int(below + scale(10))), GREEN, -1)
            if len(poses) >= 15 and done >= 1.0:
                centre = np.median(np.array(poses), axis=0)
                voice.say("Ready. Turn your head to move the arm. Nod to pick up.")
                return float(centre[0]), float(centre[1]), float(centre[2])
        shown(canvas)
        if cv2.waitKey(1) & 0xFF == 27:
            return None


def dot_path():
    """The calibration spots, row by row like a snake so the dot never jumps across the screen."""
    spots = []
    for i, row in enumerate(ROWS):
        for column in (COLUMNS if i % 2 == 0 else COLUMNS[::-1]):
            spots.append((int(column * W), int(row * H)))
    return spots


def follow_the_dot(camera, gaze, shown, voice, seconds):
    """The dot glides from spot to spot and waits at each while the eyes are seen. (map, error) or None."""
    X, Y = [], []
    spots = dot_path()
    radius = int(scale(22))
    panel = (int(0.18 * W), int(0.135 * W))
    for n, spot in enumerate(spots):
        start = spots[n - 1] if n else spot
        began = last = time.monotonic()
        looked, lost_since = 0.0, None
        while looked < seconds:
            frame, f = look(camera, gaze)
            now = time.monotonic()
            moving = min(1.0, (now - began) / GLIDE)                  # the first dot grows in instead
            ease = moving * moving * (3 - 2 * moving)                  # smooth start and stop
            at = (int(start[0] + (spot[0] - start[0]) * ease), int(start[1] + (spot[1] - start[1]) * ease))
            if moving >= 1.0 and f is not None and (n or now - began > 2 * GLIDE):   # waits while eyes are lost
                looked += now - last
                if looked > SETTLE:
                    X.append(f), Y.append(spot)
            lost_since = None if f is not None else (lost_since or now)
            last = now
            canvas = blank()
            pulse = (1 + 0.15 * np.sin(now * 6)) * (moving if n == 0 else 1.0)
            cv2.circle(canvas, at, int(radius * pulse), GOLD, -1, cv2.LINE_AA)
            cv2.circle(canvas, at, int(scale(4)), BG, -1, cv2.LINE_AA)  # a centre to look at
            if lost_since and now - lost_since > 0.5:
                centered(canvas, "I can't see your face. Look at the yellow dot.", W // 2, H // 2 - scale(20),
                         0.9, RED, 2)
                if now - lost_since > 4:
                    voice.say("I can't see your face. Please look at the yellow dot.")
            bar, tall = int(0.3 * W), int(scale(8))                    # how far along it is
            x0, y0 = (W - bar) // 2, int(H / 2 + scale(20))
            cv2.rectangle(canvas, (x0, y0), (x0 + bar, y0 + tall), (60, 48, 48), -1)
            done = (n + min(1.0, looked / seconds)) / len(spots)
            cv2.rectangle(canvas, (x0, y0), (x0 + int(bar * done), y0 + tall), GOLD, -1)
            centered(canvas, TELL[gaze.mode][0], W // 2, y0 + tall + scale(40), 0.75)
            view, nose = face_panel(frame, gaze, f is not None, panel)          # the face and its ray
            paste(canvas, view, (W - panel[0]) // 2, int(0.64 * H))
            shown(canvas)
            if cv2.waitKey(1) & 0xFF == 27:
                return None
    if len(X) < 100:
        print(f"Only {len(X)} good frames: is the face in view and well lit?")
        return None
    gmap = GazeMap.fit(X, Y)
    Y = np.asarray(Y)
    miss = np.sqrt(np.sum((gmap.predict(X) - Y) ** 2, axis=1))
    error = float(np.sqrt(np.mean(miss ** 2)))
    per_dot = [float(np.sqrt(np.mean(miss[(Y == spot).all(axis=1)] ** 2))) for spot in spots]
    print(f"Calibration done. Average error on the dots: {error:.0f} px "
          f"({100 * error / W:.1f}% of the screen width; it looks better here than it will in use)")
    print("  error at each dot (px, in the order shown): " + " ".join(f"{e:.0f}" for e in per_dot))
    return gmap, error, per_dot


def calibrate(camera, gaze, shown, voice=None, seconds=DOT_SECONDS, again=False, log=None):
    """Explain, show the gliding dot and learn the gaze map. Tries once more if it came out poor.
    Returns (map, error in pixels) or None if closed. Each try goes in the session log."""
    voice = voice or Voice(on=False)
    for attempt in (1, 2):
        began = time.monotonic()
        if not welcome(camera, gaze, shown, voice, again or attempt == 2):
            return None
        done = follow_the_dot(camera, gaze, shown, voice, seconds)
        if done is None:
            return None
        gmap, error, per_dot = done
        if log:
            log.calibration(time.monotonic() - began, error, per_dot)
        if error / W <= RETRY_ERROR or attempt == 2:
            break
        canvas = blank()
        centered(canvas, "Almost. Let's do that once more.", W // 2, H // 2, 1.3, GOLD, 2)
        shown(canvas)
        voice.say("Almost. Let's do that once more.")
        if wait(2.5):
            return None
    canvas = blank()
    centered(canvas, "All set!", W // 2, 0.42 * H, 1.8, GREEN, 3)
    centered(canvas, "Point at the table map to move. Blink twice to pick up or let go.", W // 2,
             0.54 * H, 0.9, WHITE, 2)
    shown(canvas)
    voice.say("All set. Point at the table map to move. Blink twice to pick up or let go.")
    return None if wait(3.0) else (gmap, error)


# ---- the app -------------------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Control the arm with your eyes, using a webcam.")
    parser.add_argument("--real", action="store_true", help="the real arm (default: a pretend arm)")
    parser.add_argument("--camera", default="0", help="camera number or path (default 0; Jetson /dev/video0)")
    parser.add_argument("--reuse", action="store_true", help="skip calibration, use the saved one")
    parser.add_argument("--control", choices=("tilt", "point"), default="tilt",
                        help="tilt: turn your head to move the arm, nod to pick up (default). "
                             "point: point at the table map on the screen (needs the dot calibration)")
    parser.add_argument("--pointer", choices=("head", "eyes", "both"), default="head",
                        help="head: point with your face (steady; default). eyes: eyes only, for people who "
                             "can't move their head. both: the face for the rough spot, the eyes to fine-tune")
    parser.add_argument("--dwell", type=float, default=0.65,
                        help=f"seconds of looking that press a button or choose a spot (default {eye_control.LOOK})")
    parser.add_argument("--movement", choices=("follow", "dwell"), default="follow",
                        help="follow the pointer in real time (default), or dwell to choose destinations")
    parser.add_argument("--dot-seconds", type=float, default=DOT_SECONDS,
                        help=f"seconds of looking at each calibration dot (default {DOT_SECONDS})")
    parser.add_argument("--voice", dest="voice", action="store_true", help="enable offline microphone commands")
    parser.add_argument("--no-voice", dest="voice", action="store_false", help="disable microphone input")
    parser.set_defaults(voice=False)
    parser.add_argument("--voice-model", type=Path, default=MODEL, help="local Vosk model folder")
    parser.add_argument("--microphone", help="input device index or name (default: system input)")
    parser.add_argument("--list-microphones", action="store_true", help="list input devices and exit")
    parser.add_argument("--quiet", action="store_true", help="no voice")
    parser.add_argument("--grab-mm", type=float, default=eye_control.GRAB_MM,
                        help=f"fingertips this high above the table when picking up (default {eye_control.GRAB_MM:.0f}, "
                             f"never below {eye_control.LOWEST_GRAB_MM:.0f})")
    parser.add_argument("--travel-mm", type=float, default=eye_control.HOVER_MM,
                        help=f"fingertips this high above the table at startup and while moving "
                             f"(default {eye_control.HOVER_MM:.0f}, at least 10 above --grab-mm)")
    parser.add_argument("--person", default="anonymous", help="who is using it, for the session log")
    parser.add_argument("--screen-cm", type=float, help="width of the picture on the screen in cm (for degrees)")
    parser.add_argument("--distance-cm", type=float, help="eyes to screen in cm (for degrees)")
    parser.add_argument("--windowed", action="store_true", help="a 1280 x 720 window instead of full screen")
    parser.add_argument("--fullscreen", action="store_true", help=argparse.SUPPRESS)   # the default now
    args, ros_args = parser.parse_known_args()

    if args.list_microphones:
        import sounddevice as sd
        print(sd.query_devices())
        return
    size = None if args.windowed else screen_size()
    if size:
        set_size(*size)
    camera, gaze, voice = Camera(args.camera), IrisGaze(args.pointer), Voice(on=not args.quiet and not args.voice)
    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL | cv2.WINDOW_FREERATIO)
    cv2.resizeWindow(WINDOW, W, H)
    if not args.windowed:
        # Some desktops (Remote Desktop among them) ignore the full screen request alone, so the
        # window is also put at the top left and sized to the screen once it is showing.
        for _ in range(5):
            cv2.imshow(WINDOW, blank())
            cv2.waitKey(30)
        cv2.moveWindow(WINDOW, 0, 0)
        cv2.resizeWindow(WINDOW, W, H)
        cv2.setWindowProperty(WINDOW, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    show = lambda canvas: cv2.imshow(WINDOW, canvas)       # noqa: E731
    robot, listener = None, None
    log = SessionLog(args.person, W, args.screen_cm, args.distance_cm, "real" if args.real else "pretend")
    log.event("pointer", mode=args.pointer)
    print(f"Logging this session to {log.path}")
    try:
        saved = saved_path(args.pointer)
        gmap = GazeMap.load(saved) if args.reuse and saved.exists() else None
        if gmap is None and args.control == "point":
            done = calibrate(camera, gaze, show, voice, args.dot_seconds, log=log)
            if done is None:
                return
            gmap = done[0]
            gmap.save(saved)
        robot = eye_control.arm_link.RealArm(ros_args=ros_args) if args.real else eye_control.arm_link.VirtualArm()
        robot.hover = max(args.travel_mm, max(args.grab_mm, eye_control.LOWEST_GRAB_MM) + 10)
        why = robot.start()
        if why:
            print(f"Can't start the arm: {why}")
            return
        if not args.real:
            robot.held = True        # the pretend arm: every pick-up holds something
        ctl = eye_control.GuidedControl(robot, look=args.dwell, grab_mm=args.grab_mm, movement=args.movement)
        ctl.volume, volume_checked = speaker_volume.get_volume(), time.monotonic()
        head = None
        if args.control == "tilt":                     # no screen pointer: the head's tilt moves the arm
            ctl.movement, ctl.tilt = "follow", True
            MAP["away_up"] = True                      # face in the middle; tilt up = away = up on the map
            set_size(W, H)
            centre = find_centre(camera, gaze, show, voice)
            if centre is None:
                return
            head = HeadTilt()
            head.set_centre(*centre)
            log.event("tilt centre", yaw=round(centre[0], 1), pitch=round(centre[1], 1), roll=round(centre[2], 1))
        if args.voice:
            device = int(args.microphone) if args.microphone and args.microphone.isdigit() else args.microphone
            if device is None:                         # the speaker's microphone first, the webcam's as fallback
                from eyereach.audio_devices import pick_microphone
                device = pick_microphone()
            listener = VoiceInput(args.voice_model, device).start()
        microphone_failed = False
        clicks = []

        def mouse(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                hit = looking_at(ctl, (x, y))
                if hit is not None and hit[0] == "button":
                    clicks.append(hit[1])

        cv2.setMouseCallback(WINDOW, mouse)
        blinks, last_pose = eye_control.DoubleBlink(), None
        smooth, paused, last, lost_since = OneEuro(still_hz=1.2, speed_up=0.004), False, time.monotonic(), None
        view_on = True
        view_size = face_rect()[2:]
        if args.reuse:
            voice.say(ctl.instruction())
        while True:
            frame, f = look(camera, gaze)
            now = time.monotonic()
            tilt = None
            if head is not None:                       # tilt control: a face is all it needs
                tilt = head.update(None if gaze.pose is None else tuple(np.degrees(gaze.pose[:3])), now)
                tracking = gaze.pose is not None
                point = None
                if tracking:
                    u, v, _ = ctl.robot.gripper_area()
                    if tilt.direction is not None and ctl.live_enabled:
                        u = min(1.0, max(0.0, u + tilt.direction[0] * TILT_LEAD))
                        v = min(1.0, max(0.0, v + tilt.direction[1] * TILT_LEAD))
                    point = to_screen(u, v)            # the ray goes from the nose to where the arm heads
                ctl.tilt_label = tilt.label if tracking else "no face"
                ctl.tilt_dir = tilt.direction if tracking and ctl.live_enabled else None
            else:
                raw = None if f is None else on_screen(tuple(gmap.predict(f)[0]))
                point = smooth(raw) if raw is not None else None
                # head pointing doesn't need open eyes (narrow eyes looking down are fine); eye pointing does
                eyes_open = args.pointer == "head" or (gaze.opening is not None and gaze.opening >=
                                                       max(0.10, 0.55 * (blinks.usual or gaze.opening)))
                tracking = raw is not None and eyes_open
            pose = None if gaze.pose is None else np.degrees(gaze.pose[:2])
            turning = (0.0 if pose is None or last_pose is None else
                       float(np.max(np.abs(pose - last_pose))) / max(now - last, 1e-3))   # degrees per second
            last_pose = pose
            double = blinks.update(gaze.opening, now, still=turning < STILL_DEG_S) and not paused
            if head is not None:
                double = False                         # tilt control: a nod picks up, blinks do nothing
                if tilt.nod and not paused and ctl.control_mode == "vision":
                    ctl.press("let go" if ctl.holding else "pick up", now, how="nod")
            voice_acted = False
            if listener:
                ctl.voice_available = listener.status == "Listening" and not listener.error
                ctl.microphone_status = listener.error or listener.status
                if listener.error and not microphone_failed:
                    ctl.press("stop", now)
                    microphone_failed = True
                    log.event("voice unavailable", message=listener.error)
                for text, why, lowest in listener.poll_dropped():
                    ctl.heard_command = f"{text} - {why}"
                    log.event("voice heard", text=text, why=why, lowest_confidence=lowest)
                events = listener.poll()
                # STOP wins over every other recognized action in this batch.
                if any(event.action == "stop" for event in events):
                    ctl.voice("stop", now)
                    ctl.heard_command = "stop - accepted"
                    voice_acted = True
                    clicks.clear()
                    double = False
                elif not paused and not listener.error:
                    for event in events:
                        accepted = ctl.voice(event.action, now, event.started, event.heard)
                        ctl.heard_command = event.text + (" - accepted" if accepted else " - ignored")
                        log.event("voice command", text=event.text, accepted=accepted)
                        if accepted:
                            voice_acted = True
                            clicks.clear()
                            double = False
                            break  # never execute a batch of old commands

            if clicks:
                name = clicks.pop(0)
                clicks.clear()
                if not paused:
                    ctl.press(name, now, how="click")
                    ctl.dwell = eye_control.Dwell(args.dwell)
                    double = False
            if head is not None:
                tilt_step(ctl, tilt, tracking, paused, now, voice_acted)
            else:
                ctl.step(None if paused or not tracking or voice_acted else looking_at(ctl, point), now, double,
                         suspended=paused or (ctl.control_mode == "vision" and not tracking))
            if ctl.wants_volume:                           # QUIETER / LOUDER was pressed (or + / - typed)
                ctl.volume = speaker_volume.change(ctl.wants_volume)
                ctl.wants_volume = 0
                volume_checked = now
            elif now - volume_checked > 2.0:               # follow changes made elsewhere (the volume keys)
                ctl.volume, volume_checked = speaker_volume.get_volume(), now
            for name, details in ctl.events:
                log.event(name, **details)
            ctl.events = []
            lost_since = None if tracking else (lost_since or now)
            lost = lost_since is not None and now - lost_since > 0.7     # not just a blink
            if ctl.say:                                    # what just happened, then what to do next
                voice.say(" ".join(ctl.say) + " " + ctl.instruction())
                ctl.say = []
            elif lost_since and now - lost_since > 4:
                voice.say("I can't see your face. Please look at the screen.")
            face = face_panel(frame, gaze, f is not None, view_size) if view_on else None
            canvas = draw(ctl, point if tracking else None, not lost, paused, 1 / max(now - last, 1e-3), face)
            show(canvas)
            last = now
            key = cv2.waitKey(1) & 0xFF
            if key == 27 or cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                break
            if key in (ord("+"), ord("=")):
                ctl.press("volume up", now, how="key")
            elif key in (ord("-"), ord("_")):
                ctl.press("volume down", now, how="key")
            if key == ord(" "):
                paused = not paused
                if paused:
                    ctl.press("stop", now)
            elif key in (ord("v"), ord("V")):
                view_on = not view_on
            if key in (ord("c"), ord("C")) or ctl.wants_calibration:
                ctl.press("stop", now)
                ctl.step(None, suspended=True)
                blinks = eye_control.DoubleBlink()
                ctl.wants_calibration = False
                if head is not None:
                    centre = find_centre(camera, gaze, show, voice, again=True)
                    if centre is None:
                        break
                    head.set_centre(*centre)
                    log.event("tilt centre", yaw=round(centre[0], 1), pitch=round(centre[1], 1), roll=round(centre[2], 1))
                else:
                    done = calibrate(camera, gaze, show, voice, args.dot_seconds, again=True, log=log)
                    if done is None:
                        break
                    gmap = done[0]
                    gmap.save(saved)
                ctl.accept_after = time.monotonic()
                if listener:
                    listener.poll()
                smooth = OneEuro(still_hz=1.2, speed_up=0.004)
    except KeyboardInterrupt:
        pass
    finally:
        if listener:
            listener.close()
        log.close()
        cv2.destroyAllWindows()
        gaze.close()
        camera.close()
        if robot is not None:
            if robot.closed:
                robot.open_here()
            robot.stop()                 # the real arm opens the gripper and waits for it first


if __name__ == "__main__":
    main()
