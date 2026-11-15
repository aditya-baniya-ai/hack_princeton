"""Webcam pointing from MediaPipe Face Mesh: where the face points (head) and/or the irises (eyes), mapped
to a point on the screen.

IrisGaze.features() turns a camera frame into a few numbers about where the person points:
  head  the direction the face points, from key points (eye corners, nose tip, forehead, chin, mouth
        corners) fitted to a 3D face model: a steady "ray" out of the nose. Needs some head movement.
  eyes  the irises inside the eyes. No head movement needed, but small and jittery in a webcam picture.
  both  the two together (the head for the rough spot, the eyes to fine-tune).
GazeMap learns the mapping from those numbers to screen pixels from a short calibration
(follow the dot). OneEuro steadies the result.
"""

import json
import math
from pathlib import Path

import cv2
import numpy as np

# MediaPipe Face Mesh landmark numbers (refine_landmarks=True adds the iris points 468-477)
RIGHT = dict(outer=33, inner=133, upper=159, lower=145, iris=468)    # the person's right eye
LEFT = dict(outer=263, inner=362, upper=386, lower=374, iris=473)
NOSE = 1
BLINK = 0.12      # eye opening / eye width below this is a blink: the frame is skipped
MODES = ("head", "eyes", "both")

FOREHEAD, CHIN = 10, 152
RAY = 1.6         # the ray out of the nose is drawn this many eye-spans long on the face picture


def eye_opening(pts):
    """How open the eyes are: eyelid gap / eye width, both eyes averaged (about 0.25 open, under 0.12 shut)."""
    pts = np.asarray(pts, float)
    across = pts[LEFT["outer"]] - pts[RIGHT["outer"]]
    h = across / np.linalg.norm(across)
    v = np.array([-h[1], h[0]])
    return float(np.mean([abs((pts[e["lower"]] - pts[e["upper"]]) @ v) / np.linalg.norm(pts[e["outer"]] - pts[e["inner"]])
                          for e in (RIGHT, LEFT)]))


def features_from_landmarks(pts, size):
    """pts: (478, 2) landmark pixels; size: (width, height) of the frame.
    Returns [gaze_x, gaze_y, eye_opening, head_x, head_y, face_x, face_y], or None on a blink.
    gaze_* is the iris offset from the middle of the eye in eye widths (along / down the face),
    head_* is the nose offset from between the eyes (turning the head), face_* is where the face is."""
    pts = np.asarray(pts, float)
    across = pts[LEFT["outer"]] - pts[RIGHT["outer"]]
    span = np.linalg.norm(across)
    h = across / span                      # across the face, image left -> right
    v = np.array([-h[1], h[0]])            # down the face
    gaze, opening = [], []
    for eye in (RIGHT, LEFT):
        a, b = pts[eye["inner"]], pts[eye["outer"]]
        width = np.linalg.norm(b - a)
        d = pts[eye["iris"]] - (a + b) / 2
        gaze.append(np.array([d @ h, d @ v]) / width)
        opening.append(abs((pts[eye["lower"]] - pts[eye["upper"]]) @ v) / width)
    opening = float(np.mean(opening))
    if opening < BLINK:
        return None
    eyes = (pts[RIGHT["outer"]] + pts[LEFT["outer"]]) / 2
    nose = (pts[NOSE] - eyes)
    head = np.array([nose @ h, nose @ v]) / span
    return np.array([*np.mean(gaze, axis=0), opening, *head, *(eyes / np.asarray(size, float))])


def unit(v):
    return v / (np.linalg.norm(v) + 1e-9)


def head_pose(pts3, size):
    """Which way the face points, from the 3D key points Face Mesh gives (x, y in pixels, z depth in the
    same units, smaller = nearer the camera): the eye-to-eye axis and the chin-to-forehead axis.
    Returns (yaw, pitch, roll in radians, nose x and y as parts of the frame, out-of-face direction,
    eye span in pixels), or None. yaw > 0: the face turned toward the picture's right; pitch > 0: up."""
    w, h = size
    pts3 = np.asarray(pts3, float)
    across = unit(pts3[LEFT["outer"]] - pts3[RIGHT["outer"]])        # to the person's left
    up = pts3[FOREHEAD] - pts3[CHIN]
    up = unit(up - (up @ across) * across)                             # chin to forehead, square to across
    out = np.cross(across, up)                                         # out of the face (toward the camera: -z)
    if not np.all(np.isfinite(out)):
        return None
    yaw = math.atan2(out[0], -out[2])
    pitch = math.atan2(-out[1], math.hypot(out[0], out[2]))
    roll = math.atan2(-across[1], across[0])
    span = float(np.linalg.norm(pts3[LEFT["outer"], :2] - pts3[RIGHT["outer"], :2]))
    return yaw, pitch, roll, pts3[NOSE][0] / w, pts3[NOSE][1] / h, out, span


def nose_offset(pts):
    """Where the nose tip is from the point between the eyes, in eye-widths, measured along the face's own
    across and down directions: (sideways, down). Sideways > 0 is toward the picture's right (the person's
    own left, as the face turns left); down shrinks as the head tilts up. Only the 2D picture is used, so it
    doesn't depend on the depth guess, and dividing by the eye width makes it the same near or far."""
    pts = np.asarray(pts, float)
    right_eye, left_eye = pts[RIGHT["outer"], :2], pts[LEFT["outer"], :2]
    across = left_eye - right_eye
    span = float(np.linalg.norm(across))
    if span < 1e-6:
        return None
    h = across / span
    v = np.array([-h[1], h[0]])
    off = pts[NOSE, :2] - (left_eye + right_eye) / 2
    return float(off @ h / span), float(off @ v / span)


def head_ray(pose, pts):
    """The nose tip and the end of the ray out of the face, in frame pixels: ((x, y), (x, y))."""
    out, span = pose[5], pose[6]
    nose = np.asarray(pts[NOSE][:2], float)
    flat = out[:2] / max(math.hypot(*out[:2]), 0.25)    # stays visible even when facing the camera head-on
    end = nose + RAY * span * flat * min(1.0, math.hypot(*out[:2]) * 3)
    return tuple(nose), tuple(end)


class IrisGaze:
    def __init__(self, mode="head"):
        import mediapipe as mp
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        self.mode = mode
        self.mesh = mp.solutions.face_mesh.FaceMesh(
            max_num_faces=1, refine_landmarks=True,
            min_detection_confidence=0.5, min_tracking_confidence=0.3)   # keep a turned or edge face
        self.pts = None          # landmarks of the last frame (None: no face), for the camera view
        self.opening = None      # how open the eyes were in the last frame (None: no face), for blinks
        self.pose = None         # head_pose() of the last frame, for the ray
        self.nose = None         # nose_offset() of the last frame: where the nose is from between the eyes
        self.size = (640, 480)

    def features(self, frame):
        """Features of the face in a BGR frame, or None (no face; in eyes mode also a blink)."""
        h, w = frame.shape[:2]
        self.size = (w, h)
        found = self.mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).multi_face_landmarks
        if not found:
            self.pts, self.opening, self.pose, self.nose = None, None, None, None
            return None
        marks = found[0].landmark
        self.pts = np.array([(p.x * w, p.y * h) for p in marks])
        self.opening = eye_opening(self.pts)
        self.pose = head_pose([(p.x * w, p.y * h, p.z * w) for p in marks], (w, h))
        self.nose = nose_offset(self.pts)
        head = None if self.pose is None else np.array(self.pose[:5])
        if self.mode == "head":
            return head                            # blinking doesn't move a head pointer
        eyes = features_from_landmarks(self.pts, (w, h))
        if self.mode == "eyes" or eyes is None or head is None:
            return eyes if self.mode == "eyes" else None
        return np.concatenate([eyes[:2], head[:2], eyes[2:3], head[2:]])   # both: eyes and head angles first

    def ray(self):
        """Nose tip and ray end in frame pixels for the last frame, or None."""
        return None if self.pose is None else head_ray(self.pose, self.pts)

    def close(self):
        self.mesh.close()


def in_view(point, frame_size, size):
    """A frame pixel -> where it is on camera_view's mirrored picture of the given size."""
    return int(size[0] - point[0] * size[0] / frame_size[0]), int(point[1] * size[1] / frame_size[1])


def camera_view(frame, pts, size, ok, ray=None, frame_size=(640, 480)):
    """The camera picture as a mirror, scaled to size (w, h), with the key points marked: eye corners,
    irises, nose tip, forehead and chin, and the ray out of the face (ray: from IrisGaze.ray()).
    ok: features were found this frame (green), else face lost or blinking (red)."""
    w, h = size
    if frame is None:
        view = np.zeros((h, w, 3), np.uint8)
        cv2.putText(view, "NO CAMERA", (10, h // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (80, 80, 255), 1, cv2.LINE_AA)
        return view
    fh, fw = frame.shape[:2]
    sx, sy = w / fw, h / fh
    view = cv2.flip(cv2.resize(frame, (w, h)), 1)
    color = (120, 220, 120) if ok else (80, 80, 255)
    if pts is not None:
        mark = lambda i: (int(w - pts[i][0] * sx), int(pts[i][1] * sy))     # noqa: E731  mirrored
        dot = max(2, w // 120)
        for eye in (RIGHT, LEFT):
            cv2.line(view, mark(eye["outer"]), mark(eye["inner"]), color, 1, cv2.LINE_AA)
            cv2.circle(view, mark(eye["iris"]), dot, (90, 200, 255), -1, cv2.LINE_AA)
        for i in (FOREHEAD, CHIN, 61, 291):                                  # forehead, chin, mouth corners
            cv2.circle(view, mark(i), dot, color, -1, cv2.LINE_AA)
        if ray is not None:                                                  # the ray out of the nose
            nose, end = in_view(ray[0], frame_size, size), in_view(ray[1], frame_size, size)
            cv2.arrowedLine(view, nose, end, (255, 230, 120), max(2, w // 160), cv2.LINE_AA, tipLength=0.2)
            cv2.circle(view, nose, dot + 2, (255, 230, 120), -1, cv2.LINE_AA)
    label = "face OK" if ok else ("blink / eyes not clear" if pts is not None else "NO FACE")
    cv2.rectangle(view, (0, h - 26), (w - 1, h - 1), (30, 24, 24), -1)
    cv2.rectangle(view, (0, 0), (w - 1, h - 1), color, 2)
    cv2.putText(view, label, (8, h - 9), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    return view


class Camera:
    """A camera on this computer (/dev/video0, 0), or a stream from another one (http://.../stream.mjpg, e.g. the
    laptop's webcam from tools/laptop_camera.py). status says how it is doing, for the screen."""

    def __new__(cls, source=0):
        if str(source).startswith(("http://", "https://")) and cls is Camera:
            return super().__new__(StreamCamera)
        return super().__new__(cls)

    def __init__(self, source=0):
        self.source = source
        source = int(source) if str(source).isdigit() else source
        self.cap = cv2.VideoCapture(source)
        if not self.cap.isOpened():
            raise RuntimeError(f"Can't open the camera {source!r}")
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.status = "connected"

    def read(self):
        ok, frame = self.cap.read()
        return frame if ok else None

    def close(self):
        self.cap.release()


class StreamCamera(Camera):
    """An MJPEG stream over HTTP, read in the background. read() gives the newest picture only (waiting a moment
    for a fresh one), so nothing piles up and the picture stays live; None when the stream is down. It keeps
    reconnecting, so the laptop can start streaming before or after the app."""

    STALE = 1.5          # seconds without a new picture: the stream counts as down

    def __init__(self, source):
        import threading
        self.source, self.status = source, f"waiting for {source}"
        self.frame, self.number, self.at = None, 0, 0.0
        self.fresh = threading.Condition()
        self.stopped = False
        self.thread = threading.Thread(target=self._run, name="eyereach-stream", daemon=True)
        self.thread.start()

    def _run(self):
        import time
        import urllib.request
        while not self.stopped:
            try:
                with urllib.request.urlopen(self.source, timeout=3) as stream:
                    buffer = b""
                    while not self.stopped:
                        chunk = stream.read(16384)
                        if not chunk:
                            break
                        buffer += chunk
                        end = buffer.rfind(b"\xff\xd9")               # the newest complete JPEG ...
                        start = buffer.rfind(b"\xff\xd8", 0, end) if end > 0 else -1
                        if start < 0:
                            buffer = buffer[-200000:]
                            continue
                        jpeg, buffer = buffer[start:end + 2], buffer[end + 2:]   # ... older ones are dropped
                        image = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
                        if image is not None:
                            with self.fresh:
                                self.frame, self.number, self.at = image, self.number + 1, time.monotonic()
                                self.status = "connected"
                                self.fresh.notify_all()
            except Exception as error:                              # not there yet, or it went away
                self.status = f"waiting for {self.source} ({type(error).__name__})"
            time.sleep(1.0)

    def read(self):
        import time
        with self.fresh:
            seen = self.number
            self.fresh.wait_for(lambda: self.number != seen, timeout=0.1)
            if self.frame is None or time.monotonic() - self.at > self.STALE:
                if self.frame is not None:
                    self.status = f"waiting for {self.source} (no pictures)"
                return None
            return self.frame

    def close(self):
        self.stopped = True


class GazeMap:
    """Features -> screen pixels: a ridge regression on a second-degree polynomial of the gaze
    numbers plus the head numbers. shortcut: no per-eye model or head-pose correction beyond
    that; upgrade if the calibration error stays above a few percent of the screen."""

    def __init__(self, mean, std, weights):
        self.mean, self.std, self.weights = map(np.asarray, (mean, std, weights))

    @staticmethod
    def expand(f):
        gx, gy = f[:, 0], f[:, 1]
        return np.column_stack([gx, gy, gx * gx, gy * gy, gx * gy, f[:, 2:]])

    def design(self, f):
        a = (self.expand(np.atleast_2d(np.asarray(f, float))) - self.mean) / self.std
        return np.column_stack([np.ones(len(a)), a])

    @classmethod
    def fit(cls, features, targets, ridge=1e-2):
        a = cls.expand(np.asarray(features, float))
        mean, std = a.mean(0), a.std(0) + 1e-9
        z = np.column_stack([np.ones(len(a)), (a - mean) / std])
        reg = ridge * np.eye(z.shape[1])
        reg[0, 0] = 0                                    # don't shrink the constant term
        return cls(mean, std, np.linalg.solve(z.T @ z + reg, z.T @ np.asarray(targets, float)))

    def predict(self, f):
        return self.design(f) @ self.weights

    def save(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps({k: getattr(self, k).tolist() for k in ("mean", "std", "weights")}))

    @classmethod
    def load(cls, path):
        return cls(**json.loads(Path(path).read_text()))


class OneEuro:
    """Steadies a moving point: strong smoothing while it is nearly still (a steady pointer for holding
    on a spot), less when it moves fast (so a deliberate move isn't laggy). The One Euro filter
    (Casiez, Roussel and Vogel, CHI 2012)."""

    def __init__(self, still_hz=0.4, speed_up=0.0015, dhz=1.0):
        self.still_hz, self.speed_up, self.dhz = still_hz, speed_up, dhz
        self.x = self.dx = self.t = None

    @staticmethod
    def alpha(hz, dt):
        tau = 1.0 / (2 * math.pi * hz)
        return 1.0 / (1.0 + tau / dt)

    def __call__(self, point, now=None):
        if point is None:
            return None if self.x is None else tuple(self.x)    # keep the last spot while the face is lost
        import time
        now = time.monotonic() if now is None else now
        p = np.asarray(point, float)
        if self.x is None:
            self.x, self.dx, self.t = p, np.zeros(2), now
            return tuple(p)
        dt = max(now - self.t, 1e-3)
        self.t = now
        a = self.alpha(self.dhz, dt)
        self.dx = a * (p - self.x) / dt + (1 - a) * self.dx
        hz = self.still_hz + self.speed_up * float(np.linalg.norm(self.dx))
        a = self.alpha(hz, dt)
        self.x = a * p + (1 - a) * self.x
        return tuple(self.x)

    def reset(self):
        self.x = self.dx = self.t = None


class Smooth:
    """Steadies the gaze point: the average of the last few frames, but a real jump is followed at once."""

    def __init__(self, frames=6, jump=150):
        self.recent, self.frames, self.jump = [], frames, jump

    def __call__(self, point):
        if point is None:
            return None
        if self.recent and np.hypot(*(np.asarray(point) - np.mean(self.recent, axis=0))) > self.jump:
            self.recent = []
        self.recent = (self.recent + [point])[-self.frames:]
        return tuple(np.mean(self.recent, axis=0))
