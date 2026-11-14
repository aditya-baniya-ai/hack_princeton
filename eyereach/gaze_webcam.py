"""Webcam eye tracking: iris position from MediaPipe Face Mesh, mapped to a point on the screen.

IrisGaze.features() turns a camera frame into a few numbers about where the eyes point.
GazeMap learns the mapping from those numbers to screen pixels from a short calibration
(look at the dots). Smooth steadies the result.
"""

import json
from pathlib import Path

import cv2
import numpy as np

# MediaPipe Face Mesh landmark numbers (refine_landmarks=True adds the iris points 468-477)
RIGHT = dict(outer=33, inner=133, upper=159, lower=145, iris=468)    # the person's right eye
LEFT = dict(outer=263, inner=362, upper=386, lower=374, iris=473)
NOSE = 1
BLINK = 0.12      # eye opening / eye width below this is a blink: the frame is skipped


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


class IrisGaze:
    def __init__(self):
        import mediapipe as mp
        self.mesh = mp.solutions.face_mesh.FaceMesh(
            max_num_faces=1, refine_landmarks=True,
            min_detection_confidence=0.5, min_tracking_confidence=0.5)

    def features(self, frame):
        """Features of the face in a BGR frame, or None (no face, or a blink)."""
        h, w = frame.shape[:2]
        found = self.mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).multi_face_landmarks
        if not found:
            return None
        pts = np.array([(p.x * w, p.y * h) for p in found[0].landmark])
        return features_from_landmarks(pts, (w, h))

    def close(self):
        self.mesh.close()


class Camera:
    def __init__(self, source=0):
        source = int(source) if str(source).isdigit() else source
        self.cap = cv2.VideoCapture(source)
        if not self.cap.isOpened():
            raise RuntimeError(f"Can't open the camera {source!r}")
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    def read(self):
        ok, frame = self.cap.read()
        return frame if ok else None

    def close(self):
        self.cap.release()


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
