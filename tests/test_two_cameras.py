"""Two cameras: the laptop's webcam streamed over the network (the face) and the robot camera on the Jetson."""

import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from eyereach.gaze_webcam import Camera, StreamCamera  # noqa: E402


def free_port():
    import socket
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class StreamCameraTest(unittest.TestCase):
    def test_waits_for_the_laptop_then_streams_then_notices_it_stopped(self):
        port = free_port()
        with tempfile.TemporaryDirectory() as d:
            picture = Path(d) / "face.jpg"
            cv2.imwrite(str(picture), np.full((240, 320, 3), 120, np.uint8))
            cam = Camera(f"http://127.0.0.1:{port}/stream.mjpg")       # before the laptop streams
            self.assertIsInstance(cam, StreamCamera)
            self.assertIsNone(cam.read())
            self.assertIn("waiting", cam.status)
            laptop = subprocess.Popen([sys.executable, str(ROOT / "tools" / "laptop_camera.py"), "--picture",
                                       str(picture), "--port", str(port), "--size", "320x240"],
                                      stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            try:
                frame, end = None, time.monotonic() + 8
                while frame is None and time.monotonic() < end:
                    frame = cam.read()
                self.assertIsNotNone(frame)                               # it connected by itself
                self.assertEqual(frame.shape, (240, 320, 3))
                self.assertEqual(cam.status, "connected")
                fresh = sum(cam.read() is not None for _ in range(20))
                self.assertGreater(fresh, 15)
            finally:
                laptop.terminate()
                laptop.wait(5)
            time.sleep(StreamCamera.STALE + 0.5)
            self.assertIsNone(cam.read())                                 # the laptop stopped: no stale picture
            cam.close()

    def test_a_local_camera_number_is_not_a_stream(self):
        self.assertFalse(issubclass(Camera, StreamCamera))


class LayoutTest(unittest.TestCase):
    def test_robot_face_map_buttons_and_volume_dont_overlap(self):
        from eyereach import run_webcam as r
        r.MAP["away_up"] = True
        try:
            for size in ((1800, 1130), (1280, 720), (1920, 1080)):
                r.set_size(*size)
                boxes = {"robot": (lambda a: (a[0], a[1], a[0] + a[2], a[1] + a[3]))(r.robot_rect()),
                         "face": (lambda a: (a[0], a[1], a[0] + a[2], a[1] + a[3]))(r.face_rect()),
                         "map": r.PLAY, "volume": r.volume_area()}
                for i, b in enumerate(r.button_rects(4)):
                    boxes[f"button{i}"] = b
                names = list(boxes)
                for i, a in enumerate(names):
                    for b in names[i + 1:]:
                        x0, y0, x1, y1 = boxes[a]
                        u0, v0, u1, v1 = boxes[b]
                        overlap = min(x1, u1) > max(x0, u0) and min(y1, v1) > max(y0, v0)
                        self.assertFalse(overlap, f"{a} and {b} overlap at {size}")
                    self.assertLessEqual(boxes[a][2], size[0])
                    self.assertLessEqual(boxes[a][3], size[1])
        finally:
            r.MAP["away_up"] = False
            r.set_size(1280, 720)


class SharedPictureTest(unittest.TestCase):
    def test_the_agent_reads_the_picture_the_app_keeps_fresh(self):
        try:
            from agent import task_agent
        except ImportError as error:
            self.skipTest(f"the agent's packages are in ~/eyereach_agent_venv ({error})")
        with tempfile.TemporaryDirectory() as d:
            shared = Path(d) / "table_camera.jpg"
            cv2.imwrite(str(shared), np.full((480, 640, 3), 200, np.uint8))
            seen = []
            original = task_agent.Robot.from_image
            task_agent.Robot.from_image = classmethod(lambda cls, image, **kw: seen.append(image.shape) or "ok")
            try:
                self.assertEqual(task_agent.Robot.from_camera(str(shared)), "ok")
            finally:
                task_agent.Robot.from_image = original
            self.assertEqual(seen, [(480, 640, 3)])


if __name__ == "__main__":
    unittest.main()
