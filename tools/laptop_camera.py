"""Run this ON THE LAPTOP: it streams the laptop's webcam to the Jetson, which uses it to watch your face.

    pip install opencv-python          (once)
    python3 laptop_camera.py           then start the app on the Jetson: launch/eyereach.sh

The Jetson reads http://<this laptop>:8080/stream.mjpg (over the USB cable the laptop is 192.168.55.100;
launch/eyereach.sh's FACE_CAMERA setting). Only Python and OpenCV are needed, on macOS, Windows or Linux.
On a Mac, the first run asks for camera permission for the Terminal: allow it, then run it again.

    --camera 1          another webcam
    --port 8080         another port
    --picture face.jpg  serve a still picture instead of a camera (for tests)
    --size 640x480      the picture size sent (smaller is quicker; the face tracker needs little)

Stop with Ctrl+C.
"""

import argparse
import socket
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2

BOUNDARY = b"eyereachframe"


class Latest:
    """The newest JPEG, shared by the camera thread and every viewer."""

    def __init__(self):
        self.jpeg, self.number = None, 0
        self.changed = threading.Condition()

    def put(self, jpeg):
        with self.changed:
            self.jpeg, self.number = jpeg, self.number + 1
            self.changed.notify_all()

    def wait_newer(self, number, timeout=2.0):
        with self.changed:
            self.changed.wait_for(lambda: self.number != number, timeout)
            return self.jpeg, self.number


def capture(latest, args):
    width, height = (int(x) for x in args.size.split("x"))
    if args.picture:
        image = cv2.imread(args.picture)
        if image is None:
            sys.exit(f"Can't read {args.picture}")
        jpeg = cv2.imencode(".jpg", cv2.resize(image, (width, height)), [cv2.IMWRITE_JPEG_QUALITY, 80])[1].tobytes()
        while True:
            latest.put(jpeg)
            time.sleep(1 / args.fps)
    source = int(args.camera) if str(args.camera).isdigit() else args.camera
    cam = cv2.VideoCapture(source)
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    if not cam.isOpened():
        sys.exit(f"Can't open camera {args.camera} (on a Mac: allow camera access for the Terminal, then retry)")
    failures = 0
    while True:
        ok, frame = cam.read()
        if not ok:
            failures += 1
            if failures > 50:
                sys.exit("The camera stopped giving pictures.")
            time.sleep(0.05)
            continue
        failures = 0
        if frame.shape[1] != width:
            frame = cv2.resize(frame, (width, height))
        latest.put(cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])[1].tobytes())


def handler_for(latest):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):               # quiet: one line when someone connects is enough
            pass

        def do_GET(self):
            if self.path.startswith("/snapshot.jpg"):
                jpeg, _ = latest.wait_newer(-1)
                self.send_response(200)
                self.send_header("Content-Type", "image/jpeg")
                self.send_header("Content-Length", str(len(jpeg)))
                self.end_headers()
                self.wfile.write(jpeg)
                return
            if not self.path.startswith("/stream.mjpg"):
                self.send_error(404, "Try /stream.mjpg")
                return
            print(f"The Jetson ({self.client_address[0]}) is watching.", flush=True)
            self.send_response(200)
            self.send_header("Content-Type", f"multipart/x-mixed-replace; boundary={BOUNDARY.decode()}")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            number = 0
            try:
                while True:
                    jpeg, number = latest.wait_newer(number)
                    if jpeg is None:
                        continue
                    self.wfile.write(b"--" + BOUNDARY + b"\r\nContent-Type: image/jpeg\r\nContent-Length: "
                                     + str(len(jpeg)).encode() + b"\r\n\r\n" + jpeg + b"\r\n")
            except (BrokenPipeError, ConnectionResetError):
                print("The Jetson stopped watching.", flush=True)

    return Handler


def addresses():
    """This computer's addresses, to tell the person (the USB one is usually 192.168.55.100)."""
    found = set()
    try:
        for info in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
            found.add(info[4][0])
    except OSError:
        pass
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("192.168.55.1", 9))            # the Jetson over USB: picks the right interface, sends nothing
        found.add(s.getsockname()[0])
        s.close()
    except OSError:
        pass
    return sorted(a for a in found if not a.startswith("127."))


def main():
    parser = argparse.ArgumentParser(description="Stream this laptop's webcam to the Jetson (MJPEG over HTTP).")
    parser.add_argument("--camera", default="0", help="camera number (default 0, the built-in one)")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--size", default="640x480")
    parser.add_argument("--fps", type=float, default=20.0, help="--picture only: how often it's sent")
    parser.add_argument("--picture", help="serve this still picture instead of a camera (tests)")
    args = parser.parse_args()
    latest = Latest()
    threading.Thread(target=capture, args=(latest, args), daemon=True).start()
    server = ThreadingHTTPServer(("0.0.0.0", args.port), handler_for(latest))
    server.daemon_threads = True
    print(f"Streaming the {'picture' if args.picture else 'camera'} on port {args.port}.")
    for a in addresses():
        print(f"  http://{a}:{args.port}/stream.mjpg")
    print("Now start the app on the Jetson. Ctrl+C stops.", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
