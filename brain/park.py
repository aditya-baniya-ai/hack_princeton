"""Saved robot-frame drop destination and a non-blocking return/lower/release/lift job."""
import json
import math
from pathlib import Path

import arm
import arm_link
from fetch import FetchJob

PARK_FILE = Path.home() / '.eyereach' / 'park.json'


def position(x, y, drop_mm=50.0, travel_mm=120.0):
    """Validate mm coordinates in the arm's frame, not pixels or table-camera fractions."""
    x, y, drop_mm, travel_mm = map(float, (x, y, drop_mm, travel_mm))
    if not all(map(math.isfinite, (x, y, drop_mm, travel_mm))):
        raise ValueError('park coordinates must be finite numbers')
    u, v = arm_link.point_to_area(x, y)
    if not (0 <= u <= 1 and 0 <= v <= 1):
        raise ValueError('park position is outside the arm workspace')
    if not 30 <= drop_mm <= travel_mm - 10:
        raise ValueError('park drop height must be at least 30 mm and 10 mm below travel height')
    down, why = arm.solve(x, y, drop_mm)
    if down is None:
        raise ValueError(why or 'park drop height is unreachable')
    # Use the same travel-height solver as actual steering.
    probe = arm_link.VirtualArm()
    probe.hover = travel_mm
    up = probe.hover_joints(x, y)
    if up is None or arm.path_problem(up, down):
        raise ValueError('park lowering path is unreachable')
    return dict(x=x, y=y, drop_mm=drop_mm)


def load(path=None, travel_mm=120.0):
    path = PARK_FILE if path is None else Path(path)
    if not path.exists():
        return None
    try:
        saved = json.loads(path.read_text())
        return position(saved['x'], saved['y'], saved['drop_mm'], travel_mm)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise ValueError(f'cannot load park position: {exc}') from exc


def configure(enabled=None, coordinates=None, drop_mm=None, travel_mm=120.0, path=None):
    """Automatically use a saved destination unless park was explicitly disabled."""
    if enabled is False:
        if coordinates is not None:
            raise ValueError('--no-park cannot be combined with --park-position')
        return None
    if coordinates is not None:
        saved = position(*coordinates, 50.0 if drop_mm is None else drop_mm, travel_mm)
    else:
        saved = load(path, travel_mm)
        if saved is not None and drop_mm is not None:
            saved = position(saved['x'], saved['y'], drop_mm, travel_mm)
    if enabled is True and saved is None:
        raise ValueError('no saved park position: move above the drop spot and press P, or use --park-position X Y')
    return saved


def save(saved, path=None, travel_mm=120.0):
    saved = position(saved['x'], saved['y'], saved['drop_mm'], travel_mm)
    path = PARK_FILE if path is None else Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(saved))
    temporary.replace(path)
    return saved


class ParkJob(FetchJob):
    """Carry an already grabbed object to park, lower, release and lift; STOP remains live."""
    GLIDE_SECONDS = 120.0
    ACTION_SECONDS = 60.0

    def __init__(self, robot, saved, speeds=None, pickup=None):
        saved = position(saved['x'], saved['y'], saved['drop_mm'], robot.hover)
        x, y = saved['x'], saved['y']
        drop = arm_link.point_to_area(x, y)
        if pickup is None:
            super().__init__(robot, x, y, drop_mm=saved['drop_mm'], drop=drop)
            self.plan = [('glide', (x, y)), ('down', 'release')]
        else:
            px, py, grab_mm = pickup
            super().__init__(robot, px, py, grab_mm=grab_mm, drop=drop, drop_mm=saved['drop_mm'])
        self.speeds = speeds
        robot.stage = 'Going to park'

    def step(self):
        if self.speeds is None:
            return super().step()
        robot = self.robot
        glide, speed = robot.glide, robot.speed
        limits = self.speeds[2:] if robot.closed else self.speeds[:2]
        try:
            robot.glide, robot.speed = min(glide, limits[0]), min(speed, limits[1])
            return super().step()
        finally:
            robot.glide, robot.speed = glide, speed
