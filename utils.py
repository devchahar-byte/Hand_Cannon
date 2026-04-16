# =============================================================================
# utils.py — Geometry helpers for the hand-gesture pipeline
# =============================================================================
from __future__ import annotations

import math
import time
from collections import deque
from typing import Tuple, List

import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Landmark type alias  (x, y, z) — all floats 0-1 from MediaPipe
# ─────────────────────────────────────────────────────────────────────────────
Point2D = Tuple[float, float]
Point3D = Tuple[float, float, float]


# ─────────────────────────────────────────────────────────────────────────────
# Distance / angle maths
# ─────────────────────────────────────────────────────────────────────────────
def euclidean(p1: Point2D, p2: Point2D) -> float:
    """Return Euclidean distance between two 2-D points."""
    return math.hypot(p2[0] - p1[0], p2[1] - p1[1])


def euclidean3d(p1: Point3D, p2: Point3D) -> float:
    """Return Euclidean distance in 3-D landmark space."""
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(p1, p2)))


def angle_between(a: Point2D, b: Point2D, c: Point2D) -> float:
    """
    Compute the angle (degrees) at vertex *b* formed by segments b→a and b→c.
    Useful for knuckle-bend detection.
    """
    ba = (a[0] - b[0], a[1] - b[1])
    bc = (c[0] - b[0], c[1] - b[1])
    dot = ba[0] * bc[0] + ba[1] * bc[1]
    mag_ba = math.hypot(*ba)
    mag_bc = math.hypot(*bc)
    if mag_ba == 0 or mag_bc == 0:
        return 0.0
    cos_angle = max(-1.0, min(1.0, dot / (mag_ba * mag_bc)))
    return math.degrees(math.acos(cos_angle))


def normalised_dist(p1: Point2D, p2: Point2D,
                    frame_w: int, frame_h: int) -> float:
    """
    Euclidean distance between two pixel-space points, normalised by the
    frame diagonal so that thresholds are resolution-independent.
    """
    diag = math.hypot(frame_w, frame_h)
    return euclidean(p1, p2) / diag


def midpoint(p1: Point2D, p2: Point2D) -> Point2D:
    return ((p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2)


# ─────────────────────────────────────────────────────────────────────────────
# Coordinate mapping
# ─────────────────────────────────────────────────────────────────────────────
def remap(value: float,
          in_min: float, in_max: float,
          out_min: float, out_max: float,
          clamp: bool = True) -> float:
    """Linearly map *value* from [in_min, in_max] to [out_min, out_max]."""
    if in_max == in_min:
        return out_min
    mapped = (value - in_min) / (in_max - in_min) * (out_max - out_min) + out_min
    if clamp:
        mapped = max(out_min, min(out_max, mapped))
    return mapped


# ─────────────────────────────────────────────────────────────────────────────
# Smoothing
# ─────────────────────────────────────────────────────────────────────────────
class EMAFilter:
    """
    Exponential Moving Average filter for (x, y) coordinates.
    Higher *alpha* = faster / less smooth.  Lower = smoother / more lag.
    """

    def __init__(self, alpha: float = 0.25):
        self._alpha = alpha
        self._x: float | None = None
        self._y: float | None = None

    def update(self, x: float, y: float) -> Tuple[float, float]:
        if self._x is None:
            self._x, self._y = x, y
        else:
            self._x = self._alpha * x + (1 - self._alpha) * self._x
            self._y = self._alpha * y + (1 - self._alpha) * self._y
        return self._x, self._y

    def reset(self):
        self._x = self._y = None


class MovingAverageFilter:
    """Simple sliding-window moving average for (x, y) pairs."""

    def __init__(self, window: int = 7):
        self._buf_x: deque = deque(maxlen=window)
        self._buf_y: deque = deque(maxlen=window)

    def update(self, x: float, y: float) -> Tuple[float, float]:
        self._buf_x.append(x)
        self._buf_y.append(y)
        return (sum(self._buf_x) / len(self._buf_x),
                sum(self._buf_y) / len(self._buf_y))

    def reset(self):
        self._buf_x.clear()
        self._buf_y.clear()


# ─────────────────────────────────────────────────────────────────────────────
# FPS counter
# ─────────────────────────────────────────────────────────────────────────────
class FPSCounter:
    """Rolling-window FPS estimator."""

    def __init__(self, window: int = 30):
        self._times: deque = deque(maxlen=window)

    def tick(self) -> float:
        self._times.append(time.perf_counter())
        if len(self._times) < 2:
            return 0.0
        elapsed = self._times[-1] - self._times[0]
        return (len(self._times) - 1) / elapsed if elapsed > 0 else 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Velocity tracker (for swipe detection)
# ─────────────────────────────────────────────────────────────────────────────
class VelocityTracker:
    """
    Tracks the velocity (px/s) of a 2-D point across frames.
    Call .update(x, y) each frame; read .vx / .vy.
    """

    def __init__(self, history: int = 5):
        self._hist: deque = deque(maxlen=history)   # (t, x, y)

    def update(self, x: float, y: float):
        self._hist.append((time.perf_counter(), x, y))

    @property
    def vx(self) -> float:
        return self._velocity()[0]

    @property
    def vy(self) -> float:
        return self._velocity()[1]

    def _velocity(self) -> Tuple[float, float]:
        if len(self._hist) < 2:
            return 0.0, 0.0
        t0, x0, y0 = self._hist[0]
        t1, x1, y1 = self._hist[-1]
        dt = t1 - t0
        if dt < 1e-6:
            return 0.0, 0.0
        return (x1 - x0) / dt, (y1 - y0) / dt

    def reset(self):
        self._hist.clear()


# ─────────────────────────────────────────────────────────────────────────────
# Cooldown timer
# ─────────────────────────────────────────────────────────────────────────────
class Cooldown:
    """Simple one-shot cooldown so actions don't fire every frame."""

    def __init__(self, seconds: float):
        self._period = seconds
        self._last = 0.0

    def ready(self) -> bool:
        return (time.perf_counter() - self._last) >= self._period

    def reset(self):
        self._last = time.perf_counter()


# ─────────────────────────────────────────────────────────────────────────────
# Skin-colour mask (HSV fallback)
# ─────────────────────────────────────────────────────────────────────────────
def skin_mask(frame_bgr: np.ndarray,
              lower=(0, 20, 70), upper=(25, 255, 255)) -> np.ndarray:
    """
    Return a binary mask of skin-coloured pixels using HSV thresholding.
    Used as a fallback visualisation when MediaPipe loses tracking.
    """
    import cv2
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv,
                       np.array(lower, dtype=np.uint8),
                       np.array(upper, dtype=np.uint8))
    # Morphological cleanup
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    return mask
