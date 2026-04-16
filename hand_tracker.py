# =============================================================================
# hand_tracker.py — Thin wrapper around MediaPipe Hands
# =============================================================================
"""
Abstracts MediaPipe so the rest of the pipeline talks to a clean API.

Landmark indices (MediaPipe canonical hand model):
  0  WRIST
  1  THUMB_CMC      2  THUMB_MCP      3  THUMB_IP       4  THUMB_TIP
  5  INDEX_FINGER_MCP  6  INDEX_FINGER_PIP  7  INDEX_FINGER_DIP  8  INDEX_FINGER_TIP
  9  MIDDLE_FINGER_MCP 10 MIDDLE_FINGER_PIP 11 MIDDLE_FINGER_DIP 12 MIDDLE_FINGER_TIP
  13 RING_FINGER_MCP   14 RING_FINGER_PIP   15 RING_FINGER_DIP   16 RING_FINGER_TIP
  17 PINKY_MCP        18 PINKY_PIP         19 PINKY_DIP         20 PINKY_TIP
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np

from config import (
    MAX_NUM_HANDS,
    MIN_DETECTION_CONFIDENCE,
    MIN_TRACKING_CONFIDENCE,
    MODEL_COMPLEXITY,
    SHOW_LANDMARKS,
    COLOR_GREEN,
)


# ─────────────────────────────────────────────────────────────────────────────
# Named landmark indices for readable code
# ─────────────────────────────────────────────────────────────────────────────
class LM:
    WRIST            = 0
    THUMB_CMC        = 1;  THUMB_MCP   = 2;  THUMB_IP  = 3;  THUMB_TIP  = 4
    INDEX_MCP        = 5;  INDEX_PIP   = 6;  INDEX_DIP = 7;  INDEX_TIP  = 8
    MIDDLE_MCP       = 9;  MIDDLE_PIP  = 10; MIDDLE_DIP= 11; MIDDLE_TIP = 12
    RING_MCP         = 13; RING_PIP    = 14; RING_DIP  = 15; RING_TIP   = 16
    PINKY_MCP        = 17; PINKY_PIP   = 18; PINKY_DIP = 19; PINKY_TIP  = 20

    # Convenience groups
    TIPS   = [THUMB_TIP, INDEX_TIP, MIDDLE_TIP, RING_TIP, PINKY_TIP]
    PIPS   = [THUMB_IP,  INDEX_PIP, MIDDLE_PIP, RING_PIP, PINKY_PIP]
    MCPS   = [THUMB_MCP, INDEX_MCP, MIDDLE_MCP, RING_MCP, PINKY_MCP]


@dataclass
class HandResult:
    """Processed result for a single detected hand."""

    # Raw pixel-space coordinates — list of (x_px, y_px) for all 21 landmarks
    landmarks_px: List[Tuple[int, int]] = field(default_factory=list)

    # Normalised coordinates (0-1) — (x, y, z) tuples for all 21 landmarks
    landmarks_norm: List[Tuple[float, float, float]] = field(default_factory=list)

    # Handedness string: "Left" or "Right" (from camera perspective = mirrored)
    handedness: str = "Unknown"

    # Per-finger extension flags  [thumb, index, middle, ring, pinky]
    fingers_up: List[bool] = field(default_factory=lambda: [False] * 5)

    # Number of extended fingers
    fingers_count: int = 0

    def lm(self, idx: int) -> Tuple[int, int]:
        """Return pixel-space (x, y) for landmark *idx*."""
        return self.landmarks_px[idx]

    def lm_norm(self, idx: int) -> Tuple[float, float, float]:
        """Return normalised (x, y, z) for landmark *idx*."""
        return self.landmarks_norm[idx]


class HandTracker:
    """
    Wraps MediaPipe Hands into a simple frame-by-frame API.

    Usage:
        tracker = HandTracker()
        while True:
            frame = ...  # BGR from OpenCV
            results = tracker.process(frame)  # list[HandResult]
            tracker.draw(frame, results)
    """

    def __init__(self):
        self._mp_hands = mp.solutions.hands
        self._mp_draw  = mp.solutions.drawing_utils
        self._mp_styles= mp.solutions.drawing_styles

        self.hands = self._mp_hands.Hands(
            static_image_mode       = False,
            max_num_hands           = MAX_NUM_HANDS,
            model_complexity        = MODEL_COMPLEXITY,
            min_detection_confidence= MIN_DETECTION_CONFIDENCE,
            min_tracking_confidence = MIN_TRACKING_CONFIDENCE,
        )

        # Custom landmark style — subtle dots instead of the default big blobs
        self._dot_spec = self._mp_draw.DrawingSpec(
            color=COLOR_GREEN, thickness=1, circle_radius=3
        )
        self._conn_spec = self._mp_draw.DrawingSpec(
            color=(255, 255, 255), thickness=1
        )

    # ── Public API ───────────────────────────────────────────────────────────

    def process(self, frame_bgr: np.ndarray) -> List[HandResult]:
        """
        Detect hands in *frame_bgr* and return a list of HandResult objects
        (one per detected hand, up to MAX_NUM_HANDS).
        """
        h, w = frame_bgr.shape[:2]

        # MediaPipe works on RGB
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        mp_result = self.hands.process(rgb)

        results: List[HandResult] = []
        if not mp_result.multi_hand_landmarks:
            return results

        for hand_lms, handedness in zip(
            mp_result.multi_hand_landmarks,
            mp_result.multi_handedness,
        ):
            hr = HandResult()
            hr.handedness = handedness.classification[0].label

            # Convert normalised → pixel coordinates
            for lm in hand_lms.landmark:
                hr.landmarks_px.append((int(lm.x * w), int(lm.y * h)))
                hr.landmarks_norm.append((lm.x, lm.y, lm.z))

            hr.fingers_up    = self._fingers_extended(hr)
            hr.fingers_count = sum(hr.fingers_up)
            results.append(hr)

        return results

    def draw(self, frame: np.ndarray, results: List[HandResult]):
        """Overlay skeleton landmarks on *frame* (in-place) if enabled."""
        if not SHOW_LANDMARKS or not results:
            return

        # Re-run MediaPipe drawing on the frame
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        mp_result = self.hands.process(rgb)
        if not mp_result.multi_hand_landmarks:
            return
        for hand_lms in mp_result.multi_hand_landmarks:
            self._mp_draw.draw_landmarks(
                frame,
                hand_lms,
                self._mp_hands.HAND_CONNECTIONS,
                self._dot_spec,
                self._conn_spec,
            )

    def close(self):
        self.hands.close()

    # ── Private helpers ───────────────────────────────────────────────────────

    def _fingers_extended(self, hr: HandResult) -> List[bool]:
        """
        Determine which fingers are extended.

        Strategy:
          • Thumb: compare TIP x vs IP x (accounts for left/right hand)
          • Other fingers: TIP y < PIP y (tip is above the PIP joint)
        """
        lm = hr.landmarks_px
        up = [False] * 5

        # Thumb — uses x-axis comparison; flip for right hand
        if hr.handedness == "Right":
            up[0] = lm[LM.THUMB_TIP][0] < lm[LM.THUMB_IP][0]
        else:
            up[0] = lm[LM.THUMB_TIP][0] > lm[LM.THUMB_IP][0]

        # Index → Pinky — uses y-axis (smaller y = higher on screen)
        finger_pairs = [
            (LM.INDEX_TIP,  LM.INDEX_PIP),
            (LM.MIDDLE_TIP, LM.MIDDLE_PIP),
            (LM.RING_TIP,   LM.RING_PIP),
            (LM.PINKY_TIP,  LM.PINKY_PIP),
        ]
        for i, (tip, pip) in enumerate(finger_pairs, start=1):
            up[i] = lm[tip][1] < lm[pip][1]

        return up
