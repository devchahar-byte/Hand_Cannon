# =============================================================================
# hud.py — Heads-up-display overlay drawn on each camera frame
# =============================================================================
"""
Draws all on-screen UI chrome: mode banner, gesture label, FPS counter,
finger status dots, and the virtual canvas composite.
"""
from __future__ import annotations

from typing import List, Optional

import cv2
import numpy as np

from config import *
from gesture_controller import Gesture, Mode
from hand_tracker import HandResult
from utils import FPSCounter


class HUD:
    def __init__(self, frame_w: int, frame_h: int):
        self.fw = frame_w
        self.fh = frame_h
        self._fps = FPSCounter(window=30)

    def render(self,
               frame: np.ndarray,
               mode: Mode,
               gesture: Gesture,
               hands: List[HandResult],
               canvas: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Composite all HUD elements onto *frame* and return the result.
        Does NOT modify the input frame in-place if possible.
        """
        self._fps.tick()
        out = frame.copy()

        # ── Virtual canvas overlay ────────────────────────────────────────────
        if canvas is not None and mode == Mode.CANVAS:
            mask = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
            _, mask = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)
            canvas_visible = cv2.addWeighted(out, 1.0, canvas, CANVAS_OPACITY * 3, 0)
            out = np.where(mask[:, :, np.newaxis] > 0, canvas_visible, out)

            # Show current colour swatch in corner
            color = CANVAS_COLORS[0]   # Controller tracks index; pass it in if needed
            cv2.rectangle(out, (self.fw - 60, self.fh - 60),
                          (self.fw - 10, self.fh - 10), color, -1)
            cv2.rectangle(out, (self.fw - 60, self.fh - 60),
                          (self.fw - 10, self.fh - 10), COLOR_WHITE, 2)

        # ── Mode banner (top bar) ─────────────────────────────────────────────
        if SHOW_MODE_BANNER:
            overlay = out.copy()
            cv2.rectangle(overlay, (0, 0), (self.fw, 48), COLOR_BG, -1)
            cv2.addWeighted(overlay, 0.75, out, 0.25, 0, out)
            cv2.putText(out, mode.label(),
                        (16, 34), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                        mode.color(), 2, cv2.LINE_AA)

        # ── Gesture label ────────────────────────────────────────────────────
        if SHOW_GESTURE_LABEL and gesture != Gesture.NONE:
            label = gesture.value
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
            px = self.fw // 2 - tw // 2
            # Semi-transparent pill background
            overlay = out.copy()
            cv2.rectangle(overlay, (px - 12, self.fh - 56),
                          (px + tw + 12, self.fh - 20), COLOR_BG, -1)
            cv2.addWeighted(overlay, 0.65, out, 0.35, 0, out)
            cv2.putText(out, label,
                        (px, self.fh - 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                        COLOR_WHITE, 2, cv2.LINE_AA)

        # ── FPS counter ──────────────────────────────────────────────────────
        if SHOW_FPS:
            fps_val = self._fps.tick()
            color = COLOR_GREEN if fps_val >= 25 else COLOR_YELLOW if fps_val >= 15 else COLOR_RED
            cv2.putText(out, f"FPS: {fps_val:.0f}",
                        (self.fw - 130, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.75,
                        color, 2, cv2.LINE_AA)

        # ── Finger status dots ───────────────────────────────────────────────
        if hands:
            self._draw_finger_dots(out, hands[0].fingers_up)

        # ── Active region rectangle (mouse mode only) ─────────────────────────
        if mode == Mode.MOUSE:
            cv2.rectangle(out,
                          (MOUSE_MARGIN_X, MOUSE_MARGIN_Y),
                          (self.fw - MOUSE_MARGIN_X, self.fh - MOUSE_MARGIN_Y),
                          COLOR_GRAY, 1)

        # ── Controls cheatsheet ──────────────────────────────────────────────
        self._draw_cheatsheet(out, mode)

        return out

    # ── Private helpers ───────────────────────────────────────────────────────

    def _draw_finger_dots(self, frame: np.ndarray, fingers: List[bool]):
        """Five small circles at top-left showing which fingers are extended."""
        labels = ["T", "I", "M", "R", "P"]
        for i, (up, label) in enumerate(zip(fingers, labels)):
            cx = 20 + i * 36
            cy = 72
            color = COLOR_GREEN if up else (60, 60, 60)
            cv2.circle(frame, (cx, cy), 14, color, -1)
            cv2.circle(frame, (cx, cy), 14, COLOR_WHITE, 1)
            cv2.putText(frame, label, (cx - 7, cy + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLOR_WHITE, 1, cv2.LINE_AA)

    def _draw_cheatsheet(self, frame: np.ndarray, mode: Mode):
        """Tiny gesture reference in the bottom-right corner."""
        if mode == Mode.MOUSE:
            lines = [
                "1 finger  → Move cursor",
                "Pinch     → Click/Drag",
                "Mid+Thumb → Scroll",
                "V sign    → Right Click",
                "Palm      → Freeze",
            ]
        elif mode == Mode.MEDIA:
            lines = [
                "Swipe →   → Next track",
                "Swipe ←   → Prev track",
                "Palm      → Play/Pause",
                "Thumb ↑   → Vol Up",
                "Thumb ↓   → Vol Down",
            ]
        else:
            lines = [
                "1 finger  → Draw",
                "2 fingers → Erase",
                "Pinch     → Next colour",
                "Palm      → Lift pen",
                "C         → Clear canvas",
                "S         → Save canvas",
            ]

        x0 = self.fw - 240
        y0 = 65
        for i, ln in enumerate(lines):
            cv2.putText(frame, ln, (x0, y0 + i * 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, COLOR_GRAY, 1, cv2.LINE_AA)
