# =============================================================================
# gesture_controller.py — Gesture classifier + system action executor
# =============================================================================
"""
Three operating modes, switchable at runtime with keyboard keys 1/2/3:

  Mode 1 – MOUSE CONTROL
    • Index finger up only    → move cursor
    • Index+Thumb pinch       → left click / drag
    • Middle+Thumb pinch      → scroll (move hand up/down)
    • Two fingers (V sign)    → right click
    • Open palm               → freeze cursor (pause mode)

  Mode 2 – MEDIA CONTROLLER
    • Swipe right             → next track
    • Swipe left              → prev track
    • Open palm               → play / pause
    • Thumbs up               → volume up
    • Thumbs down             → volume down

  Mode 3 – VIRTUAL CANVAS
    • Index up, others down   → draw
    • Index + middle up       → erase
    • Pinch (thumb+index)     → cycle colour
    • Open palm               → lift pen (stop drawing)
"""
from __future__ import annotations

import enum
import time
from typing import List, Optional, Tuple

import cv2
import numpy as np
import pyautogui

from config import *
from hand_tracker import HandResult, LM
from utils import (
    Cooldown, EMAFilter, MovingAverageFilter,
    VelocityTracker, normalised_dist, remap, midpoint, euclidean,
)

# Silence PyAutoGUI's built-in fail-safe pause for lower latency
pyautogui.PAUSE = 0.0
pyautogui.FAILSAFE = True


# ─────────────────────────────────────────────────────────────────────────────
# Mode enumeration
# ─────────────────────────────────────────────────────────────────────────────
class Mode(enum.Enum):
    MOUSE  = 1
    MEDIA  = 2
    CANVAS = 3

    def label(self) -> str:
        labels = {Mode.MOUSE: "🖱  Mouse Control", Mode.MEDIA: "🎵  Media Control", Mode.CANVAS: "🎨  Virtual Canvas"}
        return labels[self]

    def color(self):
        colors = {Mode.MOUSE: COLOR_BLUE, Mode.MEDIA: COLOR_GREEN, Mode.CANVAS: COLOR_YELLOW}
        return colors[self]


# ─────────────────────────────────────────────────────────────────────────────
# Gesture names
# ─────────────────────────────────────────────────────────────────────────────
class Gesture(enum.Enum):
    NONE         = "None"
    POINTING     = "Pointing (Move)"
    PINCH        = "Pinch (Click)"
    PINCH_MIDDLE = "Scroll Pinch"
    OPEN_PALM    = "Open Palm"
    VICTORY      = "Victory (Right Click)"
    SWIPE_LEFT   = "Swipe Left"
    SWIPE_RIGHT  = "Swipe Right"
    SWIPE_UP     = "Swipe Up"
    SWIPE_DOWN   = "Swipe Down"
    THUMBS_UP    = "Thumbs Up"
    THUMBS_DOWN  = "Thumbs Down"
    DRAW         = "Drawing"
    ERASE        = "Erasing"


# ─────────────────────────────────────────────────────────────────────────────
# Main controller class
# ─────────────────────────────────────────────────────────────────────────────
class GestureController:

    def __init__(self, frame_w: int, frame_h: int):
        self.frame_w = frame_w
        self.frame_h = frame_h
        self.screen_w, self.screen_h = pyautogui.size()

        self.mode: Mode = Mode.MOUSE

        # Smoothing filters for cursor movement
        self._cursor_filter = MovingAverageFilter(window=MOUSE_SMOOTHING)

        # Velocity tracker for swipe detection
        self._vel_tracker = VelocityTracker(history=8)

        # State tracking
        self._pinch_start_time: Optional[float] = None
        self._last_pinch_time: float = 0.0
        self._dragging: bool = False
        self._scroll_ref_y: Optional[int] = None

        # Swipe state
        self._swipe_start_x: Optional[float] = None
        self._swipe_start_y: Optional[float] = None
        self._swipe_started_at: Optional[float] = None

        # Cooldowns
        self._click_cooldown   = Cooldown(0.35)
        self._scroll_cooldown  = Cooldown(0.04)
        self._media_cooldown   = Cooldown(MEDIA_SWIPE_COOLDOWN_S)
        self._rclick_cooldown  = Cooldown(0.5)

        # Canvas state
        self._canvas: Optional[np.ndarray] = None
        self._prev_draw_pt: Optional[Tuple[int, int]] = None
        self._canvas_color_idx: int = 0
        self._color_cycle_cooldown = Cooldown(0.5)

        # Displayed gesture name
        self.current_gesture: Gesture = Gesture.NONE
        self.gesture_history: list = []

    # ── Initialise canvas ─────────────────────────────────────────────────────
    def init_canvas(self, h: int, w: int):
        if self._canvas is None:
            self._canvas = np.zeros((h, w, 3), dtype=np.uint8)

    def get_canvas(self) -> Optional[np.ndarray]:
        return self._canvas

    def clear_canvas(self):
        if self._canvas is not None:
            self._canvas[:] = 0
        self._prev_draw_pt = None

    def save_canvas(self, path: str = "canvas_output.png"):
        if self._canvas is not None:
            cv2.imwrite(path, self._canvas)
            return True
        return False

    # ── Main entry point ─────────────────────────────────────────────────────
    def process(self, hands: List[HandResult]) -> Gesture:
        """
        Called once per frame with the list of detected HandResult objects.
        Classifies the gesture and executes the appropriate system action.
        Returns the detected Gesture enum for display purposes.
        """
        if not hands:
            self._release_drag()
            self._scroll_ref_y = None
            self._prev_draw_pt = None
            self._swipe_start_x = None
            self.current_gesture = Gesture.NONE
            return Gesture.NONE

        hand = hands[0]   # Use first (and usually only) detected hand

        if self.mode == Mode.MOUSE:
            gesture = self._handle_mouse(hand)
        elif self.mode == Mode.MEDIA:
            gesture = self._handle_media(hand)
        else:
            gesture = self._handle_canvas(hand)

        self.current_gesture = gesture
        return gesture

    # ─────────────────────────────────────────────────────────────────────────
    # MODE 1 – MOUSE CONTROL
    # ─────────────────────────────────────────────────────────────────────────
    def _handle_mouse(self, hand: HandResult) -> Gesture:
        lm   = hand.landmarks_px
        norm = hand.landmarks_norm
        fup  = hand.fingers_up          # [thumb, idx, mid, ring, pinky]
        w, h = self.frame_w, self.frame_h

        # ── Landmark pixel positions ─────────────────────────────────────────
        idx_tip  = lm[LM.INDEX_TIP]
        mid_tip  = lm[LM.MIDDLE_TIP]
        thm_tip  = lm[LM.THUMB_TIP]

        # Normalised distances (resolution-independent)
        d_index_thumb  = normalised_dist(idx_tip,  thm_tip, w, h)
        d_middle_thumb = normalised_dist(mid_tip, thm_tip, w, h)

        # ── Update velocity tracker for swipe detection ─────────────────────
        wrist = lm[LM.WRIST]
        self._vel_tracker.update(wrist[0], wrist[1])

        # ── Open palm — freeze cursor ────────────────────────────────────────
        if hand.fingers_count >= OPEN_PALM_FINGER_COUNT:
            self._release_drag()
            return Gesture.OPEN_PALM

        # ── Scroll mode — middle+thumb pinch, move hand up/down ──────────────
        if d_middle_thumb < SCROLL_PINCH_THRESHOLD and fup[1]:
            if self._scroll_ref_y is None:
                self._scroll_ref_y = mid_tip[1]
            else:
                delta_norm = (mid_tip[1] - self._scroll_ref_y) / h
                if abs(delta_norm) > SCROLL_DEAD_ZONE and self._scroll_cooldown.ready():
                    direction = 1 if delta_norm < 0 else -1
                    pyautogui.scroll(direction * SCROLL_SENSITIVITY)
                    self._scroll_cooldown.reset()
                    self._scroll_ref_y = mid_tip[1]
            return Gesture.PINCH_MIDDLE
        else:
            self._scroll_ref_y = None

        # ── Victory / V-sign — right click ───────────────────────────────────
        if fup[1] and fup[2] and not fup[3] and not fup[4]:
            # Check index and middle are spread apart (not pinching together)
            spread = normalised_dist(idx_tip, mid_tip, w, h)
            if spread > 0.04:
                if self._rclick_cooldown.ready():
                    pyautogui.rightClick()
                    self._rclick_cooldown.reset()
                return Gesture.VICTORY

        # ── Index+Thumb pinch — click or drag ────────────────────────────────
        if d_index_thumb < PINCH_THRESHOLD:
            now = time.perf_counter()

            if self._pinch_start_time is None:
                self._pinch_start_time = now

            held_frames = (now - self._pinch_start_time)

            if held_frames >= (DRAG_HOLD_FRAMES / TARGET_FPS):
                # Held long enough — drag
                if not self._dragging:
                    pyautogui.mouseDown()
                    self._dragging = True
                # Move cursor while dragging
                cx, cy = self._map_to_screen(idx_tip, w, h)
                sx, sy = self._cursor_filter.update(cx, cy)
                pyautogui.moveTo(int(sx), int(sy))
                return Gesture.PINCH

            # Still in pinch but not yet held long enough; move cursor
            cx, cy = self._map_to_screen(idx_tip, w, h)
            sx, sy = self._cursor_filter.update(cx, cy)
            pyautogui.moveTo(int(sx), int(sy))
            return Gesture.PINCH

        else:
            # Pinch released
            if self._pinch_start_time is not None:
                held = time.perf_counter() - self._pinch_start_time
                if not self._dragging and held < (DRAG_HOLD_FRAMES / TARGET_FPS):
                    # Short pinch → click
                    if self._click_cooldown.ready():
                        now = time.perf_counter()
                        if (now - self._last_pinch_time) < DOUBLE_PINCH_GAP:
                            pyautogui.doubleClick()
                        else:
                            pyautogui.click()
                        self._last_pinch_time = now
                        self._click_cooldown.reset()
                self._release_drag()
            self._pinch_start_time = None

        # ── Pointing — move cursor (index finger only) ────────────────────────
        if fup[1] and not fup[2]:
            cx, cy = self._map_to_screen(idx_tip, w, h)
            sx, sy = self._cursor_filter.update(cx, cy)
            pyautogui.moveTo(int(sx), int(sy))
            return Gesture.POINTING

        # ── No specific gesture detected ─────────────────────────────────────
        return Gesture.NONE

    # ─────────────────────────────────────────────────────────────────────────
    # MODE 2 – MEDIA CONTROL
    # ─────────────────────────────────────────────────────────────────────────
    def _handle_media(self, hand: HandResult) -> Gesture:
        lm   = hand.landmarks_px
        fup  = hand.fingers_up
        w, h = self.frame_w, self.frame_h

        wrist = lm[LM.WRIST]
        self._vel_tracker.update(wrist[0], wrist[1])

        # ── Open palm — play/pause ───────────────────────────────────────────
        if hand.fingers_count >= OPEN_PALM_FINGER_COUNT:
            if self._media_cooldown.ready():
                pyautogui.press("playpause")
                self._media_cooldown.reset()
            return Gesture.OPEN_PALM

        # ── Thumbs up — volume up ────────────────────────────────────────────
        if fup[0] and not any(fup[1:]):
            if self._media_cooldown.ready():
                pyautogui.press("volumeup")
                self._media_cooldown.reset()
            return Gesture.THUMBS_UP

        # ── Thumbs down (wrist above thumb tip) — volume down ────────────────
        thm_tip = lm[LM.THUMB_TIP]
        wrist_pt = lm[LM.WRIST]
        if fup[0] and not any(fup[1:]) and thm_tip[1] > wrist_pt[1]:
            if self._media_cooldown.ready():
                pyautogui.press("volumedown")
                self._media_cooldown.reset()
            return Gesture.THUMBS_DOWN

        # ── Swipe left/right — prev / next track ─────────────────────────────
        vx = self._vel_tracker.vx
        vy = self._vel_tracker.vy
        speed = abs(vx)

        if speed > SWIPE_MIN_VELOCITY * (w / 640):   # Scale with resolution
            if self._media_cooldown.ready():
                if vx > 0:
                    pyautogui.press("nexttrack")
                    self._media_cooldown.reset()
                    return Gesture.SWIPE_RIGHT
                else:
                    pyautogui.press("prevtrack")
                    self._media_cooldown.reset()
                    return Gesture.SWIPE_LEFT

        return Gesture.NONE

    # ─────────────────────────────────────────────────────────────────────────
    # MODE 3 – VIRTUAL CANVAS
    # ─────────────────────────────────────────────────────────────────────────
    def _handle_canvas(self, hand: HandResult) -> Gesture:
        lm   = hand.landmarks_px
        fup  = hand.fingers_up
        w, h = self.frame_w, self.frame_h

        self.init_canvas(h, w)

        idx_tip = lm[LM.INDEX_TIP]
        mid_tip = lm[LM.MIDDLE_TIP]
        thm_tip = lm[LM.THUMB_TIP]

        d_pinch = normalised_dist(idx_tip, thm_tip, w, h)

        # ── Cycle colour — pinch ─────────────────────────────────────────────
        if d_pinch < PINCH_THRESHOLD and self._color_cycle_cooldown.ready():
            self._canvas_color_idx = (self._canvas_color_idx + 1) % len(CANVAS_COLORS)
            self._color_cycle_cooldown.reset()
            self._prev_draw_pt = None
            return Gesture.PINCH

        # ── Erase — index + middle up ────────────────────────────────────────
        if fup[1] and fup[2] and not fup[3] and not fup[4]:
            mid = midpoint(idx_tip, mid_tip)
            ex, ey = int(mid[0]), int(mid[1])
            cv2.circle(self._canvas, (ex, ey), CANVAS_ERASE_SIZE, (0, 0, 0), -1)
            self._prev_draw_pt = None
            return Gesture.ERASE

        # ── Open palm — lift pen ─────────────────────────────────────────────
        if hand.fingers_count >= OPEN_PALM_FINGER_COUNT:
            self._prev_draw_pt = None
            return Gesture.OPEN_PALM

        # ── Draw — index finger up only ───────────────────────────────────────
        if fup[1] and not fup[2]:
            color = CANVAS_COLORS[self._canvas_color_idx]
            if self._prev_draw_pt is not None:
                cv2.line(self._canvas, self._prev_draw_pt, idx_tip,
                         color, CANVAS_BRUSH_SIZE)
            else:
                cv2.circle(self._canvas, idx_tip, CANVAS_BRUSH_SIZE // 2, color, -1)
            self._prev_draw_pt = idx_tip
            return Gesture.DRAW

        self._prev_draw_pt = None
        return Gesture.NONE

    # ─────────────────────────────────────────────────────────────────────────
    # Helpers
    # ─────────────────────────────────────────────────────────────────────────
    def _map_to_screen(self, pt: Tuple[int, int],
                        fw: int, fh: int) -> Tuple[float, float]:
        """
        Map a point in camera space (with margins stripped) to screen coordinates.
        """
        sx = remap(pt[0], MOUSE_MARGIN_X, fw - MOUSE_MARGIN_X,
                   0, self.screen_w)
        sy = remap(pt[1], MOUSE_MARGIN_Y, fh - MOUSE_MARGIN_Y,
                   0, self.screen_h)
        return sx * MOUSE_SPEED, sy * MOUSE_SPEED

    def _release_drag(self):
        if self._dragging:
            pyautogui.mouseUp()
            self._dragging = False

    # ─────────────────────────────────────────────────────────────────────────
    # Mode switching
    # ─────────────────────────────────────────────────────────────────────────
    def set_mode(self, mode: Mode):
        self._release_drag()
        self._scroll_ref_y = None
        self._prev_draw_pt = None
        self._pinch_start_time = None
        self._swipe_start_x = None
        self._cursor_filter.reset()
        self._vel_tracker.reset()
        self.mode = mode
