# =============================================================================
# main.py — Hand Gesture Control System  |  Entry point
# =============================================================================
"""
Run with:
    python main.py

Controls:
  1  → Mouse control mode
  2  → Media control mode
  3  → Virtual canvas mode
  c  → Clear canvas (canvas mode only)
  s  → Save canvas to PNG (canvas mode only)
  q  → Quit

Pipeline overview (one frame):
  ┌──────────────┐
  │  WebCam BGR  │  OpenCV VideoCapture
  └──────┬───────┘
         │  flip (mirror)
  ┌──────▼───────┐
  │  Pre-process │  Resize, colour-space conversion (BGR→RGB for MP)
  └──────┬───────┘
         │
  ┌──────▼───────┐
  │  MediaPipe   │  21 3-D hand landmarks detected at 30+ fps
  │  Hand model  │
  └──────┬───────┘
         │  HandResult objects
  ┌──────▼───────┐
  │  Gesture     │  Euclidean distances, angle checks, velocity tracking
  │  Classifier  │  → Gesture enum
  └──────┬───────┘
         │  system call
  ┌──────▼───────────────────────────────┐
  │  PyAutoGUI / keyboard  (OS actions)  │
  └──────────────────────────────────────┘
         │  also
  ┌──────▼───────┐
  │  HUD Overlay │  FPS, mode, gesture label, finger dots
  └──────┬───────┘
         │
  ┌──────▼───────┐
  │  cv2.imshow  │  Display window
  └──────────────┘
"""
from __future__ import annotations

import sys
import time

import cv2

from config import (
    CAMERA_INDEX, FRAME_WIDTH, FRAME_HEIGHT, TARGET_FPS,
    KEY_QUIT, KEY_MODE_MOUSE, KEY_MODE_MEDIA, KEY_MODE_CANVAS,
    KEY_CLEAR_CANVAS, KEY_SAVE_CANVAS,
    SKIN_HSV_LOWER, SKIN_HSV_UPPER,
    COLOR_RED,
)
from gesture_controller import GestureController, Mode
from hand_tracker import HandTracker
from hud import HUD
from utils import FPSCounter, skin_mask


# ─────────────────────────────────────────────────────────────────────────────
# Camera setup
# ─────────────────────────────────────────────────────────────────────────────
def open_camera(index: int, width: int, height: int) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(index, cv2.CAP_ANY)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open camera at index {index}.")
        sys.exit(1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS,          TARGET_FPS)
    # Reduce buffer to minimise latency
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[INFO] Camera opened: {actual_w}×{actual_h}")
    return cap, actual_w, actual_h


# ─────────────────────────────────────────────────────────────────────────────
# Main loop
# ─────────────────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  Hand Gesture Control System")
    print("=" * 60)
    print("  Press  1  →  Mouse Control")
    print("  Press  2  →  Media Control")
    print("  Press  3  →  Virtual Canvas")
    print("  Press  q  →  Quit")
    print("=" * 60)

    # ── Open webcam ──────────────────────────────────────────────────────────
    cap, fw, fh = open_camera(CAMERA_INDEX, FRAME_WIDTH, FRAME_HEIGHT)

    # ── Initialise subsystems ────────────────────────────────────────────────
    tracker    = HandTracker()
    controller = GestureController(fw, fh)
    hud        = HUD(fw, fh)

    # Frame timing for stable loop rate
    frame_interval = 1.0 / TARGET_FPS
    prev_frame_time = 0.0

    # ── Window configuration ─────────────────────────────────────────────────
    window_name = "Hand Gesture Control  |  q=quit  1=Mouse  2=Media  3=Canvas"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, fw, fh)

    # ── No-hand warning state ────────────────────────────────────────────────
    no_hand_frames = 0
    NO_HAND_WARN_AFTER = TARGET_FPS * 3   # warn after 3 s of no detection

    # ── Main frame loop ──────────────────────────────────────────────────────
    while True:
        # Rate-limit the loop so we don't spin the CPU needlessly
        now = time.perf_counter()
        if (now - prev_frame_time) < frame_interval:
            continue
        prev_frame_time = now

        ret, frame = cap.read()
        if not ret:
            print("[WARN] Dropped frame — camera read failed.")
            continue

        # ── Pre-processing ────────────────────────────────────────────────────
        # Mirror the frame so it feels like a looking-glass
        frame = cv2.flip(frame, 1)

        # ── Hand landmark detection ───────────────────────────────────────────
        hands = tracker.process(frame)

        # Draw landmarks on frame if enabled
        tracker.draw(frame, hands)

        # ── Gesture classification + system action ────────────────────────────
        gesture = controller.process(hands)

        # ── Track missing-hand frames ─────────────────────────────────────────
        if not hands:
            no_hand_frames += 1
        else:
            no_hand_frames = 0

        # ── Fallback skin-mask visualisation when hand is lost ────────────────
        if no_hand_frames > 10:
            smask = skin_mask(frame, SKIN_HSV_LOWER, SKIN_HSV_UPPER)
            # Tint the frame slightly red to warn user
            if no_hand_frames > NO_HAND_WARN_AFTER:
                red_overlay = frame.copy()
                red_overlay[:, :, 2] = 255
                frame = cv2.addWeighted(frame, 0.92, red_overlay, 0.08, 0)
                cv2.putText(frame, "No hand detected — show your palm to camera",
                            (fw // 2 - 280, fh // 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.75, COLOR_RED, 2, cv2.LINE_AA)

        # ── Render HUD ────────────────────────────────────────────────────────
        canvas = controller.get_canvas()
        output = hud.render(frame, controller.mode, gesture, hands, canvas)

        # ── Display ───────────────────────────────────────────────────────────
        cv2.imshow(window_name, output)

        # ── Keyboard input ────────────────────────────────────────────────────
        key = cv2.waitKey(1) & 0xFF

        if key == KEY_QUIT:
            print("[INFO] Quit requested.")
            break
        elif key == KEY_MODE_MOUSE:
            controller.set_mode(Mode.MOUSE)
            print("[INFO] Switched to Mouse Control mode.")
        elif key == KEY_MODE_MEDIA:
            controller.set_mode(Mode.MEDIA)
            print("[INFO] Switched to Media Control mode.")
        elif key == KEY_MODE_CANVAS:
            controller.set_mode(Mode.CANVAS)
            print("[INFO] Switched to Virtual Canvas mode.")
        elif key == KEY_CLEAR_CANVAS:
            controller.clear_canvas()
            print("[INFO] Canvas cleared.")
        elif key == KEY_SAVE_CANVAS:
            path = "canvas_output.png"
            if controller.save_canvas(path):
                print(f"[INFO] Canvas saved to {path}")
            else:
                print("[WARN] No canvas to save — switch to Canvas mode first.")

        # Exit if window was closed by the user
        if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
            break

    # ── Cleanup ───────────────────────────────────────────────────────────────
    print("[INFO] Releasing resources…")
    tracker.close()
    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Done. Goodbye!")


if __name__ == "__main__":
    main()
