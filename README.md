# Hand Gesture Control System

Control your computer's mouse cursor, media playback, and a virtual drawing canvas — entirely through hand movements captured by your webcam.

---

## Architecture

```
WebCam (BGR)
    │
    ▼  cv2.flip() + BGR→RGB
Pre-processing
    │
    ▼  mediapipe.solutions.hands
Hand Landmark Detection  ──► 21 (x, y, z) keypoints at 30+ fps
    │
    ▼  GestureController.process()
Gesture Classification
    ├─ Euclidean distance  (pinch detection)
    ├─ y-coordinate comparison (finger extension)
    ├─ Velocity tracking   (swipe detection)
    └─ Angle analysis      (knuckle bend)
    │
    ▼
System Action Executor
    ├─ pyautogui.moveTo / click / scroll / drag
    ├─ pyautogui.press('playpause' / 'nexttrack' …)
    └─ cv2 canvas drawing
    │
    ▼
HUD Overlay → cv2.imshow()
```

---

## File Structure

```
hand_gesture_control/
│
├── main.py               # Entry point — camera loop & orchestration
├── hand_tracker.py       # MediaPipe wrapper; produces HandResult objects
├── gesture_controller.py # Gesture classifier + OS action dispatcher
├── hud.py                # On-screen overlay (FPS, mode, gesture label)
├── utils.py              # Math helpers (EMA filter, velocity, distances)
├── config.py             # All tunable thresholds in one place
└── requirements.txt
```

---

## Setup

```bash
# 1. Clone / download this project
cd hand_gesture_control

# 2. Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run
python main.py
```

> **Linux users**: PyAutoGUI needs `python3-xlib` and a display server.
> ```bash
> sudo apt install python3-xlib scrot
> ```

---

## Modes & Gestures

### Mode 1 — Mouse Control  `[press 1]`

| Gesture | Action |
|---|---|
| ☝️ Index finger only | Move cursor |
| 🤌 Index + Thumb pinch (quick) | Left click |
| 🤌 Hold pinch | Click-and-drag |
| 🤌 Pinch twice fast | Double-click |
| 🖖 Middle + Thumb pinch + move | Scroll up / down |
| ✌️ V-sign (spread) | Right-click |
| 🖐 Open palm (≥4 fingers) | Freeze cursor |

### Mode 2 — Media Control  `[press 2]`

| Gesture | Action |
|---|---|
| 👋 Swipe right | Next track |
| 👋 Swipe left | Prev track |
| 🖐 Open palm | Play / Pause |
| 👍 Thumbs up | Volume up |
| 👎 Thumbs down | Volume down |

### Mode 3 — Virtual Canvas  `[press 3]`

| Gesture | Action |
|---|---|
| ☝️ Index up only | Draw |
| ✌️ Index + middle up | Erase |
| 🤌 Index + Thumb pinch | Cycle brush colour |
| 🖐 Open palm | Lift pen |
| `c` key | Clear canvas |
| `s` key | Save canvas to `canvas_output.png` |

---

## How It Works — The Math

### Pinch Detection
```python
# Normalised Euclidean distance between fingertips
d = math.hypot(tip1_x - tip2_x, tip1_y - tip2_y) / math.hypot(frame_w, frame_h)
if d < PINCH_THRESHOLD:   # default 0.055
    trigger_click()
```

### Finger Extension
```python
# A finger is "up" if its tip pixel is higher than its PIP joint
finger_up = landmark[TIP].y < landmark[PIP].y
```

### Swipe Detection
```python
# Wrist position tracked across 8 frames → velocity in px/s
vx = (x_now - x_8_frames_ago) / elapsed_seconds
if abs(vx) > SWIPE_MIN_VELOCITY:
    trigger_next_track() if vx > 0 else trigger_prev_track()
```

### Cursor Smoothing
The raw index-fingertip position is noisy. We apply a **Moving Average Filter** (7-frame sliding window) before calling `pyautogui.moveTo()`:
```
smoothed_x = mean(last 7 raw_x readings)
```

---

## Tuning for Your Environment

All thresholds live in `config.py`. Key parameters:

| Parameter | Default | What to change |
|---|---|---|
| `PINCH_THRESHOLD` | 0.055 | Lower if clicks trigger too easily |
| `MOUSE_SMOOTHING` | 7 | Increase for smoother (but laggier) cursor |
| `SWIPE_MIN_VELOCITY` | 40 | Increase to avoid accidental swipes |
| `SKIN_HSV_LOWER/UPPER` | (0,20,70) | Adjust for your skin tone / lighting |
| `MIN_DETECTION_CONFIDENCE` | 0.80 | Lower if hand detection is unreliable |

---

## Performance Tips

- Keep **well-lit** conditions — MediaPipe struggles with shadows on the hand.
- Position your webcam so the hand is **against a plain background**.
- If CPU usage is too high, set `MODEL_COMPLEXITY = 0` in `config.py` (lite model).
- Close other camera-using applications before running.

---

## Extending the Project

**Add a gesture**: define a new `Gesture` enum member, add detection logic in `_handle_mouse()`, and bind it to a `pyautogui` call.

**Dual-hand support**: set `MAX_NUM_HANDS = 2` in `config.py` and handle `hands[1]` for a second action channel.

**Volume slider**: track the distance between both hands' index fingers and map it linearly to system volume.

**Zoom**: detect a two-hand spread/pinch and call `pyautogui.hotkey('ctrl', '+')` / `'-'`.
