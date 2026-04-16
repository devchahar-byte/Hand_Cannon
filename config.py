# =============================================================================
# config.py — Central configuration for Hand Gesture Control
# Tune these values to match your lighting, hand size, and webcam resolution.
# =============================================================================

# ── Camera ────────────────────────────────────────────────────────────────────
CAMERA_INDEX        = 0          # 0 = default webcam; change if using external cam
FRAME_WIDTH         = 1280       # Capture width  (px)
FRAME_HEIGHT        = 720        # Capture height (px)
TARGET_FPS          = 30         # Desired pipeline frame-rate

# ── MediaPipe Hand Tracking ───────────────────────────────────────────────────
MAX_NUM_HANDS           = 1      # Track only 1 hand (cheaper; raise to 2 for dual-hand)
MIN_DETECTION_CONFIDENCE= 0.80   # Initial hand-detection confidence threshold
MIN_TRACKING_CONFIDENCE = 0.70   # Subsequent frame tracking confidence threshold
MODEL_COMPLEXITY        = 1      # 0 = lite, 1 = full (more accurate, slightly slower)

# ── Display ───────────────────────────────────────────────────────────────────
SHOW_FPS            = True       # Draw FPS counter on screen
SHOW_LANDMARKS      = True       # Draw the 21-point skeleton
SHOW_GESTURE_LABEL  = True       # Print detected gesture name
SHOW_MODE_BANNER    = True       # Show current mode at top of frame
CANVAS_OPACITY      = 0.35       # Alpha for virtual-canvas overlay (0–1)

# ── Mouse Control ─────────────────────────────────────────────────────────────
# The usable region of the frame (margins avoid edge jitter).
MOUSE_MARGIN_X      = 150        # px from left/right edges (dead zone)
MOUSE_MARGIN_Y       = 100        # px from top/bottom edges (dead zone)
MOUSE_SMOOTHING     = 7          # Exponential-moving-average window (higher = smoother but laggier)
MOUSE_SPEED         = 1.0        # Multiplier applied on top of screen mapping

# ── Gesture Thresholds ────────────────────────────────────────────────────────
# Distances are normalised (0-1) relative to image diagonal unless noted.
PINCH_THRESHOLD         = 0.055  # Index-tip ↔ Thumb-tip  → CLICK
DOUBLE_PINCH_GAP        = 0.30   # Seconds; two pinches within this window → DOUBLE-CLICK
SCROLL_PINCH_THRESHOLD  = 0.060  # Middle-tip ↔ Thumb-tip → SCROLL mode
SCROLL_SENSITIVITY      = 25     # Pixel scroll amount per frame of detected scroll
DRAG_HOLD_FRAMES        = 8      # Frames pinch must be held before drag begins

# Open-palm detection (≥4 fingers extended)
OPEN_PALM_FINGER_COUNT  = 4      # Minimum extended fingers to count as "open palm"

# Swipe gesture
SWIPE_MIN_VELOCITY      = 40     # px/frame to register as a swipe
SWIPE_CONFIRMATION_MS   = 180    # Must sustain direction for this many ms

# Victory / peace sign (index + middle up, rest down)
VICTORY_THRESHOLD       = 0.045  # Ring/Pinky curl threshold for victory check

# ── Scroll ────────────────────────────────────────────────────────────────────
SCROLL_DEAD_ZONE        = 0.015  # Normalised delta below which scrolling is ignored

# ── Media Controller Mode ─────────────────────────────────────────────────────
MEDIA_SWIPE_COOLDOWN_S  = 0.8    # Seconds between repeated media swipe actions
VOLUME_STEP             = 5      # % volume change per pinch-scroll tick

# ── Virtual Canvas Mode ───────────────────────────────────────────────────────
CANVAS_BRUSH_SIZE       = 8      # Drawing brush radius (px)
CANVAS_ERASE_SIZE       = 40     # Eraser radius (px)
CANVAS_COLORS = [                # Cycle through with thumb-index pinch
    (0,   255, 150),             # Neon green
    (0,   180, 255),             # Sky blue
    (255,  60,  60),             # Red
    (255, 200,   0),             # Yellow
    (200,   0, 255),             # Purple
    (255, 255, 255),             # White
]

# ── Skin-Colour HSV Fallback (used when MediaPipe loses the hand) ─────────────
# Tweak the upper/lower bounds for different skin tones.
SKIN_HSV_LOWER  = (0,  20,  70)
SKIN_HSV_UPPER  = (25, 255, 255)

# ── Keyboard Shortcuts (mode switching) ──────────────────────────────────────
KEY_QUIT        = ord('q')
KEY_MODE_MOUSE  = ord('1')
KEY_MODE_MEDIA  = ord('2')
KEY_MODE_CANVAS = ord('3')
KEY_CLEAR_CANVAS= ord('c')
KEY_SAVE_CANVAS = ord('s')

# ── Colours used in the HUD overlay ──────────────────────────────────────────
COLOR_GREEN  = (0,   255, 100)
COLOR_BLUE   = (100, 180, 255)
COLOR_RED    = (0,   80,  255)
COLOR_YELLOW = (0,   220, 255)
COLOR_WHITE  = (255, 255, 255)
COLOR_GRAY   = (160, 160, 160)
COLOR_BG     = (20,  20,   20)
