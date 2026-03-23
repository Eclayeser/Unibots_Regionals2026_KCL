# ─────────────────────────────────────────────────────────────────────────────
# AprilTag Navigator – Tag ID Configuration
# ─────────────────────────────────────────────────────────────────────────────

# ── Our basket ───────────────────────────────────────────────────────────────
# The two tag IDs printed on OUR basket.
APRILTAG_OWN_TAG_IDS: set[int] = {8, 9}

# ── Opponent baskets ─────────────────────────────────────────────────────────
# All tag IDs that belong to OPPONENT baskets (up to 6 tags).
APRILTAG_OPPONENT_TAG_IDS: set[int] = {16, 19, 20, 21, 22, 1}  # ← UPDATE with real IDs

# Legacy alias – kept so any other file that still references APRILTAG_TARGET_IDS
# continues to work without modification.
APRILTAG_TARGET_IDS: set[int] = APRILTAG_OWN_TAG_IDS

# ── Tag family ────────────────────────────────────────────────────────────────
# Must match what is physically printed. Common: "tag36h11", "tagStandard41h12"
APRILTAG_FAMILY = "tag36h11"

# ── Physical tag size ─────────────────────────────────────────────────────────
# Side-length of the printed tag square in metres. Measure your actual tag.
APRILTAG_TAG_SIZE_M = 0.1

# ── Docking trigger distance (own basket) ────────────────────────────────────
# Robot locks into the docking sequence when closer than this to OUR basket tag.
DISTANCE_UNTIL_DOCKING_CM = 35

# ── Opponent divert distance ──────────────────────────────────────────────────
# When searching for an opponent basket (because own basket is blocked), the
# robot approaches an opponent tag only until this distance, then gives up and
# resumes searching for its own basket.  This clears the path without committing
# to a full approach.
OPPONENT_DIVERT_DISTANCE_CM = 85   # cm  ← adjust as needed

# ── 360° rotation detection ───────────────────────────────────────────────────
# How long (seconds) a continuous pivot_left(20) takes to complete a full 360°
# rotation at the speed used in the storageFull search branch.
# Calibrate physically: run pivot_left(20) and time one full rotation.
TIME_FOR_FULL_ROTATION_S = 5.0     # seconds  ← CALIBRATE on your robot

# ── Camera intrinsics (from OpenCV calibration) ───────────────────────────────
# (fx, fy, cx, cy) in pixels.  Replace with values from your calibration run.
APRILTAG_CAMERA_PARAMS: tuple[float, float, float, float] = (
    564.98,   # fx  ← placeholder
    565.87,   # fy  ← placeholder
    303.59,   # cx
    282.69,   # cy
)

# ── Frame dimensions ──────────────────────────────────────────────────────────
# Must match the resolution set in cv2.VideoCapture.
APRILTAG_FRAME_SIZE: tuple[int, int] = (640, 480)   # (width, height)