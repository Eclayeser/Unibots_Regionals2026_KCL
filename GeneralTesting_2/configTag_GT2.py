# ─────────────────────────────────────────────────────────────────────────────
# AprilTag Navigator
# ─────────────────────────────────────────────────────────────────────────────

# The two tag IDs printed on your basket
APRILTAG_TARGET_IDS: set[int] = {1, 2}

# Tag family — must match what is printed. Common: "tag36h11", "tagStandard41h12"
APRILTAG_FAMILY = "tag36h11"

# Physical side-length of the printed tag square in metres
APRILTAG_TAG_SIZE_M = 0.1           # ← measure your actual tag

# Camera intrinsics from OpenCV calibration: (fx, fy, cx, cy) in pixels
# Replace with real values from your calibration run
APRILTAG_CAMERA_PARAMS: tuple[float, float, float, float] = (
    564.98,   # fx  ← placeholder
    565.87,   # fy  ← placeholder
    303.59,   # cx  (frame_width  / 2 as rough fallback)
    282.69,   # cy  (frame_height / 2 as rough fallback)
)

# Frame dimensions must match cv2.VideoCapture resolution
APRILTAG_FRAME_SIZE: tuple[int, int] = (640, 480)   # (width, height)

# Lock conditions — robot is considered docked when ALL three are satisfied
APRILTAG_LOCK_DISTANCE_CM      = 15.0   # must be ≤ this distance
APRILTAG_LOCK_CENTER_TOL_PX    = 40     # tag centre within this many px of image centre
APRILTAG_LOCK_YAW_DEG          = 10.0   # tag yaw within this many degrees of head-on