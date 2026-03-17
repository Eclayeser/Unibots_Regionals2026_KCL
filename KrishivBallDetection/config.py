from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class HSVRange:
    lower: np.ndarray
    upper: np.ndarray


PING_PONG_PRESETS: dict[str, tuple[HSVRange, ...]] = {
    "orange": (
        HSVRange(np.array([5, 120, 120], dtype=np.uint8), np.array([25, 255, 255], dtype=np.uint8)),
    ),
    "white": (
        HSVRange(np.array([0, 0, 200], dtype=np.uint8), np.array([179, 50, 255], dtype=np.uint8)),
    ),
}

DEFAULT_PING_PONG_PROFILE = "orange"

PING_PONG_MIN_AREA = 90
PING_PONG_MAX_AREA = 80000
PING_PONG_MIN_CIRCULARITY = 0.58
PING_PONG_RADIUS_RANGE_PX = (7.0, 160.0)

STEEL_MAX_SATURATION = 70
STEEL_MIN_VALUE = 20
STEEL_MAX_VALUE = 185
STEEL_MIN_AREA = 30
STEEL_MAX_AREA = 6000
STEEL_MIN_CIRCULARITY = 0.62
STEEL_RADIUS_RANGE_PX = (4.0, 42.0)
STEEL_MAX_MEAN_VALUE = 170.0

SPECULAR_MAX_SATURATION = 70
SPECULAR_MIN_VALUE = 190
SPECULAR_MIN_RATIO = 0.002
SPECULAR_MAX_RATIO = 0.38

OBSTACLE_MIN_AREA = 1200
OBSTACLE_MAX_AREA = 120000
OBSTACLE_MIN_VALUE = 15
OBSTACLE_MAX_VALUE = 210
OBSTACLE_MIN_SATURATION = 35
OBSTACLE_TOP_IGNORE_RATIO = 0.12
OBSTACLE_SIZE_FULL_SCALE = 180.0

FRAME_GAUSSIAN_BLUR = (5, 5)
MASK_OPEN_KERNEL = (3, 3)
MASK_CLOSE_KERNEL = (5, 5)

BALL_TYPE_PRIORITY = {"steel": 1, "PingPong": 0}
BALL_DIAMETER_CM = {"steel": 2.0, "PingPong": 4.0}

GRAB_MAX_ABS_ANGLE_DEG = 4.5
GRAB_DISTANCE_CM = {"steel": 14.0, "PingPong": 18.0}

TARGET_DISTANCE_NEAR_CM = 15.0
TARGET_DISTANCE_FAR_CM = 130.0
MIN_TRACKING_MAGNITUDE = 0.18

ATTRACTION_WEIGHT = 1.0
REPULSION_WEIGHT = 0.75

# ── Steel ball CLAHE preprocessing ───────────────────────────────────────────
STEEL_CLAHE_CLIP_LIMIT: float = 2.0
STEEL_CLAHE_TILE_GRID_SIZE: tuple = (8, 8)

# ── Steel ball morphological close (fills specular highlight voids) ───────────
STEEL_MORPH_CLOSE_BIG_KERNEL: tuple = (11, 11)

# ── Steel ball strict contour filters ────────────────────────────────────────
STEEL_MIN_CIRCULARITY_STRICT: float = 0.75   # replaces relaxed STEEL_MIN_CIRCULARITY for new pipeline
STEEL_MIN_CONVEXITY: float = 0.82            # convexity = contourArea / convexHullArea

# ── Kalman filter tracker constants ──────────────────────────────────────────
KALMAN_ROI_HALF_SIZE: int = 100         # initial ROI is 200×200 px centred on prediction
KALMAN_MAX_LOST_FRAMES: int = 5         # frames to coast on Kalman prediction without a measurement
KALMAN_ROI_EXPAND_PX: int = 20          # expand ROI by this many px per consecutive lost frame

# ── Debug stream ──────────────────────────────────────────────────────────────
DEBUG_STREAM_PORT: int = 5000
DEBUG_TRAJECTORY_LENGTH: int = 8        # number of past predicted positions drawn as trajectory
