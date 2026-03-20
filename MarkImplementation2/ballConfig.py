# ballConfig.py
# ─────────────────────────────────────────────────────────────────────────────
# Configuration for the ping-pong ball detector / APF vector generator.
# Replace the camera section values with your own calibration output.
# ─────────────────────────────────────────────────────────────────────────────

CONFIG = {

    # ── Camera ───────────────────────────────────────────────────────────────
    "camera": {
        "frame_width":  640,
        "frame_height": 480,

        # Output of cv2.calibrateCamera() — 3x3 intrinsic matrix
        "camera_matrix": [
            [612.4, 0.0,   318.7],   # <- replace with your fx, 0, cx
            [0.0,   611.8, 241.3],   # <- replace with your  0, fy, cy
            [0.0,   0.0,   1.0  ],
        ],

        # Distortion coefficients [k1, k2, p1, p2, k3]
        "dist_coeffs": [-0.38, 0.21, 0.001, -0.0005, -0.07],  # <- replace with yours
    },

    # ── Pre-processing ────────────────────────────────────────────────────────
    # Gaussian blur applied to the BGR frame BEFORE HSV conversion.
    # This is critical — blurring first stabilises HSV values significantly.
    # Must be odd integers.
    "gaussian_blur_kernel": [7, 7],

    # ── Ping-pong ball detection ──────────────────────────────────────────────
    "ping_pong": {
        "diameter_mm": 40.0,

        # HSV range for orange ping-pong balls.
        # S_min=120, V_min=120 are from the working reference.
        # If balls appear dark/dim lower both toward 80.
        "ball_hsv_lower": [5,  120, 120],   # [H_min, S_min, V_min]
        "ball_hsv_upper": [25, 255, 255],   # [H_max, S_max, V_max]

        # Contour area bounds (pixels squared).
        "min_area_px2": 90,
        "max_area_px2": 80000,

        # Radius bounds as secondary guard.
        "min_radius_px": 7.0,
        "max_radius_px": 160.0,

        # Circularity = 4pi*area/perimeter^2.
        # 0.58 from working reference — lower = more permissive.
        "circularity_threshold": 0.58,
    },

    # ── Obstacle detection ────────────────────────────────────────────────────
    # White floor has S~0 and is excluded by min_saturation alone.
    # No separate floor mask needed.
    "obstacles": {
        "min_area_px2": 1200,
        "max_area_px2": 120000,

        "min_value":      15,    # pixels darker than this are ignored
        "max_value":     210,    # pixels brighter than this are ignored
        "min_saturation": 35,    # KEY: white/grey floor (S~0) excluded here

        # Ignore the top N% of the frame (background, ceiling lights).
        "top_ignore_ratio": 0.12,
    },

    # ── Morphology ────────────────────────────────────────────────────────────
    # iterations=1 matches working reference.
    "morphology": {
        "open_kernel":  [3, 3],
        "close_kernel": [5, 5],
        "iterations":   1,
    },

    # ── Artificial Potential Field ────────────────────────────────────────────
    "apf": {
        "k_att":            1.0,
        "k_rep":            0.75,
        "rep_influence_cm": 30.0,
        "min_magnitude":    0.18,   # matches MIN_TRACKING_MAGNITUDE in reference

        # "bottom_center" | "center" | [x, y]
        "robot_position": "bottom_center",
    },
}