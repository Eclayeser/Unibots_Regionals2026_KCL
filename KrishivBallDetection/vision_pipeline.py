from __future__ import annotations

import cv2
import numpy as np

try:
    from . import ball_detector_runtime as bdr
    from .ping_pong_detector import extract_ping_pong
    from .steel_ball_detector import extract_steel_floor_disturbance, FloorColorTracker
except ImportError:
    import ball_detector_runtime as bdr
    from ping_pong_detector import extract_ping_pong
    from steel_ball_detector import extract_steel_floor_disturbance, FloorColorTracker


# Module-level floor tracker (persistent across frames)
_FLOOR_TRACKER = FloorColorTracker(history_frames=3)


def detect_balls_pipeline(frame: np.ndarray, ping_pong_profile: str = "orange") -> dict:
    """
    Complete ball detection pipeline:
    1. Preprocess frame (undistort, blur, convert to HSV/grayscale)
    2. Detect ping-pong balls (HSV mask + contour + Hough confirmation)
    3. Shield ping-pong regions to prevent false steel detections
    4. Detect steel balls (floor-disturbance method with Hough circles)
    5. Merge and return all detections
    """
    
    if frame is None or frame.ndim != 3:
        raise ValueError("detect_balls expects a BGR frame with shape (H, W, 3)")

    calibration = bdr._get_calibration(frame.shape)
    undistorted = bdr._undistort_frame(frame, calibration)

    blurred = cv2.GaussianBlur(undistorted, bdr.FRAME_GAUSSIAN_BLUR, 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)

    # Stage 1: Detect ping-pong balls
    ping_pong, pp_centres, pp_mask = extract_ping_pong(
        frame=undistorted,
        hsv=hsv,
        gray=gray,
        profile=ping_pong_profile,
        estimate_distance_fn=bdr._estimate_distance_cm,
        bearing_fn=bdr._bearing_deg_from_x,
        calibration=calibration,
    )

    # Stage 2: Detect steel balls (using floor-disturbance method)
    steel, steel_mask, disturbance_mask = extract_steel_floor_disturbance(
        frame=undistorted,
        hsv=hsv,
        gray=gray,
        pp_mask=pp_mask,
        pp_centres=pp_centres,
        floor_tracker=_FLOOR_TRACKER,
        estimate_distance_fn=bdr._estimate_distance_cm,
        bearing_fn=bdr._bearing_deg_from_x,
        calibration=calibration,
    )

    # Merge and sort detections
    all_detections = steel + ping_pong
    all_detections = bdr._prioritize_balls(all_detections)

    # Build combined mask
    combined_mask = cv2.bitwise_or(pp_mask, steel_mask)

    return {
        "frame": undistorted,
        "calibration": calibration,
        "detected_balls": all_detections,
        "ball_mask": combined_mask,
        "ping_pong_mask": pp_mask,
        "steel_mask": steel_mask,
        "disturbance_mask": disturbance_mask,
        "steel_kalman_roi": None,
        "steel_kalman_point": None,
        "steel_hough_circles": [],
        "steel_trajectory": [],
    }
