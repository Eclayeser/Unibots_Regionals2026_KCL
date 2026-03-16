from __future__ import annotations

import math
import time
import logging
from dataclasses import dataclass

import cv2
import numpy as np

try:
    from .ball_detection_krishiv import CAMERA_MATRIX, DIST_COEFFS
    from .config import (
        ATTRACTION_WEIGHT,
        BALL_DIAMETER_CM,
        BALL_TYPE_PRIORITY,
        DEFAULT_PING_PONG_PROFILE,
        FRAME_GAUSSIAN_BLUR,
        GRAB_DISTANCE_CM,
        GRAB_MAX_ABS_ANGLE_DEG,
        HSVRange,
        MASK_CLOSE_KERNEL,
        MASK_OPEN_KERNEL,
        MIN_TRACKING_MAGNITUDE,
        OBSTACLE_MAX_AREA,
        OBSTACLE_MAX_VALUE,
        OBSTACLE_MIN_AREA,
        OBSTACLE_MIN_SATURATION,
        OBSTACLE_MIN_VALUE,
        OBSTACLE_SIZE_FULL_SCALE,
        OBSTACLE_TOP_IGNORE_RATIO,
        PING_PONG_MAX_AREA,
        PING_PONG_MIN_AREA,
        PING_PONG_MIN_CIRCULARITY,
        PING_PONG_PRESETS,
        PING_PONG_RADIUS_RANGE_PX,
        REPULSION_WEIGHT,
        SPECULAR_MAX_RATIO,
        SPECULAR_MAX_SATURATION,
        SPECULAR_MIN_RATIO,
        SPECULAR_MIN_VALUE,
        STEEL_MAX_AREA,
        STEEL_MAX_MEAN_VALUE,
        STEEL_MAX_SATURATION,
        STEEL_MAX_VALUE,
        STEEL_MIN_AREA,
        STEEL_MIN_CIRCULARITY,
        STEEL_MIN_VALUE,
        STEEL_RADIUS_RANGE_PX,
        TARGET_DISTANCE_FAR_CM,
        TARGET_DISTANCE_NEAR_CM,
    )
except ImportError:
    from ball_detection_krishiv import CAMERA_MATRIX, DIST_COEFFS
    from config import (
        ATTRACTION_WEIGHT,
        BALL_DIAMETER_CM,
        BALL_TYPE_PRIORITY,
        DEFAULT_PING_PONG_PROFILE,
        FRAME_GAUSSIAN_BLUR,
        GRAB_DISTANCE_CM,
        GRAB_MAX_ABS_ANGLE_DEG,
        HSVRange,
        MASK_CLOSE_KERNEL,
        MASK_OPEN_KERNEL,
        MIN_TRACKING_MAGNITUDE,
        OBSTACLE_MAX_AREA,
        OBSTACLE_MAX_VALUE,
        OBSTACLE_MIN_AREA,
        OBSTACLE_MIN_SATURATION,
        OBSTACLE_MIN_VALUE,
        OBSTACLE_SIZE_FULL_SCALE,
        OBSTACLE_TOP_IGNORE_RATIO,
        PING_PONG_MAX_AREA,
        PING_PONG_MIN_AREA,
        PING_PONG_MIN_CIRCULARITY,
        PING_PONG_PRESETS,
        PING_PONG_RADIUS_RANGE_PX,
        REPULSION_WEIGHT,
        SPECULAR_MAX_RATIO,
        SPECULAR_MAX_SATURATION,
        SPECULAR_MIN_RATIO,
        SPECULAR_MIN_VALUE,
        STEEL_MAX_AREA,
        STEEL_MAX_MEAN_VALUE,
        STEEL_MAX_SATURATION,
        STEEL_MAX_VALUE,
        STEEL_MIN_AREA,
        STEEL_MIN_CIRCULARITY,
        STEEL_MIN_VALUE,
        STEEL_RADIUS_RANGE_PX,
        TARGET_DISTANCE_FAR_CM,
        TARGET_DISTANCE_NEAR_CM,
    )


@dataclass(frozen=True, slots=True)
class CameraCalibration:
    camera_matrix: np.ndarray
    dist_coeffs: np.ndarray
    optimal_matrix: np.ndarray
    map1: np.ndarray
    map2: np.ndarray
    fx: float
    cx: float
    frame_size: tuple[int, int]


_CALIBRATION_CACHE: dict[tuple[int, int], CameraCalibration] = {}
_OPEN_KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, MASK_OPEN_KERNEL)
_CLOSE_KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, MASK_CLOSE_KERNEL)
_STEEL_MASK_MAX_SAT = min(STEEL_MAX_SATURATION, 50)
_STEEL_MASK_MIN_VAL = max(STEEL_MIN_VALUE, 30)
_STEEL_MASK_MAX_VAL = min(STEEL_MAX_VALUE, 150)
_STEEL_LOCAL_MAX_SAT = min(STEEL_MAX_SATURATION, 48)
_STEEL_LOCAL_MAX_MEAN_VAL = min(STEEL_MAX_MEAN_VALUE, 145.0)
_STEEL_MIN_CIRCULARITY_RELAXED = max(0.45, STEEL_MIN_CIRCULARITY - 0.16)
_REPULSION_GLOBAL_SCALE = 0.25
_RUNTIME_LOG_HEARTBEAT = 45
log = logging.getLogger(__name__)


def _get_calibration(frame_shape: tuple[int, int, int]) -> CameraCalibration:
    height, width = frame_shape[:2]
    frame_size = (width, height)
    cached = _CALIBRATION_CACHE.get(frame_size)
    if cached is not None:
        return cached

    optimal_matrix, _ = cv2.getOptimalNewCameraMatrix(
        CAMERA_MATRIX,
        DIST_COEFFS,
        frame_size,
        0.0,
    )
    map1, map2 = cv2.initUndistortRectifyMap(
        CAMERA_MATRIX,
        DIST_COEFFS,
        None,
        optimal_matrix,
        frame_size,
        cv2.CV_16SC2,
    )
    calibration = CameraCalibration(
        camera_matrix=CAMERA_MATRIX,
        dist_coeffs=DIST_COEFFS,
        optimal_matrix=optimal_matrix,
        map1=map1,
        map2=map2,
        fx=float(optimal_matrix[0, 0]),
        cx=float(optimal_matrix[0, 2]),
        frame_size=frame_size,
    )
    _CALIBRATION_CACHE[frame_size] = calibration
    return calibration


def set_ping_pong_profile(profile: str) -> str:
    if profile not in PING_PONG_PRESETS:
        valid = ", ".join(sorted(PING_PONG_PRESETS))
        raise ValueError(f"Unknown ping-pong profile '{profile}'. Expected one of: {valid}")
    return profile


def _undistort_frame(frame: np.ndarray, calibration: CameraCalibration) -> np.ndarray:
    return cv2.remap(frame, calibration.map1, calibration.map2, cv2.INTER_LINEAR)


def _build_mask(hsv: np.ndarray, ranges: tuple[HSVRange, ...]) -> np.ndarray:
    mask = cv2.inRange(hsv, ranges[0].lower, ranges[0].upper)
    for hsv_range in ranges[1:]:
        mask = cv2.bitwise_or(mask, cv2.inRange(hsv, hsv_range.lower, hsv_range.upper))
    return mask


def _clean_mask(mask: np.ndarray) -> np.ndarray:
    opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, _OPEN_KERNEL, iterations=1)
    return cv2.morphologyEx(opened, cv2.MORPH_CLOSE, _CLOSE_KERNEL, iterations=1)


def _contour_circularity(contour: np.ndarray) -> float:
    perimeter = cv2.arcLength(contour, True)
    if perimeter == 0:
        return 0.0
    area = cv2.contourArea(contour)
    return 4.0 * math.pi * area / (perimeter * perimeter)


def _estimate_distance_cm(radius_px: float, ball_type: str, calibration: CameraCalibration) -> float:
    apparent_diameter_px = max(radius_px * 2.0, 1e-6)
    return (BALL_DIAMETER_CM[ball_type] * calibration.fx) / apparent_diameter_px


def _bearing_deg_from_x(x_px: float, calibration: CameraCalibration) -> float:
    return math.degrees(math.atan2(x_px - calibration.cx, calibration.fx))


def _draw_ball_mask(mask: np.ndarray, ball: dict) -> None:
    cv2.circle(mask, (int(ball["x"]), int(ball["y"])), int(ball["radius"] * 1.2), 255, thickness=-1)


def _extract_ping_pong(
    hsv: np.ndarray,
    calibration: CameraCalibration,
    profile: str,
    accepted_ball_mask: np.ndarray,
) -> tuple[list[dict], np.ndarray]:
    profile_ranges = PING_PONG_PRESETS[profile]
    mask = _clean_mask(_build_mask(hsv, profile_ranges))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    detections: list[dict] = []
    min_radius, max_radius = PING_PONG_RADIUS_RANGE_PX
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < PING_PONG_MIN_AREA or area > PING_PONG_MAX_AREA:
            continue
        circularity = _contour_circularity(contour)
        if circularity < PING_PONG_MIN_CIRCULARITY:
            continue
        (cx, cy), radius = cv2.minEnclosingCircle(contour)
        if radius < min_radius or radius > max_radius:
            continue

        ball = {
            "type": "PingPong",
            "x": int(cx),
            "y": int(cy),
            "radius": float(radius),
            "distance": float(_estimate_distance_cm(radius, "PingPong", calibration)),
            "angle": float(_bearing_deg_from_x(cx, calibration)),
            "confidence": float(min(1.0, 0.55 + 0.45 * circularity)),
        }
        detections.append(ball)
        _draw_ball_mask(accepted_ball_mask, ball)

    return detections, mask


def _highlight_ratio(specular_mask: np.ndarray, cx: int, cy: int, radius: int) -> float:
    if radius <= 1:
        return 0.0
    y_grid, x_grid = np.ogrid[-radius: radius + 1, -radius: radius + 1]
    circle_mask = (x_grid * x_grid + y_grid * y_grid) <= radius * radius
    y1 = max(cy - radius, 0)
    y2 = min(cy + radius + 1, specular_mask.shape[0])
    x1 = max(cx - radius, 0)
    x2 = min(cx + radius + 1, specular_mask.shape[1])
    roi = specular_mask[y1:y2, x1:x2]
    local_circle_mask = circle_mask[: roi.shape[0], : roi.shape[1]]
    inside_circle = int(np.count_nonzero(local_circle_mask))
    if inside_circle == 0:
        return 0.0
    highlights = int(np.count_nonzero(roi[local_circle_mask]))
    return highlights / inside_circle


def _overlaps_existing(ball: dict, accepted: list[dict]) -> bool:
    for existing in accepted:
        centre_distance = math.hypot(ball["x"] - existing["x"], ball["y"] - existing["y"])
        if centre_distance <= max(ball["radius"], existing["radius"]):
            return True
    return False


def _extract_steel(
    hsv: np.ndarray,
    calibration: CameraCalibration,
    accepted_ball_mask: np.ndarray,
) -> tuple[list[dict], np.ndarray]:
    sat = hsv[:, :, 1]
    val = hsv[:, :, 2]

    grey_mask = cv2.inRange(
        hsv,
        np.array([0, 0, _STEEL_MASK_MIN_VAL], dtype=np.uint8),
        np.array([179, _STEEL_MASK_MAX_SAT, _STEEL_MASK_MAX_VAL], dtype=np.uint8),
    )
    specular_mask = cv2.inRange(
        hsv,
        np.array([0, 0, SPECULAR_MIN_VALUE], dtype=np.uint8),
        np.array([179, SPECULAR_MAX_SATURATION, 255], dtype=np.uint8),
    )
    working_mask = _clean_mask(grey_mask)
    contours, _ = cv2.findContours(working_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    min_radius, max_radius = STEEL_RADIUS_RANGE_PX
    accepted: list[dict] = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < STEEL_MIN_AREA or area > STEEL_MAX_AREA:
            continue
        circularity = _contour_circularity(contour)
        if circularity < _STEEL_MIN_CIRCULARITY_RELAXED:
            continue

        x, y, w, h = cv2.boundingRect(contour)
        bbox_area = max(w * h, 1)
        fill_ratio = area / float(bbox_area)
        if fill_ratio < 0.34:
            continue

        (cx, cy), radius = cv2.minEnclosingCircle(contour)
        if radius < min_radius or radius > max_radius:
            continue

        centre_x = int(cx)
        centre_y = int(cy)
        local_radius = max(int(radius), 2)
        ratio = _highlight_ratio(specular_mask, centre_x, centre_y, local_radius)
        if ratio > SPECULAR_MAX_RATIO:
            continue

        y1 = max(centre_y - local_radius, 0)
        y2 = min(centre_y + local_radius + 1, hsv.shape[0])
        x1 = max(centre_x - local_radius, 0)
        x2 = min(centre_x + local_radius + 1, hsv.shape[1])
        local_sat = float(np.mean(sat[y1:y2, x1:x2]))
        local_val = float(np.mean(val[y1:y2, x1:x2]))
        if local_sat > _STEEL_LOCAL_MAX_SAT or local_val > _STEEL_LOCAL_MAX_MEAN_VAL:
            continue

        ball = {
            "type": "steel",
            "x": centre_x,
            "y": centre_y,
            "radius": float(radius),
            "distance": float(_estimate_distance_cm(radius, "steel", calibration)),
            "angle": float(_bearing_deg_from_x(cx, calibration)),
            "confidence": float(min(1.0, 0.55 + 0.30 * circularity + 0.25 * min(1.0, ratio / max(SPECULAR_MIN_RATIO, 1e-6)))),
            "specular_ratio": float(ratio),
        }
        if _overlaps_existing(ball, accepted):
            continue
        accepted.append(ball)
        _draw_ball_mask(accepted_ball_mask, ball)

    return accepted, working_mask


def _prioritize_balls(detections: list[dict]) -> list[dict]:
    return sorted(
        detections,
        key=lambda item: (BALL_TYPE_PRIORITY[item["type"]], item["distance"], abs(item["angle"])),
    )


def detect_balls(frame: np.ndarray, ping_pong_profile: str = DEFAULT_PING_PONG_PROFILE) -> dict:
    if frame is None or frame.ndim != 3:
        raise ValueError("detect_balls expects a BGR frame with shape (H, W, 3)")

    profile = set_ping_pong_profile(ping_pong_profile)
    calibration = _get_calibration(frame.shape)
    undistorted = _undistort_frame(frame, calibration)
    blurred = cv2.GaussianBlur(undistorted, FRAME_GAUSSIAN_BLUR, 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    accepted_ball_mask = np.zeros(hsv.shape[:2], dtype=np.uint8)

    ping_pong, ping_pong_mask = _extract_ping_pong(hsv, calibration, profile, accepted_ball_mask)
    steel, steel_mask = _extract_steel(hsv, calibration, accepted_ball_mask)
    detections = _prioritize_balls(steel + ping_pong)

    return {
        "frame": undistorted,
        "calibration": calibration,
        "detected_balls": detections,
        "ball_mask": accepted_ball_mask,
        "ping_pong_mask": ping_pong_mask,
        "steel_mask": steel_mask,
    }


def detect_obstacles(frame: np.ndarray, ball_mask: np.ndarray) -> list[dict]:
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    sat = hsv[:, :, 1]
    val = hsv[:, :, 2]

    obstacle_mask = np.zeros_like(val)
    obstacle_mask[(val >= OBSTACLE_MIN_VALUE) & (val <= OBSTACLE_MAX_VALUE)] = 255
    obstacle_mask[sat < OBSTACLE_MIN_SATURATION] = 0
    obstacle_mask[ball_mask > 0] = 0

    top_ignore = int(frame.shape[0] * OBSTACLE_TOP_IGNORE_RATIO)
    obstacle_mask[:top_ignore, :] = 0
    obstacle_mask = _clean_mask(obstacle_mask)

    contours, _ = cv2.findContours(obstacle_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    obstacles: list[dict] = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < OBSTACLE_MIN_AREA or area > OBSTACLE_MAX_AREA:
            continue
        x, y, width, height = cv2.boundingRect(contour)
        moment = cv2.moments(contour)
        if moment["m00"] == 0:
            centre_x = x + width // 2
            centre_y = y + height // 2
        else:
            centre_x = int(moment["m10"] / moment["m00"])
            centre_y = int(moment["m01"] / moment["m00"])
        obstacles.append(
            {
                "x": centre_x,
                "y": centre_y,
                "size": float(math.sqrt(area)),
                "area": float(area),
                "bbox": (x, y, width, height),
            }
        )
    return obstacles


def is_ball_grabbable(ball: dict, calibration: CameraCalibration) -> bool:
    max_distance = GRAB_DISTANCE_CM[ball["type"]]
    return abs(ball["angle"]) <= GRAB_MAX_ABS_ANGLE_DEG and ball["distance"] <= max_distance


def _distance_to_strength(distance_cm: float) -> float:
    clipped = min(max(distance_cm, TARGET_DISTANCE_NEAR_CM), TARGET_DISTANCE_FAR_CM)
    span = max(TARGET_DISTANCE_FAR_CM - TARGET_DISTANCE_NEAR_CM, 1.0)
    strength = 1.0 - ((clipped - TARGET_DISTANCE_NEAR_CM) / span)
    return float(np.clip(strength, 0.0, 1.0))


def _vector_from_angle(angle_deg: float, strength: float) -> np.ndarray:
    radians = math.radians(angle_deg)
    return np.array([math.sin(radians), math.cos(radians)], dtype=np.float32) * strength


def compute_navigation_vector(
    target_ball: dict | None,
    obstacles: list[dict],
    frame_shape: tuple[int, int, int],
    calibration: CameraCalibration,
) -> dict | None:
    if target_ball is None:
        return None

    attraction_strength = max(MIN_TRACKING_MAGNITUDE, _distance_to_strength(target_ball["distance"]))
    resultant = _vector_from_angle(target_ball["angle"], attraction_strength * ATTRACTION_WEIGHT)

    frame_height = float(frame_shape[0])
    for obstacle in obstacles:
        obstacle_angle = _bearing_deg_from_x(obstacle["x"], calibration)
        size_scale = min(obstacle["size"] / OBSTACLE_SIZE_FULL_SCALE, 1.0)
        bottom_bias = min(max(obstacle["y"] / frame_height, 0.0), 1.0)
        centrality = max(0.0, 1.0 - min(abs(obstacle_angle) / 90.0, 1.0))
        strength = (
            REPULSION_WEIGHT
            * _REPULSION_GLOBAL_SCALE
            * size_scale
            * (0.15 + 0.25 * bottom_bias)
            * (0.20 + 0.30 * centrality)
        )
        resultant -= _vector_from_angle(obstacle_angle, strength)

    lateral = float(resultant[0])
    forward = float(resultant[1])
    angle = math.degrees(math.atan2(lateral, forward))
    magnitude = float(np.clip(np.linalg.norm(resultant), 0.0, 1.0))
    return {"angle": angle, "magnitude": magnitude}


def process_ball_frame(frame: np.ndarray, ping_pong_profile: str = DEFAULT_PING_PONG_PROFILE) -> dict:
    if not hasattr(process_ball_frame, "_log_state"):
        process_ball_frame._log_state = None
        process_ball_frame._log_counter = 0
        process_ball_frame._last_log_counter = -10_000

    detection = detect_balls(frame, ping_pong_profile=ping_pong_profile)
    calibration: CameraCalibration = detection["calibration"]
    detected_balls: list[dict] = detection["detected_balls"]
    target_ball = detected_balls[0] if detected_balls else None
    obstacles = detect_obstacles(detection["frame"], detection["ball_mask"])

    locked_ball = ""
    move_target = None
    grab_ready = False

    if target_ball is not None:
        grab_ready = is_ball_grabbable(target_ball, calibration)
        if grab_ready:
            locked_ball = target_ball["type"]
        else:
            move_target = compute_navigation_vector(target_ball, obstacles, detection["frame"].shape, calibration)

    process_ball_frame._log_counter += 1
    if target_ball is None:
        runtime_state = "SEARCHING"
        state_detail = "no targets"
    elif grab_ready:
        runtime_state = f"CAPTURING {target_ball['type']}"
        state_detail = f"dist={target_ball['distance']:.1f}cm"
    elif move_target is not None:
        runtime_state = f"TRACKING {target_ball['type']}"
        state_detail = f"angle={move_target['angle']:.1f}deg mag={move_target['magnitude']:.2f}"
    else:
        runtime_state = "TRACKING"
        state_detail = "nav=none"

    if (
        runtime_state != process_ball_frame._log_state
        or (process_ball_frame._log_counter - process_ball_frame._last_log_counter) >= _RUNTIME_LOG_HEARTBEAT
    ):
        log.info(f"CV runtime state: {runtime_state} | {state_detail}")
        process_ball_frame._log_state = runtime_state
        process_ball_frame._last_log_counter = process_ball_frame._log_counter

    return {
        "detected_balls": detected_balls,
        "target_ball": target_ball,
        "obstacles": obstacles,
        "move_target": move_target,
        "locked_ball": locked_ball,
        "grab_ready": grab_ready,
        "camera_fx": calibration.fx,
        "camera_cx": calibration.cx,
    }


def benchmark_pipeline(frame: np.ndarray, runs: int = 30) -> dict:
    if runs <= 0:
        raise ValueError("runs must be positive")
    durations_ms: list[float] = []
    for _ in range(runs):
        start = time.perf_counter()
        process_ball_frame(frame)
        durations_ms.append((time.perf_counter() - start) * 1000.0)
    samples = np.array(durations_ms, dtype=np.float32)
    return {
        "runs": runs,
        "mean_ms": float(samples.mean()),
        "median_ms": float(np.median(samples)),
        "p95_ms": float(np.percentile(samples, 95)),
    }
