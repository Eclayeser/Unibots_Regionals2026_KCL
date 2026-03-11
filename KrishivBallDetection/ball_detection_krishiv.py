"""
Ball Detection Module — UniBots Competition
============================================
Detects ALL visible ping-pong balls and steel ball bearings from
640×480 BGR webcam frames using HSV colour filtering, contour
analysis, Hough circle confirmation, and specular highlight checks.

Pipeline
--------
    Ping-pong : HSV colour mask  →  contour  →  Hough circle (confirm)
    Steel     : HSV colour mask  →  contour  →  Hough circle (confirm)  →  specular (check)

Public API
----------
    detect_balls(frame, target="all") -> list[dict]
    BallTracker                       -> class  (persistent IDs across frames)
    detection_worker(in_q, out_q)     -> None   (multiprocessing target)

Output dict per ball
--------------------
    id        : int    — persistent tracking ID (assigned by BallTracker)
    type      : str    — "ping_pong" | "steel"
    x         : int    — pixel x centre in frame
    y         : int    — pixel y centre in frame
    radius    : float  — apparent pixel radius
    distance  : float  — estimated distance from lens (cm), pinhole model
    confirmed : bool   — True if Hough circle backs up the HSV detection
    method    : str    — "hsv+circle+specular" | "hsv+circle" | "hsv"
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np

# ===================================================================
# Configuration dataclasses
# ===================================================================

@dataclass(frozen=True, slots=True)
class HSVRange:
    """A single lower/upper HSV bound pair."""
    lower: np.ndarray
    upper: np.ndarray

@dataclass(frozen=True, slots=True)
class ContourLimits:
    """Area and circularity thresholds for contour filtering."""
    min_area: int      = 80
    max_area: int      = 80_000
    min_circularity: float = 0.55

# ===================================================================
# Camera / lens calibration  (update after checkerboard calibration)
# ===================================================================

CAMERA_MATRIX = np.array(
    [[500, 0.0, 320.0],
     [0.0, 500, 240.0],
     [0.0, 0.0, 1.0]], dtype=np.float64,
)
DIST_COEFFS = np.array([0.000000, 0.000000, 0.000000, 0.000000, 0.000000], dtype=np.float64)


_NEW_CAM, _ROI = cv2.getOptimalNewCameraMatrix(
    CAMERA_MATRIX, DIST_COEFFS, (640, 480), alpha=0.0,
)
UNDISTORT_MAP1, UNDISTORT_MAP2 = cv2.initUndistortRectifyMap(
    CAMERA_MATRIX, DIST_COEFFS, None, _NEW_CAM, (640, 480), cv2.CV_16SC2,
)

FOCAL_PX: float = _NEW_CAM[0, 0]

# ===================================================================
# Ball physical properties
# ===================================================================

PING_PONG_DIAMETER_CM: float = 4.0   # 40 mm
STEEL_BALL_DIAMETER_CM: float = 2.0  # 20 mm

# ===================================================================
# HSV colour presets
# ===================================================================

PING_PONG_PRESETS: dict[str, list[HSVRange]] = {
    "orange": [HSVRange(np.array([5, 120, 120]),  np.array([25, 255, 255]))],
    "white":  [HSVRange(np.array([0,   0, 200]),  np.array([179, 50, 255]))],
}

# >>> Change this before the match <<<
PING_PONG_COLOUR: str = "orange"
PING_PONG_RANGES: list[HSVRange] = PING_PONG_PRESETS[PING_PONG_COLOUR]

STEEL_RANGES: list[HSVRange] = [
    HSVRange(np.array([0, 0,  45]), np.array([179, 35, 120])),
]

# Specular highlight thresholds — used for steel ball verification.
# A steel ball should have at least one bright specular spot inside it.
SPECULAR_V_THRESH: int = 200
SPECULAR_S_MAX: int = 50
SPECULAR_MIN_PIXELS: int = 3   # minimum bright pixels inside contour mask

# Contour limits
PING_PONG_CONTOUR = ContourLimits(min_area=100,  max_area=80_000, min_circularity=0.55)
STEEL_CONTOUR     = ContourLimits(min_area=60,   max_area=30_000, min_circularity=0.70)

# ===================================================================
# Hough Circle detection params
# ===================================================================

HOUGH_DP          = 1.2
HOUGH_MIN_DIST    = 50
HOUGH_PARAM1      = 100
HOUGH_PARAM2      = 45
HOUGH_MIN_RADIUS  = 12
HOUGH_MAX_RADIUS  = 180
HOUGH_MATCH_DIST  = 50

# ===================================================================
# Reusable kernels
# ===================================================================

_BLUR_KSIZE = (7, 7)
_MORPH_KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

# ===================================================================
# Internal helpers
# ===================================================================

def undistort(frame: np.ndarray) -> np.ndarray:
    return cv2.remap(frame, UNDISTORT_MAP1, UNDISTORT_MAP2, cv2.INTER_LINEAR)


def build_mask(hsv: np.ndarray, ranges: list[HSVRange]) -> np.ndarray:
    mask = cv2.inRange(hsv, ranges[0].lower, ranges[0].upper)
    for r in ranges[1:]:
        mask = cv2.bitwise_or(mask, cv2.inRange(hsv, r.lower, r.upper))
    return mask


def morph_clean(mask: np.ndarray) -> np.ndarray:
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  _MORPH_KERNEL, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, _MORPH_KERNEL, iterations=1)
    return mask


def all_contours(
    mask: np.ndarray,
    limits: ContourLimits,
) -> list[tuple[np.ndarray, float, tuple[int, int], float]]:
    """Return all sufficiently-circular blobs sorted largest-first.
    Each element: (contour, area, (cx, cy), radius).
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    results: list[tuple[np.ndarray, float, tuple[int, int], float]] = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < limits.min_area or area > limits.max_area:
            continue
        perim = cv2.arcLength(cnt, True)
        if perim == 0:
            continue
        circ = 4.0 * math.pi * area / (perim * perim)
        if circ < limits.min_circularity:
            continue
        (cx, cy), radius = cv2.minEnclosingCircle(cnt)
        results.append((cnt, area, (int(cx), int(cy)), radius))
    results.sort(key=lambda c: c[1], reverse=True)
    return results


def estimate_distance(apparent_diam_px: float, real_diam_cm: float) -> float:
    """Pinhole-model distance: z = (D_real * f) / D_apparent."""
    if apparent_diam_px <= 0:
        return float("inf")
    return (real_diam_cm * FOCAL_PX) / apparent_diam_px


# ===================================================================
# Hough Circle detection
# ===================================================================

def find_circles(gray: np.ndarray) -> list[tuple[int, int, int]]:
    circles = cv2.HoughCircles(
        gray,
        cv2.HOUGH_GRADIENT,
        dp=HOUGH_DP,
        minDist=HOUGH_MIN_DIST,
        param1=HOUGH_PARAM1,
        param2=HOUGH_PARAM2,
        minRadius=HOUGH_MIN_RADIUS,
        maxRadius=HOUGH_MAX_RADIUS,
    )
    if circles is None:
        return []
    raw = [(int(x), int(y), int(r)) for x, y, r in circles[0]]
    out: list[tuple[int, int, int]] = []
    edges = cv2.Canny(gray, HOUGH_PARAM1 // 2, HOUGH_PARAM1) if raw else None
    h, w = gray.shape[:2]
    for cx, cy, cr in raw:
        hits = 0
        samples = 24
        for i in range(samples):
            angle = 2.0 * math.pi * i / samples
            px = int(cx + cr * math.cos(angle))
            py = int(cy + cr * math.sin(angle))
            if 0 <= px < w and 0 <= py < h and edges[py, px] > 0:
                hits += 1
        if hits / samples >= 0.35:
            out.append((cx, cy, cr))
    out.sort(key=lambda c: c[2], reverse=True)
    return out


def _match_circle_to_contour(
    contour_centre: tuple[int, int],
    circles: list[tuple[int, int, int]],
) -> Optional[tuple[int, int, int]]:
    """Find the Hough circle closest to a contour centre (within threshold)."""
    cx, cy = contour_centre
    best, best_d = None, HOUGH_MATCH_DIST
    for hx, hy, hr in circles:
        d = math.hypot(hx - cx, hy - cy)
        if d < best_d:
            best_d = d
            best = (hx, hy, hr)
    return best


# ===================================================================
# Specular highlight check (steel ball verification)
# ===================================================================

def _has_specular(hsv: np.ndarray, contour: np.ndarray, img_shape: tuple) -> bool:
    """Check if contour region contains bright specular highlights.
    Steel ball bearings are shiny and should have at least a small
    cluster of very bright, low-saturation pixels inside.
    """
    mask = np.zeros(img_shape[:2], dtype=np.uint8)
    cv2.drawContours(mask, [contour], -1, 255, -1)
    bright = (hsv[:, :, 2] > SPECULAR_V_THRESH) & (hsv[:, :, 1] < SPECULAR_S_MAX)
    return int(np.count_nonzero(bright & (mask > 0))) >= SPECULAR_MIN_PIXELS


# ===================================================================
# Core detection — returns ALL detected balls
# ===================================================================

def detect_balls(
    frame: np.ndarray,
    target: str = "all",
) -> list[dict]:
    """Detect all balls in a 640x480 BGR frame.

    Pipeline per user spec:
        Ping-pong : HSV mask → contour → Hough circle (confirm)
        Steel     : HSV mask → contour → Hough circle (confirm) → specular (check)

    Parameters
    ----------
    frame  : BGR uint8 ndarray, 640x480.
    target : "all" | "ping_pong" | "steel"

    Returns
    -------
    list[dict], each with keys:
        type      : str    — "ping_pong" | "steel"
        x         : int    — pixel x centre
        y         : int    — pixel y centre
        radius    : float  — apparent pixel radius
        distance  : float  — estimated cm from camera
        confirmed : bool   — Hough circle backed
        method    : str    — detection method description
    Note: 'id' is NOT set here — use BallTracker.update() to assign IDs.
    """
    frame = undistort(frame)
    blurred = cv2.GaussianBlur(frame, _BLUR_KSIZE, 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)

    circles = find_circles(gray)
    detections: list[dict] = []

    # --- Ping-pong: HSV → contour → circle confirm ---
    if target in ("all", "ping_pong"):
        pp_mask = morph_clean(build_mask(hsv, PING_PONG_RANGES))
        for cnt, area, (cx, cy), radius in all_contours(pp_mask, PING_PONG_CONTOUR):
            confirmed = _match_circle_to_contour((cx, cy), circles) is not None
            method = "hsv+circle" if confirmed else "hsv"
            detections.append({
                "type": "ping_pong",
                "x": cx,
                "y": cy,
                "radius": round(radius, 1),
                "distance": round(estimate_distance(radius * 2, PING_PONG_DIAMETER_CM), 1),
                "confirmed": confirmed,
                "method": method,
            })

    # --- Steel: HSV → contour → circle confirm → specular check ---
    if target in ("all", "steel"):
        st_mask = morph_clean(build_mask(hsv, STEEL_RANGES))
        for cnt, area, (cx, cy), radius in all_contours(st_mask, STEEL_CONTOUR):
            confirmed = _match_circle_to_contour((cx, cy), circles) is not None
            has_spec = _has_specular(hsv, cnt, frame.shape)
            if confirmed and has_spec:
                method = "hsv+circle+specular"
            elif confirmed:
                method = "hsv+circle"
            elif has_spec:
                method = "hsv+specular"
            else:
                method = "hsv"
            detections.append({
                "type": "steel",
                "x": cx,
                "y": cy,
                "radius": round(radius, 1),
                "distance": round(estimate_distance(radius * 2, STEEL_BALL_DIAMETER_CM), 1),
                "confirmed": confirmed,
                "method": method,
            })

    return detections


# ===================================================================
# BallTracker — persistent IDs across frames via nearest-neighbour
# ===================================================================

class BallTracker:
    """Assigns stable integer IDs to balls across consecutive frames.

    Uses greedy nearest-neighbour matching on (x, y) pixel position.
    A ball that disappears for more than `max_lost` frames loses its ID.
    """

    def __init__(self, max_lost: int = 15, match_radius: int = 80):
        self._next_id: int = 1
        self._max_lost: int = max_lost
        self._match_radius: int = match_radius
        # tracked[id] = {"type", "x", "y", "radius", "distance", "confirmed", "method", "id", "lost"}
        self._tracked: dict[int, dict] = {}

    def reset(self):
        self._tracked.clear()
        self._next_id = 1

    def update(self, detections: list[dict]) -> list[dict]:
        """Match new detections against tracked balls, assign IDs.

        Returns a new list of dicts — same fields as detect_balls() but
        with an 'id' key added.
        """
        used_track_ids: set[int] = set()
        used_det_indices: set[int] = set()
        result: list[dict] = []

        # Build distance pairs: (dist, track_id, det_index)
        pairs: list[tuple[float, int, int]] = []
        for tid, tb in self._tracked.items():
            for di, det in enumerate(detections):
                d = math.hypot(det["x"] - tb["x"], det["y"] - tb["y"])
                if d < self._match_radius and det["type"] == tb["type"]:
                    pairs.append((d, tid, di))

        pairs.sort(key=lambda p: p[0])

        # Greedy match
        for d, tid, di in pairs:
            if tid in used_track_ids or di in used_det_indices:
                continue
            used_track_ids.add(tid)
            used_det_indices.add(di)
            det = detections[di]
            entry = {**det, "id": tid}
            self._tracked[tid] = {**entry, "lost": 0}
            result.append(entry)

        # New balls — unmatched detections get fresh IDs
        for di, det in enumerate(detections):
            if di in used_det_indices:
                continue
            tid = self._next_id
            self._next_id += 1
            entry = {**det, "id": tid}
            self._tracked[tid] = {**entry, "lost": 0}
            result.append(entry)

        # Age out unmatched tracked balls
        lost_ids = []
        for tid in self._tracked:
            if tid not in used_track_ids:
                self._tracked[tid]["lost"] += 1
                if self._tracked[tid]["lost"] > self._max_lost:
                    lost_ids.append(tid)
        for tid in lost_ids:
            del self._tracked[tid]

        return result


# ===================================================================
# Multiprocessing worker
# ===================================================================

def detection_worker(frame_queue, result_queue) -> None:
    """Pull frames → detect → push results.  Send None to stop."""
    tracker = BallTracker()
    while True:
        frame = frame_queue.get()
        if frame is None:
            break
        raw = detect_balls(frame)
        result = tracker.update(raw)
        if result_queue.full():
            try:
                result_queue.get_nowait()
            except Exception:
                pass
        result_queue.put(result)


# ===================================================================
# Self-test / benchmark
# ===================================================================

if __name__ == "__main__":
    from multiprocessing import Process, Queue
    import time

    dummy = np.zeros((480, 640, 3), dtype=np.uint8)
    fq: Queue = Queue(maxsize=2)
    rq: Queue = Queue(maxsize=2)

    proc = Process(target=detection_worker, args=(fq, rq))
    proc.start()

    fq.put(dummy)
    rq.get(timeout=5)

    N = 100
    t0 = time.perf_counter()
    for _ in range(N):
        fq.put(dummy)
        rq.get(timeout=5)
    ms = (time.perf_counter() - t0) / N * 1000

    print(f"Latency : {ms:.1f} ms/frame")
    print(f"FPS     : {1000 / ms:.0f}")

    fq.put(None)
    proc.join(timeout=5)
    print("OK" if not proc.is_alive() else "Worker hung")
