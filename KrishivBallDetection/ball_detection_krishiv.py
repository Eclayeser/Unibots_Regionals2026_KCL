"""
Ball Detection Module — UniBots Competition
============================================
Detects ping-pong balls and steel ball bearings from 640×480 BGR
webcam frames using HSV colour filtering and contour analysis.

Runs as a multiprocessing worker on a Raspberry Pi 5 at ≥ 15 FPS.

Public API
----------
    detect_ball(frame, target="all")  -> dict
    detection_worker(in_q, out_q)     -> None   (multiprocessing target)

Output dict
-----------
    x              : float  — normalised horizontal centre [-1, 1]
    z              : float  — estimated distance from lens (cm)
    classification : str    — "ping_pong" | "steel" | "unidentified"
"""

from __future__ import annotations

import math
from dataclasses import dataclass
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
    [[400.0,   0.0, 320.0],
     [  0.0, 400.0, 240.0],
     [  0.0,   0.0,   1.0]], dtype=np.float64,
)
DIST_COEFFS = np.array([-0.35, 0.12, 0.0, 0.0, 0.0], dtype=np.float64)

# Pre-compute undistortion maps (one-off; ~1 ms per remap at runtime)
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
# Hue 0–179, Sat/Val 0–255 in OpenCV.

PING_PONG_PRESETS: dict[str, list[HSVRange]] = {
    "orange": [HSVRange(np.array([5, 120, 120]),  np.array([25, 255, 255]))],
    "white":  [HSVRange(np.array([0,   0, 200]),  np.array([179, 50, 255]))],
}

# >>> Change this before the match <<<
PING_PONG_COLOUR: str = "orange"
PING_PONG_RANGES: list[HSVRange] = PING_PONG_PRESETS[PING_PONG_COLOUR]

# Steel ball bearings — tighter ranges to reduce false positives:
#   • Very low saturation (grey, not coloured)
#   • Moderate value band  (not too dark = shadows, not too bright = specular)
#   • Higher circularity required (see STEEL_CONTOUR below)
STEEL_RANGES: list[HSVRange] = [
    HSVRange(np.array([0, 0,  45]), np.array([179, 35, 120])),
]

# Specular-highlight value threshold (V > this → zeroed before steel detection)
SPECULAR_V_THRESH: int = 200

# Contour limits — tighter for steel to reject non-ball grey blobs
PING_PONG_CONTOUR = ContourLimits(min_area=100,  max_area=80_000, min_circularity=0.55)
STEEL_CONTOUR     = ContourLimits(min_area=60,   max_area=30_000, min_circularity=0.70)

# ===================================================================
# Hough Circle detection (shape-based fallback / confirmation)
# ===================================================================
# cv2.HoughCircles params — tuned for 640×480 with balls at 0.3–3 m.
# dp=1.2  : accumulator resolution ratio (slightly coarser = faster)
# minDist  : minimum px between detected circle centres
# param1   : Canny high threshold (lower = more edges, more circles)
# param2   : accumulator vote threshold (lower = more false positives)
# minRadius/maxRadius : expected ball pixel radius range
HOUGH_DP          = 1.2
HOUGH_MIN_DIST    = 50
HOUGH_PARAM1      = 100
HOUGH_PARAM2      = 45
HOUGH_MIN_RADIUS  = 12
HOUGH_MAX_RADIUS  = 180

# Maximum px distance between an HSV contour centre and a Hough circle
# centre for them to be considered the same ball (confirmation match).
HOUGH_MATCH_DIST  = 50

# ===================================================================
# Reusable kernels (allocated once)
# ===================================================================

_BLUR_KSIZE = (7, 7)
_MORPH_KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

# ===================================================================
# Internal helpers
# ===================================================================

def undistort(frame: np.ndarray) -> np.ndarray:
    """Apply pre-computed lens distortion correction."""
    return cv2.remap(frame, UNDISTORT_MAP1, UNDISTORT_MAP2, cv2.INTER_LINEAR)


def build_mask(hsv: np.ndarray, ranges: list[HSVRange]) -> np.ndarray:
    """Union of cv2.inRange masks for each HSVRange."""
    mask = cv2.inRange(hsv, ranges[0].lower, ranges[0].upper)
    for r in ranges[1:]:
        mask = cv2.bitwise_or(mask, cv2.inRange(hsv, r.lower, r.upper))
    return mask


def suppress_specular(hsv: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Zero bright-pixel regions that would otherwise fragment steel contours."""
    mask[hsv[:, :, 2] > SPECULAR_V_THRESH] = 0
    return mask


def morph_clean(mask: np.ndarray) -> np.ndarray:
    """Open then close to remove noise and fill small holes."""
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  _MORPH_KERNEL, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, _MORPH_KERNEL, iterations=1)
    return mask


def all_contours(
    mask: np.ndarray,
    limits: ContourLimits,
) -> list[tuple[np.ndarray, float, tuple[int, int], float]]:
    """Return *all* sufficiently-circular blobs sorted largest-first.

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


def best_contour(
    mask: np.ndarray,
    limits: ContourLimits,
) -> Optional[tuple[np.ndarray, float, tuple[int, int], float]]:
    """Return (contour, area, (cx, cy), radius) of the largest
    sufficiently-circular blob, or None."""
    hits = all_contours(mask, limits)
    return hits[0] if hits else None


def estimate_distance(apparent_diam_px: float, real_diam_cm: float) -> float:
    """Pinhole-model distance: z = (D_real × f) / D_apparent."""
    if apparent_diam_px <= 0:
        return float("inf")
    return (real_diam_cm * FOCAL_PX) / apparent_diam_px


# ===================================================================
# Hough Circle detection
# ===================================================================

def find_circles(gray: np.ndarray) -> list[tuple[int, int, int]]:
    """Run HoughCircles on a grayscale image.

    Returns a list of (cx, cy, radius) tuples, sorted largest-first.
    """
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
    # Post-filter: verify each circle has sufficient edge support
    out: list[tuple[int, int, int]] = []
    edges = cv2.Canny(gray, HOUGH_PARAM1 // 2, HOUGH_PARAM1) if raw else None
    h, w = gray.shape[:2]
    for cx, cy, cr in raw:
        # Sample 24 points on the circle perimeter, count those on an edge
        hits = 0
        samples = 24
        for i in range(samples):
            angle = 2.0 * math.pi * i / samples
            px = int(cx + cr * math.cos(angle))
            py = int(cy + cr * math.sin(angle))
            if 0 <= px < w and 0 <= py < h and edges[py, px] > 0:
                hits += 1
        # require ≥ 35% of sampled perimeter pixels to be actual edges
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
# Core detection
# ===================================================================

def detect_ball(
    frame: np.ndarray,
    target: str = "all",
) -> dict:
    """Detect the largest ball in a 640×480 BGR frame.

    Uses HSV colour filtering as the primary method and Hough Circle
    detection as confirmation / fallback.

    Parameters
    ----------
    frame  : BGR uint8 ndarray, 640×480.
    target : "all" | "ping_pong" | "steel"
             When not "all", only that ball type is searched.

    Returns
    -------
    dict with keys:
        x              : float
        z              : float
        classification : str
        confirmed      : bool  — True if HSV hit is backed by a Hough circle
        method         : str   — "hsv+shape" | "hsv" | "shape" | "none"
    """
    frame = undistort(frame)
    h, w = frame.shape[:2]

    blurred = cv2.GaussianBlur(frame, _BLUR_KSIZE, 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)

    # --- Hough circles (computed once, shared across targets) ---
    circles = find_circles(gray)

    # --- HSV candidates ---
    candidates: list[tuple[str, float, tuple[int, int], float, float, bool]] = []

    if target in ("all", "ping_pong"):
        pp_mask = morph_clean(build_mask(hsv, PING_PONG_RANGES))
        pp = best_contour(pp_mask, PING_PONG_CONTOUR)
        if pp is not None:
            _, area, centre, radius = pp
            confirmed = _match_circle_to_contour(centre, circles) is not None
            candidates.append(("ping_pong", area, centre, radius, PING_PONG_DIAMETER_CM, confirmed))

    if target in ("all", "steel"):
        st_mask = morph_clean(suppress_specular(hsv, build_mask(hsv, STEEL_RANGES)))
        st = best_contour(st_mask, STEEL_CONTOUR)
        if st is not None:
            _, area, centre, radius = st
            confirmed = _match_circle_to_contour(centre, circles) is not None
            candidates.append(("steel", area, centre, radius, STEEL_BALL_DIAMETER_CM, confirmed))

    # --- pick best HSV candidate (prefer confirmed over unconfirmed) ---
    if candidates:
        # sort: confirmed first, then by area
        candidates.sort(key=lambda c: (c[5], c[1]), reverse=True)
        cls, _, (cx, _cy), radius, real_d, conf = candidates[0]
        return {
            "x": round((cx - w / 2) / (w / 2), 4),
            "z": round(estimate_distance(radius * 2, real_d), 1),
            "classification": cls,
            "confirmed": conf,
            "method": "hsv+shape" if conf else "hsv",
        }

    # --- fallback: Hough circles only (no colour match) ---
    if circles:
        hx, hy, hr = circles[0]  # largest circle
        # guess classification from radius — respect target filter
        if hr > 15:
            guess_cls, real_d = "ping_pong", PING_PONG_DIAMETER_CM
        else:
            guess_cls, real_d = "steel", STEEL_BALL_DIAMETER_CM
        # skip if the guess doesn't match the requested target
        if target != "all" and guess_cls != target:
            pass  # fall through to unidentified
        else:
            return {
                "x": round((hx - w / 2) / (w / 2), 4),
                "z": round(estimate_distance(hr * 2, real_d), 1),
                "classification": guess_cls,
                "confirmed": False,
                "method": "shape",
            }

    return {
        "x": 0.0, "z": float("inf"), "classification": "unidentified",
        "confirmed": False, "method": "none",
    }


# ===================================================================
# Object-tracking helpers  (detect → track → periodic re-detect)
# ===================================================================

TRACKER_TYPES = ("KCF", "CSRT")


def create_tracker(tracker_type: str = "KCF"):
    """Create an OpenCV object tracker.

    KCF  — fast, good for real-time on Raspberry Pi  (default)
    CSRT — more accurate, higher CPU cost
    """
    t = tracker_type.upper()
    if t == "KCF":
        return cv2.TrackerKCF.create()
    if t == "CSRT":
        return cv2.TrackerCSRT.create()
    raise ValueError(f"Unknown tracker type: {t!r}  (use {TRACKER_TYPES})")


def detection_to_bbox(
    centre: tuple[int, int],
    radius: float,
    frame_hw: tuple[int, ...] = (480, 640),
) -> tuple[int, int, int, int]:
    """Convert (centre, radius) to (x, y, w, h) bounding box for tracker init.

    The bbox is clipped to the frame boundaries.
    """
    cx, cy = centre
    r = max(int(radius), 1)
    x = max(cx - r, 0)
    y = max(cy - r, 0)
    w = min(2 * r, frame_hw[1] - x)
    h = min(2 * r, frame_hw[0] - y)
    return (x, y, w, h)


def bbox_to_centre(bbox) -> tuple[tuple[int, int], float]:
    """Convert (x, y, w, h) tracker output to ((cx, cy), radius)."""
    x, y, w, h = (int(v) for v in bbox)
    return (x + w // 2, y + h // 2), (w + h) / 4.0


# ===================================================================
# Multiprocessing worker
# ===================================================================

def detection_worker(frame_queue, result_queue) -> None:
    """Pull frames → detect → push results.  Send None to stop."""
    while True:
        frame = frame_queue.get()
        if frame is None:
            break
        result = detect_ball(frame)
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

    # warm-up
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
