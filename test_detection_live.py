"""
Live Ball Detection — Test Harness
===================================
Opens the default webcam and runs the detection pipeline with a
real-time annotated preview.

Controls
--------
    Q / ESC         quit
    O               orange ping-pong mode
    W               white  ping-pong mode
    1               detect ping-pong only
    2               detect steel only
    3               detect all (default)
    D               toggle detection overlay on/off (works on ALL stages)
    T               toggle detect-then-track mode on/off
    R               reset tracker — forget current ball and re-search
    LEFT / RIGHT    cycle pipeline stage view

Pipeline stages (arrow keys)
-----------------------------
    0  Raw camera
    1  Undistorted
    2  Gaussian blur
    3  HSV (viewable)
    4  Active mask
"""

from __future__ import annotations

import math
import time
from enum import Enum

import cv2
import numpy as np

import ball_detection_krishiv as bd

# ---- constants -------------------------------------------------------

WINDOW = "UniBots — Ball Detection"
FONT   = cv2.FONT_HERSHEY_SIMPLEX

STAGE_LABELS = (
    "0: Raw camera",
    "1: Undistorted",
    "2: Gaussian blur",
    "3: HSV",
    "4: Active mask",
    "5: Hough circles",
)
N_STAGES = len(STAGE_LABELS)

# Drawing
_XHAIR = 28
_GAP   = 5
_COL   = {
    "ping_pong": (0, 200, 255),
    "steel":     (200, 200, 200),
    "ghost":     (0, 100, 180),
    "none":      (80, 80, 80),
}

# Arrow key codes  (waitKeyEx)
_KEY_RIGHT_WIN, _KEY_RIGHT_GTK = 2555904, 65363
_KEY_LEFT_WIN,  _KEY_LEFT_GTK  = 2424832, 65361

# Sticky-tracker config
_LOCK_RADIUS_PX = 120   # if tracked ball moves < this many px, keep it
_LOST_PATIENCE  = 10    # frames before giving up on a lost ball
_REDETECT_EVERY = 30    # frames between full re-detections while tracking


class TrackState(Enum):
    """Current state of the detect-then-track loop."""
    DETECTING = "DETECTING"
    TRACKING  = "TRACKING"


# =====================================================================
# Pipeline — compute every intermediate image once per frame
# =====================================================================

def _pipeline_stages(raw: np.ndarray, target: str):
    """Return (stages_list, hsv, mask, gray, circles) — images for each view."""
    undist  = bd.undistort(raw)
    blurred = cv2.GaussianBlur(undist, bd._BLUR_KSIZE, 0)
    hsv     = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    gray    = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)

    if target == "steel":
        mask = bd.morph_clean(bd.suppress_specular(
            hsv, bd.build_mask(hsv, bd.STEEL_RANGES)))
    else:
        mask = bd.morph_clean(bd.build_mask(hsv, bd.PING_PONG_RANGES))

    circles = bd.find_circles(gray)

    # Hough circles visualisation
    hough_img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    for hx, hy, hr in circles:
        cv2.circle(hough_img, (hx, hy), hr, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.circle(hough_img, (hx, hy), 2, (0, 0, 255), 3)

    stages = [
        raw,                                        # 0 raw
        undist,                                     # 1 undistorted
        blurred,                                    # 2 blurred
        cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR),       # 3 HSV viewable
        cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR),     # 4 active mask
        hough_img,                                  # 5 hough circles
    ]
    return stages, hsv, mask, circles


# =====================================================================
# Detect-then-track: detect once → OpenCV tracker → periodic re-detect
# =====================================================================

class DetectThenTrack:
    """Detect a ball, then hand off to a lightweight OpenCV tracker.

    Flow
    ----
        DETECTING → find a ball via the full HSV / contour pipeline
                  → initialise an OpenCV tracker (KCF by default)
        TRACKING  → use tracker.update() each frame  (very fast)
                  → every _REDETECT_EVERY frames, run full detection
                    to correct drift and refresh classification / distance
                  → if the tracker loses the ball → back to DETECTING

    Toggle *tracking_enabled* (T key) to fall back to detect-every-frame.
    """

    __slots__ = (
        "state", "tracker", "tracker_type", "cls", "real_diam",
        "confirmed", "cx", "cy", "r", "lost", "frames_tracked",
        "active", "tracking_enabled",
    )

    def __init__(self, tracker_type: str = "KCF"):
        self.tracker_type = tracker_type
        self.tracking_enabled = True
        self.reset()

    def reset(self):
        self.state = TrackState.DETECTING
        self.tracker = None
        self.cls: str = ""
        self.real_diam: float = 0.0
        self.confirmed: bool = False
        self.cx: int = 0
        self.cy: int = 0
        self.r: float = 0.0
        self.lost: int = _LOST_PATIENCE
        self.frames_tracked: int = 0
        self.active: bool = False

    @property
    def locked(self) -> bool:
        return self.active and self.lost < _LOST_PATIENCE

    @property
    def needs_detection(self) -> bool:
        """True when the main loop must run the full pipeline."""
        if not self.tracking_enabled:
            return True
        if self.state == TrackState.DETECTING:
            return True
        if self.frames_tracked >= _REDETECT_EVERY:
            return True
        return False

    # ---- called by main loop ----

    def update_detect(
        self,
        frame: np.ndarray,
        candidates: list[tuple[str, float, tuple[int, int], float, float, bool]],
    ) -> tuple[dict | None, float, tuple[int, int] | None]:
        """Run after the full pipeline.  Picks best candidate and
        (re-)initialises the OpenCV tracker."""
        if not candidates:
            return self._miss()

        if self.active and self.lost < _LOST_PATIENCE:
            best_cand, best_dist = None, float("inf")
            for c in candidates:
                _, _, (cx, cy), _, _, _ = c
                d = math.hypot(cx - self.cx, cy - self.cy)
                if d < best_dist:
                    best_dist, best_cand = d, c
            if best_cand is not None and best_dist < _LOCK_RADIUS_PX:
                return self._accept(frame, best_cand)
            return self._miss()

        ranked = sorted(candidates, key=lambda c: (c[5], c[1]), reverse=True)
        return self._accept(frame, ranked[0])

    def update_track(
        self, frame: np.ndarray,
    ) -> tuple[dict | None, float, tuple[int, int] | None]:
        """Lightweight tracker update — no detection pipeline needed."""
        if self.tracker is None:
            self.state = TrackState.DETECTING
            return self._miss()

        ok, bbox = self.tracker.update(frame)
        self.frames_tracked += 1

        if ok:
            (cx, cy), radius = bd.bbox_to_centre(bbox)
            self.cx, self.cy, self.r = cx, cy, radius
            self.lost = 0
            return self._make_result("track"), radius, (cx, cy)

        # tracker lost the ball
        self.state = TrackState.DETECTING
        self.tracker = None
        self.frames_tracked = 0
        return self._miss()

    # ---- internals ----

    def _accept(self, frame, cand):
        cls, _, (cx, cy), radius, real_d, confirmed = cand
        self.cx, self.cy, self.r = cx, cy, radius
        self.cls = cls
        self.real_diam = real_d
        self.confirmed = confirmed
        self.lost = 0
        self.active = True
        self.frames_tracked = 0

        if self.tracking_enabled:
            bbox = bd.detection_to_bbox((cx, cy), radius, frame.shape[:2])
            self.tracker = bd.create_tracker(self.tracker_type)
            self.tracker.init(frame, bbox)
            self.state = TrackState.TRACKING
        else:
            self.state = TrackState.DETECTING

        method = "hsv+shape" if confirmed else "hsv"
        return self._make_result(method), radius, (cx, cy)

    def _make_result(self, method: str) -> dict:
        w = 640
        x_norm = round((self.cx - w / 2) / (w / 2), 4)
        z = round(bd.estimate_distance(self.r * 2, self.real_diam), 1)
        return {
            "x": x_norm, "z": z, "classification": self.cls,
            "confirmed": self.confirmed, "method": method,
        }

    def _miss(self):
        if self.active:
            self.lost += 1
        if self.lost >= _LOST_PATIENCE:
            self.active = False
            self.state = TrackState.DETECTING
            self.tracker = None
            return None, 0.0, None
        return None, self.r, (self.cx, self.cy)


def _find_candidates(raw, hsv, mask, target, circles):
    """Collect all valid ball candidates with Hough confirmation."""
    candidates: list[tuple[str, float, tuple[int, int], float, float, bool]] = []

    if target in ("all", "ping_pong"):
        pp_mask = bd.morph_clean(bd.build_mask(hsv, bd.PING_PONG_RANGES)) if target == "all" else mask
        for _, area, centre, radius in bd.all_contours(pp_mask, bd.PING_PONG_CONTOUR):
            confirmed = bd._match_circle_to_contour(centre, circles) is not None
            candidates.append(("ping_pong", area, centre, radius, bd.PING_PONG_DIAMETER_CM, confirmed))

    if target in ("all", "steel"):
        st_mask = bd.morph_clean(bd.suppress_specular(
            hsv, bd.build_mask(hsv, bd.STEEL_RANGES))) if target == "all" else mask
        for _, area, centre, radius in bd.all_contours(st_mask, bd.STEEL_CONTOUR):
            confirmed = bd._match_circle_to_contour(centre, circles) is not None
            candidates.append(("steel", area, centre, radius, bd.STEEL_BALL_DIAMETER_CM, confirmed))

    # --- Hough-only fallback: unmatched circles as low-priority candidates ---
    if not candidates and circles:
        for hx, hy, hr in circles:
            if hr > 15:
                guess_cls, real_d = "ping_pong", bd.PING_PONG_DIAMETER_CM
            else:
                guess_cls, real_d = "steel", bd.STEEL_BALL_DIAMETER_CM
            # respect the active target filter
            if target != "all" and guess_cls != target:
                continue
            area = math.pi * hr * hr
            candidates.append((guess_cls, area, (hx, hy), float(hr), real_d, False))

    return candidates


# =====================================================================
# Drawing
# =====================================================================

def _outlined_text(img, text, org, scale, colour, thickness=1):
    """Draw text with a black outline for readability on any background."""
    cv2.putText(img, text, org, FONT, scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
    cv2.putText(img, text, org, FONT, scale, colour, thickness, cv2.LINE_AA)

def _draw_ball(img, result, radius, centre):
    if result is None or centre is None:
        return
    cx, cy = centre
    r = max(int(radius), 1)
    col = _COL.get(result["classification"], _COL["none"])

    cv2.circle(img, (cx, cy), r, col, 2, cv2.LINE_AA)
    g, t = _GAP, _XHAIR
    cv2.line(img, (cx, cy - r - g - t), (cx, cy - r - g), col, 2)
    cv2.line(img, (cx, cy + r + g), (cx, cy + r + g + t), col, 2)
    cv2.line(img, (cx - r - g - t, cy), (cx - r - g, cy), col, 2)
    cv2.line(img, (cx + r + g, cy), (cx + r + g + t, cy), col, 2)

    # label with method info
    method = result.get("method", "")
    if method == "track":
        conf_tag = " [TRACK]"
    elif result.get("confirmed"):
        conf_tag = " [HSV+SHAPE]"
    else:
        conf_tag = f" [{method.upper()}]" if method else ""
    _outlined_text(img, f"{result['classification']}  z={result['z']:.0f}cm{conf_tag}",
                   (cx - r, cy - r - 12), 0.45, col, 2)


def _draw_ghost(img, radius, centre, lost):
    if centre is None:
        return
    a = max(1.0 - lost / _LOST_PATIENCE, 0.15)
    gc = tuple(int(c * a) for c in _COL["ghost"])
    cv2.circle(img, centre, max(int(radius), 1), gc, 1, cv2.LINE_AA)
    _outlined_text(img, "searching...",
                   (centre[0] - 36, centre[1] - max(int(radius), 1) - 8),
                   0.40, gc, 1)


def _draw_hud(img, fps, stage_idx, target, result, overlay_on, tracker):
    h, w = img.shape[:2]

    # --- TOP LEFT: FPS + stage (stacked) ---
    fps_col = (0, 255, 0) if fps >= 15 else (0, 0, 255)
    _outlined_text(img, f"FPS: {fps:.0f}", (8, 20), 0.50, fps_col, 1)
    _outlined_text(img, STAGE_LABELS[stage_idx], (8, 40), 0.40, (255, 255, 0), 1)

    # --- TOP RIGHT: status bar (single line, right-aligned) ---
    ov_txt  = "OVR:ON" if overlay_on else "OVR:OFF"
    lk_txt  = "LOCKED" if tracker.locked else "SEARCH"
    trk_txt = f"TRK:{tracker.state.value}" if tracker.tracking_enabled else "TRK:OFF"
    if tracker.state == TrackState.TRACKING:
        trk_txt += f"({tracker.frames_tracked}/{_REDETECT_EVERY})"
    mth_txt = result.get("method", "").upper() if result else ""
    status  = f"{ov_txt}  {lk_txt}  {trk_txt}  {mth_txt}"
    sw = cv2.getTextSize(status, FONT, 0.38, 1)[0][0]
    _outlined_text(img, status, (w - sw - 8, 20), 0.38, (200, 200, 200), 1)

    # --- TOP RIGHT second row: mode + colour ---
    mode_txt = f"Mode:{target}  Col:{bd.PING_PONG_COLOUR}"
    mw = cv2.getTextSize(mode_txt, FONT, 0.38, 1)[0][0]
    _outlined_text(img, mode_txt, (w - mw - 8, 40), 0.38, (200, 200, 200), 1)

    # --- BOTTOM LEFT: detection result ---
    if result and result["classification"] != "unidentified":
        txt = f"x={result['x']:+.2f}  z={result['z']:.0f}cm  [{result['classification']}]"
    else:
        txt = "No detection"
    _outlined_text(img, txt, (8, h - 24), 0.45, (255, 255, 255), 1)

    # --- BOTTOM LEFT second row: key hints ---
    _outlined_text(img, "T:track  D:overlay  R:reset  <->:stage  1/2/3:mode",
                   (8, h - 6), 0.32, (140, 140, 140), 1)


# =====================================================================
# Main loop
# =====================================================================

def main():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    if not cap.isOpened():
        print("ERROR: camera not found"); return

    print("=" * 56)
    print("  Live Ball Detection — Detect then Track")
    print("  Q quit | O/W colour | 1/2/3 mode | T track toggle")
    print("  D overlay toggle | R reset tracker | <- -> stage")
    print("=" * 56)

    stage = 0
    target = "all"
    overlay_on = True
    tracker = DetectThenTrack(tracker_type="KCF")
    fps = 0.0
    prev_t = time.perf_counter()
    last_stages: list[np.ndarray] | None = None

    while True:
        ok, raw = cap.read()
        if not ok:
            continue
        if raw.shape[:2] != (480, 640):
            raw = cv2.resize(raw, (640, 480))

        # 1. detect or track
        if tracker.needs_detection:
            # --- full pipeline (runs every _REDETECT_EVERY frames or when searching) ---
            stages, hsv, mask, circles = _pipeline_stages(raw, target)
            candidates = _find_candidates(raw, hsv, mask, target, circles)
            # pass undistorted frame so tracker init uses corrected image
            result, radius, centre = tracker.update_detect(stages[1], candidates)
            last_stages = stages
        else:
            # --- lightweight track (skips HSV/contour/Hough entirely) ---
            undist = bd.undistort(raw)
            result, radius, centre = tracker.update_track(undist)
            # keep raw + undistorted current for display
            if last_stages is not None:
                last_stages[0] = raw
                last_stages[1] = undist

        # 2. pick base display image
        if last_stages is not None:
            img = last_stages[stage].copy()
        else:
            img = raw.copy()

        # 3. overlay (drawn on ANY stage when toggled on)
        if overlay_on:
            if result is not None and centre is not None:
                _draw_ball(img, result, radius, centre)
            elif centre is not None and tracker.active:
                _draw_ghost(img, radius, centre, tracker.lost)

        # 4. HUD
        now = time.perf_counter()
        fps = 0.9 * fps + 0.1 / max(now - prev_t, 1e-6)
        prev_t = now
        _draw_hud(img, fps, stage, target, result, overlay_on, tracker)

        cv2.imshow(WINDOW, img)

        # 5. key handling
        k = cv2.waitKeyEx(1)
        kb = k & 0xFF
        if   kb in (ord("q"), 27):  break
        elif kb == ord("o"):
            bd.PING_PONG_COLOUR = "orange"
            bd.PING_PONG_RANGES = bd.PING_PONG_PRESETS["orange"]
            tracker.reset(); print("[orange — tracker reset]")
        elif kb == ord("w"):
            bd.PING_PONG_COLOUR = "white"
            bd.PING_PONG_RANGES = bd.PING_PONG_PRESETS["white"]
            tracker.reset(); print("[white — tracker reset]")
        elif kb == ord("1"): target = "ping_pong"; tracker.reset(); print("[mode: ping_pong]")
        elif kb == ord("2"): target = "steel";     tracker.reset(); print("[mode: steel]")
        elif kb == ord("3"): target = "all";       tracker.reset(); print("[mode: all]")
        elif kb == ord("d"):
            overlay_on = not overlay_on
            print(f"[overlay {'ON' if overlay_on else 'OFF'}]")
        elif kb == ord("t"):
            tracker.tracking_enabled = not tracker.tracking_enabled
            if not tracker.tracking_enabled:
                tracker.state = TrackState.DETECTING
                tracker.tracker = None
            print(f"[tracking {'ON — detect then track' if tracker.tracking_enabled else 'OFF — detect every frame'}]")
        elif kb == ord("r"):
            tracker.reset()
            print("[tracker reset — re-searching]")

        if   k in (_KEY_RIGHT_WIN, _KEY_RIGHT_GTK):
            stage = (stage + 1) % N_STAGES; print(f"  >> {STAGE_LABELS[stage]}")
        elif k in (_KEY_LEFT_WIN, _KEY_LEFT_GTK):
            stage = (stage - 1) % N_STAGES; print(f"  >> {STAGE_LABELS[stage]}")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
