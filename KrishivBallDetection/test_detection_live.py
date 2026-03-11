"""
Live Ball Detection — Test Harness (Multi-Ball)
=================================================
Opens the default webcam and runs the multi-ball detection pipeline
with real-time annotated preview showing ALL detected balls with
tracking IDs.

Controls
--------
    Q / ESC         quit
    O               orange ping-pong mode
    W               white  ping-pong mode
    1               detect ping-pong only
    2               detect steel only
    3               detect all (default)
    D               toggle detection overlay on/off
    R               reset tracker IDs
    LEFT / RIGHT    cycle pipeline stage view

Pipeline stages (arrow keys)
-----------------------------
    0  Raw camera
    1  Undistorted
    2  Gaussian blur
    3  HSV (viewable)
    4  Ping-pong mask
    5  Steel mask
    6  Hough circles
"""

from __future__ import annotations

import math
import time

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
    "4: PP mask",
    "5: Steel mask",
    "6: Hough circles",
)
N_STAGES = len(STAGE_LABELS)

_XHAIR = 28
_GAP   = 5
_COL = {
    "ping_pong": (0, 200, 255),
    "steel":     (200, 200, 200),
    "none":      (80, 80, 80),
}

_KEY_RIGHT_WIN, _KEY_RIGHT_GTK = 2555904, 65363
_KEY_LEFT_WIN,  _KEY_LEFT_GTK  = 2424832, 65361


# =====================================================================
# Pipeline — compute every intermediate image once per frame
# =====================================================================

def _pipeline_stages(raw: np.ndarray, target: str):
    """Return list of debug-view images."""
    undist  = bd.undistort(raw)
    blurred = cv2.GaussianBlur(undist, bd._BLUR_KSIZE, 0)
    hsv     = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    gray    = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)

    pp_mask = bd.morph_clean(bd.build_mask(hsv, bd.PING_PONG_RANGES))
    st_mask = bd.morph_clean(bd.build_mask(hsv, bd.STEEL_RANGES))

    circles = bd.find_circles(gray)
    hough_img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    for hx, hy, hr in circles:
        cv2.circle(hough_img, (hx, hy), hr, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.circle(hough_img, (hx, hy), 2, (0, 0, 255), 3)

    stages = [
        raw,
        undist,
        blurred,
        cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR),
        cv2.cvtColor(pp_mask, cv2.COLOR_GRAY2BGR),
        cv2.cvtColor(st_mask, cv2.COLOR_GRAY2BGR),
        hough_img,
    ]
    return stages


# =====================================================================
# Drawing helpers
# =====================================================================

def _outlined_text(img, text, org, scale, colour, thickness=1):
    cv2.putText(img, text, org, FONT, scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
    cv2.putText(img, text, org, FONT, scale, colour, thickness, cv2.LINE_AA)


def _draw_ball(img, ball: dict):
    cx, cy = ball["x"], ball["y"]
    r = max(int(ball["radius"]), 1)
    col = _COL.get(ball["type"], _COL["none"])
    ball_id = ball.get("id", "?")

    cv2.circle(img, (cx, cy), r, col, 2, cv2.LINE_AA)
    g, t = _GAP, _XHAIR
    cv2.line(img, (cx, cy - r - g - t), (cx, cy - r - g), col, 2)
    cv2.line(img, (cx, cy + r + g), (cx, cy + r + g + t), col, 2)
    cv2.line(img, (cx - r - g - t, cy), (cx - r - g, cy), col, 2)
    cv2.line(img, (cx + r + g, cy), (cx + r + g + t, cy), col, 2)

    method = ball.get("method", "")
    label = f"#{ball_id} {ball['type']}  {ball['distance']:.0f}cm  [{method}]"
    _outlined_text(img, label, (cx - r, cy - r - 12), 0.40, col, 1)


def _draw_hud(img, fps, stage_idx, target, balls, overlay_on):
    h, w = img.shape[:2]
    fps_col = (0, 255, 0) if fps >= 15 else (0, 0, 255)
    _outlined_text(img, f"FPS: {fps:.0f}", (8, 20), 0.50, fps_col, 1)
    _outlined_text(img, STAGE_LABELS[stage_idx], (8, 40), 0.40, (255, 255, 0), 1)

    mode_txt = f"Mode:{target}  Col:{bd.PING_PONG_COLOUR}  Overlay:{'ON' if overlay_on else 'OFF'}"
    mw = cv2.getTextSize(mode_txt, FONT, 0.38, 1)[0][0]
    _outlined_text(img, mode_txt, (w - mw - 8, 20), 0.38, (200, 200, 200), 1)

    if balls:
        count_pp = sum(1 for b in balls if b["type"] == "ping_pong")
        count_st = sum(1 for b in balls if b["type"] == "steel")
        txt = f"Detected: {len(balls)} ball(s)  (PP:{count_pp}  Steel:{count_st})"
    else:
        txt = "No balls detected"
    _outlined_text(img, txt, (8, h - 24), 0.42, (255, 255, 255), 1)

    _outlined_text(img, "D:overlay  R:reset  <->:stage  1/2/3:mode  O/W:colour",
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
    print("  Live Ball Detection — Multi-Ball + Tracking IDs")
    print("  Q quit | O/W colour | 1/2/3 mode | D overlay")
    print("  R reset tracker IDs | <- -> stage")
    print("=" * 56)

    stage = 0
    target = "all"
    overlay_on = True
    tracker = bd.BallTracker(max_lost=15)
    fps = 0.0
    prev_t = time.perf_counter()

    while True:
        ok, raw = cap.read()
        if not ok:
            continue
        if raw.shape[:2] != (480, 640):
            raw = cv2.resize(raw, (640, 480))

        # Detect all balls and assign persistent IDs
        raw_detections = bd.detect_balls(raw, target=target)
        balls = tracker.update(raw_detections)

        # Pipeline views
        stages = _pipeline_stages(raw, target)
        img = stages[stage].copy()

        # Draw overlay
        if overlay_on:
            for ball in balls:
                _draw_ball(img, ball)

        # HUD
        now = time.perf_counter()
        fps = 0.9 * fps + 0.1 / max(now - prev_t, 1e-6)
        prev_t = now
        _draw_hud(img, fps, stage, target, balls, overlay_on)

        cv2.imshow(WINDOW, img)

        # Key handling
        k = cv2.waitKeyEx(1)
        kb = k & 0xFF
        if   kb in (ord("q"), 27):  break
        elif kb == ord("o"):
            bd.PING_PONG_COLOUR = "orange"
            bd.PING_PONG_RANGES = bd.PING_PONG_PRESETS["orange"]
            tracker.reset(); print("[orange]")
        elif kb == ord("w"):
            bd.PING_PONG_COLOUR = "white"
            bd.PING_PONG_RANGES = bd.PING_PONG_PRESETS["white"]
            tracker.reset(); print("[white]")
        elif kb == ord("1"): target = "ping_pong"; tracker.reset(); print("[mode: ping_pong]")
        elif kb == ord("2"): target = "steel";     tracker.reset(); print("[mode: steel]")
        elif kb == ord("3"): target = "all";       tracker.reset(); print("[mode: all]")
        elif kb == ord("d"):
            overlay_on = not overlay_on
            print(f"[overlay {'ON' if overlay_on else 'OFF'}]")
        elif kb == ord("r"):
            tracker.reset()
            print("[tracker reset]")

        if   k in (_KEY_RIGHT_WIN, _KEY_RIGHT_GTK):
            stage = (stage + 1) % N_STAGES; print(f"  >> {STAGE_LABELS[stage]}")
        elif k in (_KEY_LEFT_WIN, _KEY_LEFT_GTK):
            stage = (stage - 1) % N_STAGES; print(f"  >> {STAGE_LABELS[stage]}")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
