"""
Pi Camera Detection Test (Hardware-Free)
=======================================
Standalone detector runner for Raspberry Pi camera testing.
Keeps laptop test files unchanged.

Controls
--------
Q / ESC  quit
O        orange ping-pong profile
W        white ping-pong profile
M        toggle MJPEG stream push on/off
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import cv2
import numpy as np

# Make sibling KrishivImplementation importable when running this script directly.
_ROOT = Path(__file__).resolve().parent.parent
_K_IMPL_DIR = _ROOT / "KrishivImplementation"
if _K_IMPL_DIR.exists():
    k_path = str(_K_IMPL_DIR)
    if k_path not in sys.path:
        sys.path.insert(0, k_path)

try:
    # Use the same K runtime used by main_K when available.
    import ball_detector_runtime_K as bdr
except ImportError:
    # Fallback for legacy/local testing layouts.
    import ball_detector_runtime as bdr

WINDOW = "UniBots - PI Camera Detection"
FONT = cv2.FONT_HERSHEY_SIMPLEX


def _mask_tile(mask: np.ndarray, label: str, size: tuple[int, int] = (320, 240)) -> np.ndarray:
    tile = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    tile = cv2.resize(tile, size, interpolation=cv2.INTER_NEAREST)
    cv2.putText(tile, label, (8, 22), FONT, 0.55, (0, 255, 255), 2, cv2.LINE_AA)
    return tile


def _frame_tile(frame: np.ndarray, label: str, size: tuple[int, int] = (320, 240)) -> np.ndarray:
    tile = cv2.resize(frame, size, interpolation=cv2.INTER_AREA)
    cv2.putText(tile, label, (8, 22), FONT, 0.55, (0, 255, 255), 2, cv2.LINE_AA)
    return tile


def _build_stream_panel(detection: dict, annotated: np.ndarray) -> np.ndarray:
    pp_mask = detection["ping_pong_mask"]
    apriltag_mask = detection.get("apriltag_mask", np.zeros_like(pp_mask))
    pre_steel_shield = detection.get("pre_steel_shield_mask", cv2.bitwise_or(pp_mask, apriltag_mask))
    steel_mask = detection["steel_mask"]

    top = cv2.hconcat([
        _frame_tile(annotated, "Annotated"),
        _mask_tile(pp_mask, "PingPong mask"),
        _mask_tile(apriltag_mask, "AprilTag mask"),
    ])
    bottom = cv2.hconcat([
        _mask_tile(pre_steel_shield, "Pre-steel shield"),
        _mask_tile(steel_mask, "Steel mask"),
        _mask_tile(detection["ball_mask"], "Combined mask"),
    ])
    panel = cv2.vconcat([top, bottom])

    pp_nz = int(np.count_nonzero(pp_mask))
    tag_nz = int(np.count_nonzero(apriltag_mask))
    shield_nz = int(np.count_nonzero(pre_steel_shield))
    steel_nz = int(np.count_nonzero(steel_mask))
    cv2.putText(
        panel,
        f"MaskNZ PP:{pp_nz} TAG:{tag_nz} SH:{shield_nz} ST:{steel_nz}",
        (10, panel.shape[0] - 10),
        FONT,
        0.55,
        (180, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return panel


def _draw_hud(img, fps: float, profile: str, balls: list[dict], stream_on: bool) -> None:
    count_pp = sum(1 for b in balls if b["type"] == "PingPong")
    count_st = sum(1 for b in balls if b["type"] == "steel")

    cv2.putText(img, f"FPS: {fps:.0f}", (8, 20), FONT, 0.55, (0, 220, 0), 2, cv2.LINE_AA)
    cv2.putText(
        img,
        f"Profile:{profile}  Balls:{len(balls)} (PP:{count_pp} Steel:{count_st})",
        (8, 44),
        FONT,
        0.48,
        (230, 230, 230),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        img,
        f"Stream:{'ON' if stream_on else 'OFF'}  O/W profile  M stream  Q quit",
        (8, max(img.shape[0] - 10, 12)),
        FONT,
        0.42,
        (180, 180, 180),
        1,
        cv2.LINE_AA,
    )


def _draw_ball_overlays(img: np.ndarray, balls: list[dict]) -> None:
    for ball in balls:
        bx, by = int(ball.get("x", 0)), int(ball.get("y", 0))
        br = max(int(ball.get("radius", 0)), 6)
        btype = str(ball.get("type", ""))

        if btype == "PingPong":
            colour = (0, 210, 80)  # green
        elif btype == "steel":
            colour = (0, 180, 255)  # orange
        else:
            colour = (255, 200, 0)  # fallback

        cv2.circle(img, (bx, by), br, colour, 3, cv2.LINE_AA)
        cv2.circle(img, (bx, by), 2, colour, -1, cv2.LINE_AA)

        angle = float(ball.get("angle", 0.0))
        distance = float(ball.get("distance", 0.0))
        label = f"{btype}  {angle:+.1f}deg  {distance:.1f}cm"
        label_y = max(by - br - 10, 14)
        cv2.putText(img, label, (max(bx - br, 5), label_y), FONT, 0.48, colour, 2, cv2.LINE_AA)


def _draw_apriltag_removed_boxes(img: np.ndarray, detection: dict) -> None:
    tag_mask = detection.get("apriltag_mask")
    if tag_mask is None:
        return

    contours, _ = cv2.findContours(tag_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 120:
            continue

        x, y, w, h = cv2.boundingRect(contour)
        # Draw a square around the removed tag region.
        side = max(w, h)
        cx = x + w // 2
        cy = y + h // 2
        x1 = max(cx - side // 2, 0)
        y1 = max(cy - side // 2, 0)
        x2 = min(x1 + side, img.shape[1] - 1)
        y2 = min(y1 + side, img.shape[0] - 1)

        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 2, cv2.LINE_AA)
        cv2.putText(
            img,
            "AprilTag removed",
            (x1, max(y1 - 8, 12)),
            FONT,
            0.46,
            (255, 0, 255),
            2,
            cv2.LINE_AA,
        )


def main() -> None:
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    if not cap.isOpened():
        print("ERROR: camera not found")
        return

    ping_pong_profile = "orange"
    # Default ON for Pi field testing so port 5000 is always opened.
    stream_on = True
    bdr.STREAM_DEBUG_FEED = True
    bdr.start_debug_mjpeg_server()

    fps = 0.0
    prev_t = time.perf_counter()

    print("=" * 62)
    print(" PI Camera Detection Test (Headless Stream)")
    print(" ")
    print(" VIEW ON LAPTOP:")
    print(" Open browser to: http://192.168.137.187:5000/")
    print(" or directly:      http://192.168.137.187:5000/stream")
    print(" (adjust IP if different from your Pi hotspot IP)")
    print(" ")
    print(" Running continuously... Ctrl+C to stop")
    print("=" * 62)

    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            continue
        if frame.shape[:2] != (480, 640):
            frame = cv2.resize(frame, (640, 480))

        try:
            detection = bdr.detect_balls(frame, ping_pong_profile=ping_pong_profile)
            annotated = bdr.annotate_debug_frame(detection["frame"], detection)
            vis = annotated
            _draw_apriltag_removed_boxes(vis, detection)
            _draw_ball_overlays(vis, detection.get("detected_balls", []))
            _draw_hud(
                vis,
                fps,
                ping_pong_profile,
                detection.get("detected_balls", []),
                stream_on,
            )
        except Exception as exc:
            vis = frame.copy()
            cv2.putText(vis, f"detect_balls error: {exc}", (10, 30), FONT, 0.6, (0, 0, 255), 2, cv2.LINE_AA)

        now = time.perf_counter()
        fps = 0.90 * fps + 0.10 / max(now - prev_t, 1e-6)
        prev_t = now

        if stream_on:
            bdr.push_debug_frame(vis)

    cap.release()


if __name__ == "__main__":
    main()
