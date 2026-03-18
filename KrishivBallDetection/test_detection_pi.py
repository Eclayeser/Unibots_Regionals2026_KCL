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

import time

import cv2

import ball_detector_runtime as bdr

WINDOW = "UniBots - PI Camera Detection"
FONT = cv2.FONT_HERSHEY_SIMPLEX


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


def main() -> None:
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    if not cap.isOpened():
        print("ERROR: camera not found")
        return

    ping_pong_profile = "orange"
    stream_on = bool(bdr.STREAM_DEBUG_FEED)
    if stream_on:
        bdr.start_debug_mjpeg_server()

    fps = 0.0
    prev_t = time.perf_counter()

    print("=" * 62)
    print(" PI Camera Detection Test (Headless Stream)")
    print(" ")
    print(" VIEW ON LAPTOP:")
    print(" Open browser to: http://192.168.137.187:5000/stream.mjpg")
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
            vis = bdr.annotate_debug_frame(detection["frame"], detection)
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
