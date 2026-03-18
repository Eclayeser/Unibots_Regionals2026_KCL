"""
Main Test - Camera Ball Logic (Pi Headless Stream)
==================================================
Runs the ball detection and decision logic used by main_integrated.py,
but never imports or calls motor/servo code.

VIEW ON LAPTOP:
Open browser to: http://192.168.137.187:5000/stream.mjpg
(adjust IP if different from your Pi hotspot IP)
"""

from __future__ import annotations

import os
os.environ["QT_QPA_PLATFORM"] = "offscreen"  # Must be before cv2 import on headless Pi

import math
import time

import cv2
import numpy as np

import ball_detector_runtime as bdr
from config import (
    CAPTURE_ALIGN_MAX_ABS_ANGLE_DEG,
    CAPTURE_BLOCKED_WAIT_S,
    CAPTURE_SKIP_COOLDOWN_S,
)

FONT = cv2.FONT_HERSHEY_SIMPLEX
CAPTURE_Y_THRESHOLD = 0.88


def _draw_apf_widget(img: np.ndarray, nav: dict | None) -> None:
    h, w = img.shape[:2]
    radius = 40
    cx = w - 60
    cy = 60

    cv2.circle(img, (cx, cy), radius, (220, 220, 220), 1, cv2.LINE_AA)
    cv2.putText(img, "APF", (cx - 18, cy - radius - 8), FONT, 0.45, (230, 230, 230), 1, cv2.LINE_AA)

    if nav is None:
        return

    angle_deg = float(nav.get("angle", 0.0))
    magnitude = float(np.clip(nav.get("magnitude", 0.0), 0.0, 1.0))
    length = int(round(radius * magnitude))

    theta = math.radians(angle_deg)
    ex = int(round(cx + length * math.sin(theta)))
    ey = int(round(cy - length * math.cos(theta)))

    cv2.arrowedLine(img, (cx, cy), (ex, ey), (0, 255, 255), 2, cv2.LINE_AA, tipLength=0.25)


def _draw_decision_overlay(
    img: np.ndarray,
    detection: dict,
    obstacles: list[dict],
    state: str,
    action: str,
    nav: dict | None,
    profile: str,
    fps: float,
) -> np.ndarray:
    out = bdr.annotate_debug_frame(img, detection)

    # Obstacles: red boxes
    for obstacle in obstacles:
        x, y, w, h = obstacle["bbox"]
        cv2.rectangle(out, (int(x), int(y)), (int(x + w), int(y + h)), (0, 0, 255), 2, cv2.LINE_AA)

    # Balls: orange for ping-pong, gray for steel
    for ball in detection.get("detected_balls", []):
        x = int(ball["x"])
        y = int(ball["y"])
        r = max(int(ball["radius"]), 3)
        color = (0, 165, 255) if str(ball.get("type", "")).lower() in ("pingpong", "ping_pong") else (160, 160, 160)
        cv2.rectangle(out, (x - r, y - r), (x + r, y + r), color, 2, cv2.LINE_AA)

    _draw_apf_widget(out, nav)

    cv2.putText(out, f"FPS:{fps:.0f} Profile:{profile}", (8, 20), FONT, 0.55, (0, 220, 0), 2, cv2.LINE_AA)

    state_text = f"STATE: {state}"
    action_text = f"INTENDED ACTION: {action}"
    y0 = out.shape[0] - 42
    cv2.putText(out, state_text, (10, y0), FONT, 0.72, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(out, action_text, (10, y0 + 28), FONT, 0.62, (255, 255, 0), 2, cv2.LINE_AA)

    return out


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

    blocked_since_s = 0.0
    skip_until_s = 0.0
    skip_anchor: tuple[float, float, float] | None = None

    fps = 0.0
    prev_t = time.perf_counter()

    print("=" * 72)
    print(" Main Test Camera Ball (Pi Headless Stream)")
    print(" Mirrors decision logic from main_integrated.py without motor/servo calls")
    print(" ")
    print(" VIEW ON LAPTOP:")
    print(" Open browser to: http://192.168.137.187:5000/stream.mjpg")
    print(" (adjust IP if different from your Pi hotspot IP)")
    print(" ")
    print(" Running continuously... Ctrl+C to stop")
    print("=" * 72)

    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            continue
        if frame.shape[:2] != (480, 640):
            frame = cv2.resize(frame, (640, 480))

        try:
            detection = bdr.detect_balls(frame, ping_pong_profile=ping_pong_profile)
        except Exception as exc:
            print(f"ERROR in detect_balls: {exc}")
            continue

        detected_balls = detection["detected_balls"]
        frame_height = detection["frame"].shape[0]
        now_s = time.monotonic()

        state = "SEARCHING"
        action = "Action: pivot_left"
        nav = None
        obstacles: list[dict] = []

        if not detected_balls:
            blocked_since_s = 0.0
        else:
            candidates = detected_balls
            if skip_anchor is not None and now_s < skip_until_s:
                sx, sy, sr = skip_anchor
                filtered: list[dict] = []
                for candidate in detected_balls:
                    dx = float(candidate["x"]) - sx
                    dy = float(candidate["y"]) - sy
                    if (dx * dx + dy * dy) ** 0.5 > sr:
                        filtered.append(candidate)
                if filtered:
                    candidates = filtered
            else:
                skip_anchor = None

            target = candidates[0]
            obstacles = bdr.detect_obstacles(detection["frame"], detection["ball_mask"])
            nav = bdr.compute_navigation_vector(
                target, obstacles, detection["frame"].shape, detection["calibration"]
            )

            ball_bottom_y = target["y"] + target["radius"]
            in_capture_zone = ball_bottom_y >= frame_height * CAPTURE_Y_THRESHOLD
            well_aligned = abs(float(target["angle"])) <= CAPTURE_ALIGN_MAX_ABS_ANGLE_DEG
            corridor_blocked = bdr.is_capture_corridor_blocked(
                target, obstacles, detection["frame"].shape, detection["calibration"]
            )

            if in_capture_zone:
                if well_aligned and not corridor_blocked:
                    blocked_since_s = 0.0
                    state = "CAPTURING"
                    action = "Action: confident_approach_toGrab"
                    nav = None
                else:
                    if blocked_since_s == 0.0:
                        blocked_since_s = now_s
                    elapsed = now_s - blocked_since_s

                    if elapsed >= CAPTURE_BLOCKED_WAIT_S:
                        skip_until_s = now_s + CAPTURE_SKIP_COOLDOWN_S
                        skip_radius = max(70.0, float(target["radius"]) * 2.5)
                        skip_anchor = (float(target["x"]), float(target["y"]), skip_radius)
                        blocked_since_s = 0.0
                        state = "SEARCHING"
                        action = "Action: skip_target_and_search"
                        nav = None
                    else:
                        state = "TRACKING"
                        if nav is not None:
                            action = f"Action: apf_move(angle={nav['angle']:.1f}, magnitude={nav['magnitude']:.2f})"
                        else:
                            action = "Action: pivot_left"
            else:
                blocked_since_s = 0.0
                state = "TRACKING"
                if nav is not None:
                    action = f"Action: apf_move(angle={nav['angle']:.1f}, magnitude={nav['magnitude']:.2f})"
                else:
                    action = "Action: pivot_left"

        now = time.perf_counter()
        fps = 0.90 * fps + 0.10 / max(now - prev_t, 1e-6)
        prev_t = now

        vis = _draw_decision_overlay(
            detection["frame"], detection, obstacles, state, action, nav, ping_pong_profile, fps
        )

        if stream_on:
            bdr.push_debug_frame(vis)

    cap.release()


if __name__ == "__main__":
    main()
