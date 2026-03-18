"""
Live Detection Test Harness (KrishivBallDetection runtime)
==========================================================
Uses the new `ball_detector_runtime` pipeline directly:
- steel ball detection with grey-region + Hough circles (multi-ball)
- Kalman smoothing and lost-frame coasting
- optional MJPEG debug stream (http://<pi-ip>:5000)

Controls
--------
Q / ESC         quit
O               orange ping-pong profile
W               white  ping-pong profile
M               toggle MJPEG stream on/off (runtime global)
LEFT / RIGHT    cycle view stage
"""

from __future__ import annotations

import time

import cv2
import numpy as np

import ball_detector_runtime as bdr

WINDOW = "UniBots - Detection Live (Runtime)"
FONT = cv2.FONT_HERSHEY_SIMPLEX

STAGE_LABELS = (
    "0: Raw camera",
    "1: Undistorted",
    "2: Ping-pong mask",
    "3: Steel Hough candidates mask",
    "4: Accepted ball mask",
)
N_STAGES = len(STAGE_LABELS)

KEY_RIGHT_WIN, KEY_RIGHT_GTK = 2555904, 65363
KEY_LEFT_WIN, KEY_LEFT_GTK = 2424832, 65361

COLORS = {
    "PingPong": (0, 180, 255),
    "steel": (210, 210, 210),
}


def _outlined_text(img: np.ndarray, text: str, org: tuple[int, int], scale: float, colour: tuple[int, int, int], thickness: int = 1) -> None:
    cv2.putText(img, text, org, FONT, scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
    cv2.putText(img, text, org, FONT, scale, colour, thickness, cv2.LINE_AA)


def _draw_runtime_overlay(img: np.ndarray, detection: dict) -> np.ndarray:
    """Draw local overlay with runtime detections, Hough circles, Kalman trajectory, and ROI.

    This overlay is independent of the MJPEG stream and is used for local
    interactive visualization in OpenCV windows.
    """
    out = img.copy()
    balls = detection.get("detected_balls", [])

    # Hough circle proposals used by steel detector.
    for cx, cy, r in detection.get("steel_hough_circles", []):
        cv2.circle(out, (int(cx), int(cy)), max(int(r), 2), (255, 0, 200), 1, cv2.LINE_AA)

    # Kalman trajectory (yellow polyline) when moving confirmed
    trajectory = detection.get("steel_trajectory", [])
    if trajectory and len(trajectory) >= 2:
        traj_array = np.array(trajectory, dtype=np.int32)
        cv2.polylines(out, [traj_array], False, (0, 255, 255), 2, cv2.LINE_AA)

    # Kalman ROI (cyan box) for search area visualization
    roi = detection.get("steel_kalman_roi")
    if roi is not None:
        x1, y1, x2, y2 = roi
        cv2.rectangle(out, (int(x1), int(y1)), (int(x2), int(y2)), (255, 255, 0), 1, cv2.LINE_AA)

    # Ball circles + labels
    for ball in balls:
        x = int(ball["x"])
        y = int(ball["y"])
        r = max(int(ball["radius"]), 2)
        colour = COLORS.get(ball["type"], (130, 130, 130))
        if ball.get("predicted", False):
            colour = (0, 180, 255)

        cv2.circle(out, (x, y), r, colour, 2, cv2.LINE_AA)
        pred = " [PRED]" if ball.get("predicted", False) else ""
        text = f"{ball['type']}{pred} {ball['distance']:.1f}cm {ball['angle']:.1f}deg"
        _outlined_text(out, text, (x - r, max(y - r - 8, 12)), 0.38, colour, 1)

    return out


def _draw_hud(img: np.ndarray, fps: float, stage: int, profile: str, detection: dict, stream_on: bool) -> None:
    h, w = img.shape[:2]
    fps_col = (0, 220, 0) if fps >= 15.0 else (0, 0, 255)

    _outlined_text(img, f"FPS: {fps:.0f}", (8, 20), 0.50, fps_col, 1)
    _outlined_text(img, STAGE_LABELS[stage], (8, 40), 0.42, (255, 255, 0), 1)

    balls = detection.get("detected_balls", [])
    count_pp = sum(1 for b in balls if b["type"] == "PingPong")
    count_st = sum(1 for b in balls if b["type"] == "steel")
    info = f"Detected: {len(balls)} (PP:{count_pp} Steel:{count_st})"
    _outlined_text(img, info, (8, h - 24), 0.42, (255, 255, 255), 1)

    mode_text = f"Profile:{profile} Overlay:ON Stream:{'ON' if stream_on else 'OFF'}"
    tw = cv2.getTextSize(mode_text, FONT, 0.38, 1)[0][0]
    _outlined_text(img, mode_text, (w - tw - 8, 20), 0.38, (210, 210, 210), 1)

    _outlined_text(
        img,
        "O/W:profile  M:stream  <- ->:stage  Q:quit",
        (8, h - 6),
        0.32,
        (145, 145, 145),
        1,
    )


def _build_stage_images(raw: np.ndarray, detection: dict) -> list[np.ndarray]:
    undistorted = detection["frame"]
    pp_mask = cv2.cvtColor(detection["ping_pong_mask"], cv2.COLOR_GRAY2BGR)
    steel_mask = cv2.cvtColor(detection["steel_mask"], cv2.COLOR_GRAY2BGR)
    ball_mask = cv2.cvtColor(detection["ball_mask"], cv2.COLOR_GRAY2BGR)

    # Always draw detections on every stage so debugging context is never lost.
    raw_ov = _draw_runtime_overlay(raw, detection)
    undist_ov = _draw_runtime_overlay(undistorted, detection)
    pp_mask_ov = _draw_runtime_overlay(pp_mask, detection)
    steel_mask_ov = _draw_runtime_overlay(steel_mask, detection)
    ball_mask_ov = _draw_runtime_overlay(ball_mask, detection)

    return [
        raw_ov,
        undist_ov,
        pp_mask_ov,
        steel_mask_ov,
        ball_mask_ov,
    ]


def main() -> None:
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    if not cap.isOpened():
        print("ERROR: camera not found")
        return

    print("=" * 62)
    print(" Live Detection - Runtime (Kalman steel + debug stream)")
    print(" Q quit | O/W profile | D overlay | M stream | <- -> stage")
    print(" If stream is ON: open http://<pi-ip>:5000")
    print("=" * 62)

    stage = 0
    ping_pong_profile = "orange"

    fps = 0.0
    prev_t = time.perf_counter()

    # Honor default stream toggle in runtime module.
    stream_on = bool(bdr.STREAM_DEBUG_FEED)
    if stream_on:
        bdr.start_debug_mjpeg_server()

    while True:
        ok, raw = cap.read()
        if not ok or raw is None:
            continue
        if raw.shape[:2] != (480, 640):
            raw = cv2.resize(raw, (640, 480))

        try:
            detection = bdr.detect_balls(raw, ping_pong_profile=ping_pong_profile)
        except Exception as exc:
            frame = raw.copy()
            _outlined_text(frame, f"detect_balls error: {exc}", (10, 30), 0.5, (0, 0, 255), 1)
            cv2.imshow(WINDOW, frame)
            if (cv2.waitKeyEx(1) & 0xFF) in (ord("q"), 27):
                break
            continue

        stages = _build_stage_images(raw, detection)
        img = stages[stage].copy()

        now = time.perf_counter()
        fps = 0.90 * fps + 0.10 / max(now - prev_t, 1e-6)
        prev_t = now

        _draw_hud(img, fps, stage, ping_pong_profile, detection, stream_on)
        cv2.imshow(WINDOW, img)

        k = cv2.waitKeyEx(1)
        kb = k & 0xFF

        if kb in (ord("q"), 27):
            break
        elif kb == ord("o"):
            ping_pong_profile = bdr.set_ping_pong_profile("orange")
            print("[profile: orange]")
        elif kb == ord("w"):
            ping_pong_profile = bdr.set_ping_pong_profile("white")
            print("[profile: white]")
        elif kb == ord("m"):
            stream_on = not stream_on
            bdr.STREAM_DEBUG_FEED = stream_on
            if stream_on:
                bdr.start_debug_mjpeg_server()
                print("[stream ON] http://<pi-ip>:5000")
            else:
                print("[stream OFF]")

        if k in (KEY_RIGHT_WIN, KEY_RIGHT_GTK):
            stage = (stage + 1) % N_STAGES
            print(f"  >> {STAGE_LABELS[stage]}")
        elif k in (KEY_LEFT_WIN, KEY_LEFT_GTK):
            stage = (stage - 1) % N_STAGES
            print(f"  >> {STAGE_LABELS[stage]}")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
