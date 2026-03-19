"""
test_apriltag_stream.py
=======================
Headless AprilTag detection test for Raspberry Pi (no monitor required).

Run on the Pi:
    python3 test_apriltag_stream.py

Then open in ANY browser on the same network:
    http://<pi-ip-address>:5000

Stream shows a fully annotated live view:
  - Every detected valid tag: bounding box, centre dot, ID
  - Per-tag readout: distance, yaw, lateral offset
  - Chosen (closest) tag highlighted in green; others in amber
  - Blue steering arrow showing angle and magnitude on the chosen tag
  - "NO TAGS" banner when nothing is visible
  - FPS counter in the top-right corner

Console prints the raw process_frame() dict at 1 Hz.
"""

import sys
import time
import math
import threading
import cv2
from flask import Flask, Response

# ── Import project modules ─────────────────────────────────────────────────────
# Place this script in the same folder as AprilTagNavigator_M.py and configTag_M.py.
try:
    from AprilTagNavigator_M import AprilTagNavigator
    import configTag_M as cfg
except ModuleNotFoundError as exc:
    sys.exit(
        f"[ERROR] Could not import project modules: {exc}\n"
        "Make sure this script sits next to AprilTagNavigator_M.py and configTag_M.py."
    )

# ── Flask app ──────────────────────────────────────────────────────────────────
app = Flask(__name__)

# ── Shared state (written by capture thread, read by Flask) ────────────────────
_frame_lock   = threading.Lock()
_latest_frame = None   # annotated JPEG bytes

# ── Colours (BGR) ─────────────────────────────────────────────────────────────
CLR_CHOSEN    = (0,   210,   0)   # green  – chosen tag
CLR_OTHER     = (0,   200, 255)   # amber  – other valid tags
CLR_NOTAG     = (60,   60,  60)   # dark grey – no-tag banner
CLR_TEXT      = (255, 255, 255)   # white
CLR_SUBTEXT   = (200, 200, 200)   # light grey
CLR_CROSSHAIR = (120, 120, 120)   # grey crosshair
CLR_ARROW     = (255, 100,   0)   # blue   – steering arrow

FONT      = cv2.FONT_HERSHEY_SIMPLEX
FONT_BOLD = cv2.FONT_HERSHEY_DUPLEX


# ── Drawing helpers ────────────────────────────────────────────────────────────

def _draw_tag(frame, tag: dict, colour, is_chosen: bool, move_target: dict | None):
    """Draw bounding box, centre, labels, and (if chosen) steering arrow."""
    cx, cy = tag["centre_x"], tag["centre_y"]
    dist   = tag["distance_cm"]

    # Size-proportional box: shrinks as the tag gets closer
    half      = max(20, int(1800 / max(dist, 1)))
    thickness = 3 if is_chosen else 1

    cv2.rectangle(frame, (cx - half, cy - half), (cx + half, cy + half),
                  colour, thickness)

    # Corner tick marks
    tick = max(6, half // 3)
    for (px, py) in [(cx - half, cy - half), (cx + half, cy - half),
                     (cx - half, cy + half), (cx + half, cy + half)]:
        dx = tick if px > cx else -tick
        dy = tick if py > cy else -tick
        cv2.line(frame, (px, py), (px + dx, py), colour, thickness)
        cv2.line(frame, (px, py), (px, py + dy), colour, thickness)

    # Centre dot
    cv2.circle(frame, (cx, cy), 5 if is_chosen else 3, colour, -1)

    # Tag ID label above box
    cv2.putText(frame, f"ID {tag['tag_id']}",
                (cx - half, cy - half - 8),
                FONT_BOLD, 0.65, colour, 2, cv2.LINE_AA)

    # Measurement labels below box
    label_colour = CLR_TEXT if is_chosen else CLR_SUBTEXT
    y = cy + half + 18
    for line in [
        f"Dist:    {tag['distance_cm']:+.1f} cm",
        f"Yaw:     {tag['yaw_deg']:+.1f}\u00b0",
        f"Lateral: {tag['lateral_cm']:+.1f} cm",
    ]:
        cv2.putText(frame, line, (cx - half, y),
                    FONT, 0.45, label_colour, 1, cv2.LINE_AA)
        y += 18

    # Steering arrow (chosen tag only)
    if is_chosen and move_target:
        angle_r   = math.radians(move_target["angle"])
        mag       = move_target["magnitude"]
        arrow_len = int(60 * mag)
        ex = cx + int(arrow_len * math.sin(angle_r))
        ey = cy - int(arrow_len * math.cos(angle_r))   # y-axis flipped in image
        cv2.arrowedLine(frame, (cx, cy), (ex, ey), CLR_ARROW, 2,
                        tipLength=0.3, line_type=cv2.LINE_AA)
        cv2.putText(frame,
                    f"Angle: {move_target['angle']:+.1f}\u00b0  Mag: {mag:.2f}",
                    (cx - half, y + 4),
                    FONT, 0.45, CLR_ARROW, 1, cv2.LINE_AA)


def _draw_crosshair(frame):
    h, w = frame.shape[:2]
    cx, cy = w // 2, h // 2
    cv2.line(frame, (cx - 15, cy), (cx + 15, cy), CLR_CROSSHAIR, 1)
    cv2.line(frame, (cx, cy - 15), (cx, cy + 15), CLR_CROSSHAIR, 1)
    cv2.circle(frame, (cx, cy), 2, CLR_CROSSHAIR, -1)


def _draw_fps(frame, fps: float):
    h, w = frame.shape[:2]
    label = f"FPS: {fps:.1f}"
    (tw, th), _ = cv2.getTextSize(label, FONT, 0.5, 1)
    cv2.putText(frame, label, (w - tw - 10, th + 8),
                FONT, 0.5, CLR_SUBTEXT, 1, cv2.LINE_AA)


def _draw_no_tag_banner(frame):
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 40), CLR_NOTAG, -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
    cv2.putText(frame, "NO VALID TAGS DETECTED",
                (w // 2 - 140, 28), FONT_BOLD, 0.75, CLR_SUBTEXT, 1, cv2.LINE_AA)


def _draw_chosen_summary(frame, result: dict):
    """Bottom-left translucent panel with chosen-tag summary."""
    mt    = result["moveTargetTag"]
    lines = [
        f"Chosen tag ID: {result['chosen_tag']['tag_id']}",
        f"Distance:  {result['distance_cm']:.1f} cm",
        f"Yaw:       {result['yaw_deg']:+.1f}\u00b0",
        f"Lateral:   {result['lateral_cm']:+.1f} cm",
        f"Steer:     {mt['angle']:+.1f}\u00b0  mag={mt['magnitude']:.2f}",
    ]
    if len(result["all_tags"]) > 1:
        other_ids = [t["tag_id"] for t in result["all_tags"]
                     if t["tag_id"] != result["chosen_tag"]["tag_id"]]
        lines.append(f"Other tags: {other_ids}")

    h  = frame.shape[0]
    y0 = h - len(lines) * 18 - 10
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, y0 - 6), (310, h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.45, frame, 0.55, 0, frame)
    for i, line in enumerate(lines):
        cv2.putText(frame, line, (8, y0 + i * 18),
                    FONT, 0.44, CLR_TEXT, 1, cv2.LINE_AA)


def annotate_frame(frame, result: dict, fps: float):
    """Overlay all detection info onto `frame` (modified in-place)."""
    _draw_crosshair(frame)
    _draw_fps(frame, fps)

    if not result["found"]:
        _draw_no_tag_banner(frame)
        return

    chosen_id  = result["chosen_tag"]["tag_id"]
    move_target = result["moveTargetTag"]

    # Draw non-chosen tags first so chosen renders on top
    for tag in result["all_tags"]:
        if tag["tag_id"] != chosen_id:
            _draw_tag(frame, tag, CLR_OTHER, is_chosen=False, move_target=None)

    _draw_tag(frame, result["chosen_tag"], CLR_CHOSEN,
              is_chosen=True, move_target=move_target)

    _draw_chosen_summary(frame, result)


# ── Capture & annotation thread ────────────────────────────────────────────────

def capture_loop(camera_index: int = 0):
    """Runs forever: grab frame → process → annotate → store JPEG bytes."""
    global _latest_frame

    navigator = AprilTagNavigator(
        target_tag_ids = cfg.APRILTAG_TARGET_IDS,
        camera_params  = cfg.APRILTAG_CAMERA_PARAMS,
        tag_size_m     = cfg.APRILTAG_TAG_SIZE_M,
        frame_size     = cfg.APRILTAG_FRAME_SIZE,
        tag_families   = cfg.APRILTAG_FAMILY,
    )

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open camera index {camera_index}. "
              "Try --camera 1 or check /dev/video*")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  cfg.APRILTAG_FRAME_SIZE[0])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg.APRILTAG_FRAME_SIZE[1])

    print(f"[INFO] Camera opened at index {camera_index}  "
          f"{cfg.APRILTAG_FRAME_SIZE[0]}×{cfg.APRILTAG_FRAME_SIZE[1]}")
    print(f"[INFO] Tracking tag IDs: {sorted(cfg.APRILTAG_TARGET_IDS)}")
    print(f"[INFO] Tag family: {cfg.APRILTAG_FAMILY}   "
          f"tag size: {cfg.APRILTAG_TAG_SIZE_M * 100:.0f} cm")

    fps_timer          = time.perf_counter()
    fps                = 0.0
    frame_count        = 0
    last_console_print = time.time()

    while True:
        ok, frame = cap.read()
        if not ok:
            print("[WARN] Frame grab failed – retrying …")
            time.sleep(0.1)
            continue

        result = navigator.process_frame(frame)

        # FPS counter
        frame_count += 1
        now = time.perf_counter()
        if now - fps_timer >= 1.0:
            fps         = frame_count / (now - fps_timer)
            frame_count = 0
            fps_timer   = now

        annotate_frame(frame, result, fps)

        ok2, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if ok2:
            with _frame_lock:
                _latest_frame = jpeg.tobytes()

        # Console at 1 Hz
        now_t = time.time()
        if now_t - last_console_print >= 1.0:
            last_console_print = now_t
            _console_print(result, fps)

    cap.release()


def _console_print(result: dict, fps: float):
    """Pretty-print process_frame() result to stdout."""
    sep = "─" * 52
    print(sep)
    print(f"  FPS: {fps:.1f}")
    if not result["found"]:
        print("  found: False  – no valid tags in frame")
    else:
        print(f"  found:       True")
        print(f"  chosen tag:  ID {result['chosen_tag']['tag_id']}")
        print(f"  distance_cm: {result['distance_cm']:.1f}")
        print(f"  yaw_deg:     {result['yaw_deg']:+.1f}")
        print(f"  lateral_cm:  {result['lateral_cm']:+.1f}")
        mt = result["moveTargetTag"]
        print(f"  moveTarget:  angle={mt['angle']:+.1f}°  magnitude={mt['magnitude']:.2f}")
        if len(result["all_tags"]) > 1:
            other_ids = [t["tag_id"] for t in result["all_tags"]
                         if t["tag_id"] != result["chosen_tag"]["tag_id"]]
            print(f"  other tags:  {other_ids}")
    print(sep)


# ── Flask routes ───────────────────────────────────────────────────────────────

def _mjpeg_generator():
    while True:
        with _frame_lock:
            frame = _latest_frame
        if frame is None:
            time.sleep(0.01)
            continue
        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" +
            frame +
            b"\r\n"
        )
        time.sleep(0.01)


@app.route("/")
def index():
    return (
        "<!DOCTYPE html><html><head>"
        "<title>AprilTag Test Stream</title>"
        "<style>body{margin:0;background:#111;display:flex;"
        "flex-direction:column;align-items:center;color:#ddd;"
        "font-family:monospace;}</style></head><body>"
        "<h2 style='margin:12px 0 6px'>AprilTag Navigator \u2013 Live View</h2>"
        "<img src='/stream' style='max-width:100%;border:2px solid #444'>"
        "<p style='font-size:0.8em;color:#888;margin-top:8px'>"
        "Green = chosen tag &nbsp;|&nbsp; Amber = other valid tags &nbsp;|&nbsp;"
        "Blue arrow = steering direction</p>"
        "</body></html>"
    )


@app.route("/stream")
def stream():
    return Response(
        _mjpeg_generator(),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Headless AprilTag detection test stream")
    parser.add_argument("--camera", type=int, default=0,
                        help="cv2.VideoCapture index (default: 0)")
    parser.add_argument("--port",   type=int, default=5000,
                        help="Flask port (default: 5000)")
    parser.add_argument("--host",   type=str, default="0.0.0.0",
                        help="Flask bind address (default: 0.0.0.0)")
    args = parser.parse_args()

    t = threading.Thread(target=capture_loop, args=(args.camera,), daemon=True)
    t.start()

    print()
    print("=" * 52)
    print("  AprilTag Navigator \u2014 headless test stream")
    print("=" * 52)
    print(f"  Open in browser:  http://<pi-ip>:{args.port}")
    print(f"  Camera index:     {args.camera}")
    print(f"  Press Ctrl+C to stop")
    print("=" * 52)
    print()

    app.run(host=args.host, port=args.port, threaded=True)