#!/usr/bin/env python3
"""
apriltag_debug_viewer.py  –  Headless AprilTag Detection Debug Tool
=====================================================================
No display / X server required.  Works over plain SSH.

What it does
------------
  * Runs the detector on every camera frame.
  * Prints a status line to the terminal every second.
  * Saves an annotated JPEG (debug_frame.jpg) every SAVE_INTERVAL_S seconds
    so you can scp / sftp it to your laptop and inspect it visually.
  * Exits cleanly on Ctrl-C.

Usage
-----
    python3 apriltag_debug_viewer.py

Grab the latest annotated frame from another terminal on your laptop:
    scp nagim@raspberrypi:~/extraTest/debug_frame.jpg .
"""

import cv2
import math
import time
import signal
import sys
from pupil_apriltags import Detector

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

WATCH_IDS        = {18, 19, 20}
TAG_FAMILY       = "tag36h11"
TAG_SIZE_M       = 0.1

CAMERA_PARAMS    = (564.98, 565.87, 303.59, 282.69)   # fx, fy, cx, cy
FRAME_W, FRAME_H = 640, 480
CAMERA_INDEX     = 0

SAVE_INTERVAL_S  = 2.0          # overwrite debug_frame.jpg this often
OUTPUT_PATH      = "debug_frame.jpg"

# Lock thresholds
LOCK_DISTANCE_CM    = 15.0
LOCK_CENTER_TOL_PX  = 40
LOCK_YAW_DEG        = 10.0

# ─────────────────────────────────────────────────────────────────────────────
# Detector
# ─────────────────────────────────────────────────────────────────────────────

detector = Detector(
    families          = TAG_FAMILY,
    nthreads          = 2,
    quad_decimate     = 1.0,
    quad_sigma        = 0.0,
    refine_edges      = 1,
    decode_sharpening = 0.25,
    debug             = 0,
)

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def compute_yaw_deg(R):
    return math.degrees(math.atan2(R[0, 2], R[2, 2]))

def tick(ok):
    return "OK " if ok else "---"

def tick_color(ok):
    return (0, 210, 0) if ok else (0, 60, 220)


def annotate_frame(frame, results, frame_cx):
    """Draw detection info onto frame in-place. Returns list of watched tag dicts."""
    # Cross-hair at image centre
    cv2.drawMarker(
        frame, (int(frame_cx), FRAME_H // 2),
        (200, 200, 200), cv2.MARKER_CROSS, 20, 1,
    )

    watched = []

    for tag in results:
        tid = int(tag.tag_id)

        # Non-watched tags: faint outline only
        if tid not in WATCH_IDS:
            pts = tag.corners.astype(int).reshape((-1, 1, 2))
            cv2.polylines(frame, [pts], True, (80, 80, 80), 1)
            cv2.putText(
                frame, f"ID {tid}",
                (int(tag.center[0]) + 6, int(tag.center[1])),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (80, 80, 80), 1,
            )
            continue

        # ── Pose ─────────────────────────────────────────────────────────
        x_cm        = float(tag.pose_t[0][0]) * 100.0
        z_cm        = float(tag.pose_t[2][0]) * 100.0
        distance_cm = math.sqrt(x_cm ** 2 + z_cm ** 2)
        yaw_deg     = compute_yaw_deg(tag.pose_R)
        cx          = int(tag.center[0])
        cy          = int(tag.center[1])
        err_px      = cx - int(frame_cx)

        ok_dist     = distance_cm  <= LOCK_DISTANCE_CM
        ok_centre   = abs(err_px)  <= LOCK_CENTER_TOL_PX
        ok_yaw      = abs(yaw_deg) <= LOCK_YAW_DEG
        locked      = ok_dist and ok_centre and ok_yaw

        watched.append(dict(
            tag_id=tid, cx=cx, cy=cy,
            x_cm=x_cm, z_cm=z_cm,
            distance_cm=distance_cm, yaw_deg=yaw_deg,
            err_px=err_px, locked=locked,
        ))

        # ── Bounding box ─────────────────────────────────────────────────
        pts     = tag.corners.astype(int).reshape((-1, 1, 2))
        box_col = (0, 220, 0) if locked else (0, 200, 255)
        cv2.polylines(frame, [pts], True, box_col, 2)

        # ── Centre dot + line to image centre ────────────────────────────
        cv2.circle(frame, (cx, cy), 6, box_col, -1)
        cv2.line(frame, (int(frame_cx), FRAME_H // 2), (cx, cy), (160, 160, 160), 1)

        # ── Text block ────────────────────────────────────────────────────
        lines = [
            (f"ID {tid}",                                             (0, 200, 255)),
            (f"dist  : {distance_cm:6.1f} cm  [{tick(ok_dist)}]",    tick_color(ok_dist)),
            (f"x_off : {x_cm:+6.1f} cm",                             (230, 230, 230)),
            (f"z_fwd : {z_cm:6.1f} cm",                              (230, 230, 230)),
            (f"centre: {err_px:+5d} px  [{tick(ok_centre)}]",        tick_color(ok_centre)),
            (f"yaw   : {yaw_deg:+6.1f} deg  [{tick(ok_yaw)}]",       tick_color(ok_yaw)),
            (f"LOCKED: {'YES' if locked else 'NO'}",                  (0, 230, 0) if locked else (0, 60, 240)),
        ]

        tx = min(cx + 14, FRAME_W - 210)
        ty = max(cy - 62, 20)

        for i, (text, col) in enumerate(lines):
            y = ty + i * 19
            cv2.putText(frame, text, (tx + 1, y + 1),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.46, (0, 0, 0), 2)
            cv2.putText(frame, text, (tx, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.46, col, 1)

    # ── Status bar (top) ──────────────────────────────────────────────────
    if watched:
        ids_str      = " ".join(str(t["tag_id"]) for t in watched)
        status_text  = f"FOUND: ID(s) {ids_str}"
        status_color = (0, 220, 0)
    else:
        other_ids    = [int(t.tag_id) for t in results if int(t.tag_id) not in WATCH_IDS]
        status_text  = f"NOT FOUND  |  other IDs in frame: {other_ids if other_ids else 'none'}"
        status_color = (0, 60, 230)

    cv2.rectangle(frame, (0, 0), (FRAME_W, 22), (0, 0, 0), -1)
    cv2.putText(frame, status_text, (6, 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, status_color, 1)

    # ── Footer (bottom) ───────────────────────────────────────────────────
    footer = (f"family={TAG_FAMILY}  size={TAG_SIZE_M*100:.0f}cm  "
              f"lock: d<={LOCK_DISTANCE_CM}cm  c<={LOCK_CENTER_TOL_PX}px  y<={LOCK_YAW_DEG}deg")
    cv2.rectangle(frame, (0, FRAME_H - 18), (FRAME_W, FRAME_H), (0, 0, 0), -1)
    cv2.putText(frame, footer, (4, FRAME_H - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.36, (160, 160, 160), 1)

    return watched


def print_status(watched, results, fps):
    if watched:
        for t in watched:
            print(
                f"[TAG {t['tag_id']:>3}]  "
                f"dist={t['distance_cm']:6.1f}cm  "
                f"x={t['x_cm']:+6.1f}cm  "
                f"z={t['z_cm']:6.1f}cm  "
                f"yaw={t['yaw_deg']:+6.1f}deg  "
                f"err={t['err_px']:+4d}px  "
                f"locked={t['locked']}  "
                f"fps={fps:.1f}"
            )
    else:
        other = [int(t.tag_id) for t in results if int(t.tag_id) not in WATCH_IDS]
        print(
            f"[-----]  No watched tag {sorted(WATCH_IDS)} detected  "
            f"other_ids={other if other else []}  "
            f"fps={fps:.1f}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  FRAME_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    if not cap.isOpened():
        print(f"[ERROR] Cannot open camera index {CAMERA_INDEX}.")
        sys.exit(1)

    signal.signal(signal.SIGINT, lambda *_: sys.exit(0))

    print(f"[INFO] Watching for tag IDs : {sorted(WATCH_IDS)}")
    print(f"[INFO] Tag family           : {TAG_FAMILY}")
    print(f"[INFO] Tag size             : {TAG_SIZE_M * 100:.0f} cm")
    print(f"[INFO] Saving annotated JPEG to '{OUTPUT_PATH}' every {SAVE_INTERVAL_S}s")
    print( "[INFO] Fetch it from your laptop with:")
    print(f"       scp nagim@raspberrypi:~/extraTest/{OUTPUT_PATH} .")
    print( "[INFO] Press Ctrl-C to stop.\n")

    frame_cx    = FRAME_W / 2
    frame_count = 0
    last_log_t  = time.time()
    last_save_t = time.time()

    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                print("[WARN] cap.read() failed – retrying…")
                time.sleep(0.05)
                continue

            frame_count += 1
            gray    = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            results = detector.detect(
                gray,
                estimate_tag_pose = True,
                camera_params     = CAMERA_PARAMS,
                tag_size          = TAG_SIZE_M,
            )

            watched = annotate_frame(frame, results, frame_cx)

            now = time.time()

            # Terminal log once per second
            if now - last_log_t >= 1.0:
                fps = frame_count / (now - last_log_t)
                print_status(watched, results, fps)
                frame_count = 0
                last_log_t  = now

            # Save annotated JPEG periodically
            if now - last_save_t >= SAVE_INTERVAL_S:
                cv2.imwrite(OUTPUT_PATH, frame)
                last_save_t = now

    finally:
        cap.release()
        print(f"\n[INFO] Stopped.  Last frame saved to '{OUTPUT_PATH}'.")


if __name__ == "__main__":
    main()