#!/usr/bin/env python3
"""
test_apriltag_navigator.py
==========================
Stand-alone test harness for AprilTagNavigator on a laptop webcam.

Tags under test : IDs 20, 21, 22  |  size 10 × 10 cm  |  family tag36h11
Camera          : laptop default webcam (index 0)

Keyboard controls (press while the OpenCV window is focused)
------------------------------------------------------------
  S   – save a snapshot of the current frame + console output
  C   – print a short calibration-quality report to the terminal
  Q   – quit

What is shown on screen
-----------------------
  • Red dot + ID + distance label on every detected valid tag
  • Blue circle (approaching) or Yellow circle (LOCKED) on chosen tag
  • Top-left: lockedTag status and current mode
  • Top-right: angle_deg / magnitude of moveTargetTag (if any)
  • Centre crosshair: image centre reference (alignment aid)

Console output (every frame a tag is visible)
---------------------------------------------
  [TRACK]   angle=+12.3°  mag=0.74  dist=45.1 cm  yaw=3.2°  id=20
  [LOCKED]  angle=None    mag=None  dist=13.8 cm  yaw=1.1°  id=21
  [SEARCH]  no valid tags in frame

Run
---
  pip install opencv-python pupil-apriltags
  python test_apriltag_navigator.py
"""

import math
import sys
import time
import cv2
import numpy as np

# ── Import the class under test ───────────────────────────────────────────────
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))  # adds UNIBOTS/ to path

from AprilTagNavigator_GT2 import AprilTagNavigator

# ─────────────────────────────────────────────────────────────────────────────
# Configuration  –  edit these to match your setup
# ─────────────────────────────────────────────────────────────────────────────

# YOUR PRINTED TAG IDs (not the robot's basket IDs which are 1,2)
TARGET_IDS   = {20, 21, 22}
TAG_FAMILY   = "tag36h11"
TAG_SIZE_M   = 0.10        # 10 cm side length

# Camera intrinsics.
# If you have run OpenCV camera calibration, paste your (fx, fy, cx, cy) here.
# If not, the script estimates fx/fy from FOV and falls back to frame-centre for cx/cy.
# The fallback is good enough to validate angle/distance ORDER OF MAGNITUDE.
USE_CALIBRATED_PARAMS = False   # ← set True and fill in below once calibrated
CALIBRATED_PARAMS = (
    800.0,   # fx  ← replace with your calibration result
    800.0,   # fy  ← replace with your calibration result
    320.0,   # cx  ← replace with your calibration result
    240.0,   # cy  ← replace with your calibration result
)

CAMERA_INDEX  = 0            # 0 = first webcam on the laptop
FRAME_W, FRAME_H = 640, 480  # capture resolution

# Lock thresholds  (match what the robot will use)
LOCK_DISTANCE_CM   = 15.0
LOCK_CENTER_TOL_PX = 40
LOCK_YAW_DEG       = 10.0

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def estimate_camera_params(frame_w: int, frame_h: int,
                            hfov_deg: float = 70.0) -> tuple:
    """
    Estimate (fx, fy, cx, cy) from a known horizontal field-of-view.
    Most laptop webcams have an hFOV between 60° and 78°.
    70° is a reasonable midpoint; adjust if you know your exact FOV.
    """
    fx = (frame_w / 2.0) / math.tan(math.radians(hfov_deg / 2.0))
    fy = fx   # square pixels assumed
    cx = frame_w  / 2.0
    cy = frame_h / 2.0
    return (fx, fy, cx, cy)


def draw_crosshair(frame: np.ndarray) -> None:
    """Draw a thin crosshair at the image centre for alignment reference."""
    h, w = frame.shape[:2]
    cx, cy = w // 2, h // 2
    cv2.line(frame, (cx - 20, cy), (cx + 20, cy), (180, 180, 180), 1)
    cv2.line(frame, (cx, cy - 20), (cx, cy + 20), (180, 180, 180), 1)


def draw_overlay(frame: np.ndarray, result: dict, nav: AprilTagNavigator) -> None:
    """Render all debug visuals onto *frame* in-place."""
    h, w = frame.shape[:2]

    # ── Crosshair ────────────────────────────────────────────────────────────
    draw_crosshair(frame)

    # ── Lock / mode status (top-left) ────────────────────────────────────────
    if not result["found"]:
        mode_text  = "SEARCH – no tags"
        mode_color = (100, 100, 100)
    elif result["lockedTag"]:
        mode_text  = "LOCKED"
        mode_color = (0, 255, 255)
    else:
        mode_text  = "TRACK"
        mode_color = (0, 200, 50)

    cv2.putText(frame, mode_text, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, mode_color, 2)

    # ── moveTargetTag values (top-right) ─────────────────────────────────────
    mt = result.get("moveTargetTag")
    if mt:
        ang_txt = f"angle={mt['angle']:+.1f} deg"
        mag_txt = f"mag={mt['magnitude']:.2f}"
        cv2.putText(frame, ang_txt, (w - 220, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 0), 2)
        cv2.putText(frame, mag_txt, (w - 220, 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 0), 2)

    # ── Draw all valid tags ───────────────────────────────────────────────────
    for tag in result["all_tags"]:
        cx, cy = tag["centre_x"], tag["centre_y"]
        cv2.circle(frame, (cx, cy), 6, (0, 0, 255), -1)
        label = f"ID{tag['tag_id']}  {tag['distance_cm']:.1f}cm  yaw={tag['yaw_deg']:.1f}d"
        cv2.putText(frame, label, (cx + 10, cy - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

    # ── Highlight chosen tag ─────────────────────────────────────────────────
    ct = result.get("chosen_tag")
    if ct:
        cx, cy = ct["centre_x"], ct["centre_y"]
        ring_color = (0, 255, 255) if result["lockedTag"] else (255, 60, 0)
        cv2.circle(frame, (cx, cy), 22, ring_color, 3)

        # Tolerance box around image centre
        img_cx = w // 2
        box_x1 = img_cx - LOCK_CENTER_TOL_PX
        box_x2 = img_cx + LOCK_CENTER_TOL_PX
        cv2.line(frame, (box_x1, h - 10), (box_x1, h - 40), (80, 80, 200), 1)
        cv2.line(frame, (box_x2, h - 10), (box_x2, h - 40), (80, 80, 200), 1)
        cv2.putText(frame, f"<- tol {LOCK_CENTER_TOL_PX}px ->",
                    (box_x1, h - 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (80, 80, 200), 1)

    # ── Keys legend (bottom-left) ────────────────────────────────────────────
    cv2.putText(frame, "S=snapshot  C=cal report  Q=quit",
                (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (160, 160, 160), 1)


def console_report(result: dict) -> str:
    """Return a one-line summary for the terminal."""
    if not result["found"]:
        return "[SEARCH]  no valid tags in frame"

    ct = result["chosen_tag"]
    mt = result.get("moveTargetTag")

    mode = "LOCKED" if result["lockedTag"] else "TRACK "
    angle_str = f"{mt['angle']:+.1f}°" if mt else "None    "
    mag_str   = f"{mt['magnitude']:.2f}"   if mt else "None"

    return (
        f"[{mode}]  "
        f"angle={angle_str}  mag={mag_str}  "
        f"dist={ct['distance_cm']:.1f}cm  "
        f"yaw={ct['yaw_deg']:+.1f}°  "
        f"id={ct['tag_id']}  "
        f"cx_err={ct['centre_x'] - FRAME_W//2:+d}px"
    )


def calibration_report(result: dict, camera_params: tuple) -> None:
    """Print a structured calibration quality snapshot to the terminal."""
    fx, fy, cx, cy = camera_params
    print("\n" + "=" * 60)
    print("CALIBRATION / SANITY REPORT")
    print("=" * 60)
    print(f"  Camera params used : fx={fx:.1f}  fy={fy:.1f}  cx={cx:.1f}  cy={cy:.1f}")
    print(f"  Frame size         : {FRAME_W} x {FRAME_H}")
    print(f"  Tag size           : {TAG_SIZE_M*100:.0f} cm")
    print(f"  Target IDs         : {TARGET_IDS}")

    if not result["found"]:
        print("  >> No tag visible – hold a tag in frame then press C again.")
    else:
        ct = result["chosen_tag"]
        print(f"\n  Chosen tag (ID {ct['tag_id']})")
        print(f"    distance  : {ct['distance_cm']:.1f} cm  "
              f"({'GOOD – within lock range' if ct['distance_cm'] <= LOCK_DISTANCE_CM else 'outside lock range'})")
        print(f"    x offset  : {ct['x_cm']:.1f} cm  "
              f"(+right / -left from camera centre)")
        print(f"    angle     : ", end="")
        mt = result.get("moveTargetTag")
        if mt:
            print(f"{mt['angle']:+.1f}°  (0=straight, +ve=right)")
        else:
            print("None (LOCKED – angle satisfied)")
        print(f"    yaw       : {ct['yaw_deg']:+.1f}°  "
              f"({'GOOD' if abs(ct['yaw_deg']) <= LOCK_YAW_DEG else 'too oblique'})")
        img_cx_err = ct['centre_x'] - FRAME_W // 2
        print(f"    centre_err: {img_cx_err:+d} px  "
              f"({'GOOD' if abs(img_cx_err) <= LOCK_CENTER_TOL_PX else 'outside tolerance'})")
        print(f"    lockedTag : {result['lockedTag']}")

    print("=" * 60 + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    # ── Choose camera params ──────────────────────────────────────────────────
    if USE_CALIBRATED_PARAMS:
        camera_params = CALIBRATED_PARAMS
        print(f"[INFO] Using calibrated camera params: {camera_params}")
    else:
        camera_params = estimate_camera_params(FRAME_W, FRAME_H, hfov_deg=70.0)
        print(f"[INFO] Using ESTIMATED camera params (hFOV=70°): {camera_params}")
        print("[INFO] Set USE_CALIBRATED_PARAMS=True and fill CALIBRATED_PARAMS "
              "once you have real calibration values.\n")

    # ── Instantiate navigator ─────────────────────────────────────────────────
    nav = AprilTagNavigator(
        target_tag_ids          = TARGET_IDS,
        camera_params           = camera_params,
        tag_size_m              = TAG_SIZE_M,
        frame_size              = (FRAME_W, FRAME_H),
        lock_distance_cm        = LOCK_DISTANCE_CM,
        lock_center_tolerance_px= LOCK_CENTER_TOL_PX,
        lock_yaw_deg            = LOCK_YAW_DEG,
        tag_families            = TAG_FAMILY,
    )

    # ── Open webcam ───────────────────────────────────────────────────────────
    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    if not cap.isOpened():
        sys.exit(f"ERROR: Cannot open camera index {CAMERA_INDEX}.")

    print("AprilTag Navigator test running.  Keys: S=snapshot  C=cal-report  Q=quit\n")

    snapshot_count = 0
    last_console_time = 0.0
    last_result = {"found": False, "lockedTag": False,
                   "moveTargetTag": None, "chosen_tag": None, "all_tags": []}

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            print("[WARN] cap.read() failed – retrying...")
            time.sleep(0.05)
            continue

        # ── Run detector ──────────────────────────────────────────────────────
        try:
            result = nav.process_frame(frame)
        except Exception as exc:
            print(f"[ERROR] process_frame raised: {exc}")
            result = {"found": False, "lockedTag": False,
                      "moveTargetTag": None, "chosen_tag": None, "all_tags": []}

        last_result = result

        # ── Overlay + display ─────────────────────────────────────────────────
        display = frame.copy()
        draw_overlay(display, result, nav)
        cv2.imshow("AprilTag Navigator Test", display)

        # ── Throttled console output (max 5 Hz) ───────────────────────────────
        now = time.time()
        if now - last_console_time >= 0.2:
            print(console_report(result))
            last_console_time = now

        # ── Key handling ──────────────────────────────────────────────────────
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q') or key == 27:
            break

        elif key == ord('s'):
            fname = f"snapshot_{snapshot_count:03d}.png"
            cv2.imwrite(fname, display)
            snapshot_count += 1
            print(f"[SNAPSHOT] Saved {fname}")
            print(f"           {console_report(result)}")

        elif key == ord('c'):
            calibration_report(result, camera_params)

    cap.release()
    cv2.destroyAllWindows()
    print("\nTest session ended.")


if __name__ == "__main__":
    main()