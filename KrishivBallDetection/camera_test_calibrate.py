"""
Camera Test & Calibration Utility
===================================
Two calibration modes:

MODE 1 — Quick calibration with a ping-pong ball (no printout needed)
    1) Place an orange ping-pong ball at a MEASURED distance (e.g. 30 cm).
    2) The script detects the ball and measures its pixel diameter.
    3) Press SPACE to capture a sample. Take 5+ samples at different distances.
    4) Press K to compute focal length.

MODE 2 — Full checkerboard calibration (needs a printed/phone checkerboard)
    1) Show a 9x6 inner-corner checkerboard to the camera.
    2) Press C to capture. Take 15-20 captures at different angles.
    3) Press K to compute full camera matrix + distortion.

Run:
    python camera_test_calibrate.py           # quick mode (default)
    python camera_test_calibrate.py --board    # checkerboard mode
    python camera_test_calibrate.py 1          # use camera index 1
    python camera_test_calibrate.py 1 --board  # camera 1 + checkerboard
"""

import cv2
import numpy as np
import sys
import math

# Checkerboard inner-corner dimensions (columns x rows).
BOARD_SIZE = (9, 6)
SQUARE_SIZE_MM = 25.0

FRAME_W, FRAME_H = 640, 480

# Ping-pong ball real diameter and HSV range (orange)
BALL_REAL_DIAMETER_CM = 4.0
BALL_HSV_LOWER = np.array([5, 120, 120])
BALL_HSV_UPPER = np.array([25, 255, 255])


def scan_cameras(max_index: int = 5):
    """Try opening camera indices 0..max_index-1 and report which work."""
    found = []
    for idx in range(max_index):
        cap = cv2.VideoCapture(idx)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                h, w = frame.shape[:2]
                name = cap.getBackendName()
                found.append((idx, w, h, name))
                print(f"  [OK]  Index {idx}: {w}x{h}  backend={name}")
            else:
                print(f"  [--]  Index {idx}: opened but no frame")
            cap.release()
        else:
            print(f"  [--]  Index {idx}: cannot open")
    return found


def print_camera_props(cap: cv2.VideoCapture):
    """Print useful camera properties."""
    props = {
        "Frame width":  cv2.CAP_PROP_FRAME_WIDTH,
        "Frame height": cv2.CAP_PROP_FRAME_HEIGHT,
        "FPS":          cv2.CAP_PROP_FPS,
        "Backend":      cv2.CAP_PROP_BACKEND,
        "Auto-expose":  cv2.CAP_PROP_AUTO_EXPOSURE,
        "Brightness":   cv2.CAP_PROP_BRIGHTNESS,
        "Contrast":     cv2.CAP_PROP_CONTRAST,
        "Saturation":   cv2.CAP_PROP_SATURATION,
        "Gain":         cv2.CAP_PROP_GAIN,
    }
    print("\n  Camera properties:")
    for name, prop_id in props.items():
        val = cap.get(prop_id)
        print(f"    {name:16s} = {val}")


def calibrate_checkerboard(obj_points, img_points, img_size):
    """Run full OpenCV camera calibration and return results."""
    ret, cam_mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
        obj_points, img_points, img_size, None, None
    )
    print("\n" + "=" * 56)
    print("  CHECKERBOARD CALIBRATION  (reproj error: {:.4f})".format(ret))
    print("=" * 56)

    fx, fy = cam_mtx[0, 0], cam_mtx[1, 1]
    cx, cy = cam_mtx[0, 2], cam_mtx[1, 2]

    print(f"\n  Focal length : fx={fx:.2f}  fy={fy:.2f} px")
    print(f"  Principal pt : cx={cx:.2f}  cy={cy:.2f} px")
    print(f"  Distortion   : {dist.ravel()}")
    _print_paste_block(fx, fy, cx, cy, dist.ravel())
    return cam_mtx, dist


def calibrate_from_ball(samples):
    """Compute focal length from ball samples: f = (d_px * z_cm) / D_real."""
    focals = []
    for d_px, z_cm in samples:
        f = (d_px * z_cm) / BALL_REAL_DIAMETER_CM
        focals.append(f)

    f_mean = np.mean(focals)
    f_std = np.std(focals)

    print("\n" + "=" * 56)
    print("  QUICK CALIBRATION (ping-pong ball)")
    print("=" * 56)
    print(f"\n  Samples: {len(samples)}")
    for i, (d_px, z_cm) in enumerate(samples):
        f = (d_px * z_cm) / BALL_REAL_DIAMETER_CM
        print(f"    #{i+1}  diameter={d_px:.1f}px  dist={z_cm:.0f}cm  -> f={f:.1f}px")
    print(f"\n  Focal length : {f_mean:.1f} +/- {f_std:.1f} px")
    print(f"  (assuming zero distortion)")

    cx, cy = FRAME_W / 2.0, FRAME_H / 2.0
    _print_paste_block(f_mean, f_mean, cx, cy, np.zeros(5))


def _print_paste_block(fx, fy, cx, cy, d):
    print("\n  ---- Paste into ball_detection_krishiv.py ----\n")
    print(f"CAMERA_MATRIX = np.array(")
    print(f"    [[{fx:.1f}, 0.0, {cx:.1f}],")
    print(f"     [0.0, {fy:.1f}, {cy:.1f}],")
    print(f"     [0.0, 0.0, 1.0]], dtype=np.float64,")
    print(f")")
    print(f"DIST_COEFFS = np.array([{d[0]:.6f}, {d[1]:.6f}, {d[2]:.6f}, {d[3]:.6f}, {d[4]:.6f}], dtype=np.float64)")
    print()


def _detect_ball(frame):
    """Detect the largest orange ball, return (cx, cy, radius) or None."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, BALL_HSV_LOWER, BALL_HSV_UPPER)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best = None
    best_area = 0
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 100:
            continue
        perim = cv2.arcLength(cnt, True)
        if perim == 0:
            continue
        circ = 4.0 * math.pi * area / (perim * perim)
        if circ < 0.5:
            continue
        if area > best_area:
            best_area = area
            (cx, cy), radius = cv2.minEnclosingCircle(cnt)
            best = (int(cx), int(cy), radius)
    return best


# =====================================================================
# Quick calibration mode (ping-pong ball)
# =====================================================================

def run_quick_mode(cap):
    """Live preview — detect ball, user enters distance, compute focal length."""
    samples = []  # list of (pixel_diameter, distance_cm)
    input_buf = ""
    awaiting_distance = False
    last_ball = None

    print("\n  QUICK CALIBRATION MODE")
    print("  Place orange ping-pong ball in view at a MEASURED distance.")
    print("  SPACE = capture (then type distance in cm + ENTER)")
    print("  K     = compute focal length from all samples")
    print("  Q     = quit\n")

    while True:
        ok, frame = cap.read()
        if not ok:
            continue

        display = frame.copy()
        ball = _detect_ball(frame)

        if ball:
            cx, cy, r = ball
            cv2.circle(display, (cx, cy), int(r), (0, 255, 0), 2)
            cv2.circle(display, (cx, cy), 2, (0, 0, 255), 3)
            diam_px = r * 2
            cv2.putText(display, f"Ball: {diam_px:.0f}px diameter",
                        (cx - 60, cy - int(r) - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 255, 0), 2)
            last_ball = ball

        if awaiting_distance:
            cv2.putText(display, f"Type distance (cm): {input_buf}_",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        elif ball:
            cv2.putText(display, "SPACE to capture  |  K to calibrate",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        else:
            cv2.putText(display, "No ball detected — show orange ping-pong ball",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)

        cv2.putText(display, f"Samples: {len(samples)}",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        cv2.putText(display, "SPACE:capture  K:calibrate  Q:quit",
                    (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

        cv2.imshow("Camera Test — Quick Calibration", display)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break

        elif awaiting_distance:
            if key == 13:  # ENTER
                try:
                    z_cm = float(input_buf)
                    if z_cm > 0 and last_ball:
                        diam_px = last_ball[2] * 2
                        samples.append((diam_px, z_cm))
                        print(f"  Sample #{len(samples)}: {diam_px:.1f}px at {z_cm}cm")
                except ValueError:
                    print("  Invalid number, try again.")
                input_buf = ""
                awaiting_distance = False
            elif key == 27:  # ESC to cancel
                input_buf = ""
                awaiting_distance = False
            elif key == 8:  # backspace
                input_buf = input_buf[:-1]
            elif 32 <= key < 127:
                input_buf += chr(key)

        elif key == ord(' ') and last_ball:
            awaiting_distance = True
            input_buf = ""

        elif key == ord('k'):
            if len(samples) < 2:
                print(f"  Need at least 2 samples (have {len(samples)}).")
            else:
                calibrate_from_ball(samples)


# =====================================================================
# Checkerboard calibration mode
# =====================================================================

def run_board_mode(cap):
    """Standard checkerboard calibration."""
    objp = np.zeros((BOARD_SIZE[0] * BOARD_SIZE[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:BOARD_SIZE[0], 0:BOARD_SIZE[1]].T.reshape(-1, 2)
    objp *= SQUARE_SIZE_MM

    obj_points = []
    img_points = []

    print("\n  CHECKERBOARD CALIBRATION MODE")
    print("  C = capture checkerboard frame")
    print("  K = calibrate with captured frames")
    print("  Q = quit\n")

    while True:
        ok, frame = cap.read()
        if not ok:
            continue

        display = frame.copy()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        found_board, corners = cv2.findChessboardCorners(
            gray, BOARD_SIZE,
            cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_FAST_CHECK
        )
        if found_board:
            cv2.drawChessboardCorners(display, BOARD_SIZE, corners, True)
            cv2.putText(display, "BOARD DETECTED — press C to capture",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        else:
            cv2.putText(display, "No checkerboard found",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        cv2.putText(display, f"Captures: {len(img_points)}",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        cv2.putText(display, "C:capture  K:calibrate  Q:quit",
                    (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

        cv2.imshow("Camera Test — Checkerboard", display)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break
        elif key == ord('c') and found_board:
            criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            obj_points.append(objp)
            img_points.append(corners2)
            print(f"  Captured frame {len(img_points)}")
        elif key == ord('k'):
            if len(img_points) < 5:
                print(f"  Need at least 5 captures (have {len(img_points)}).")
            else:
                calibrate_checkerboard(obj_points, img_points, (FRAME_W, FRAME_H))


# =====================================================================
# Main
# =====================================================================

def main():
    use_board = "--board" in sys.argv
    args = [a for a in sys.argv[1:] if not a.startswith("--")]

    #print("\nScanning for cameras...")
    #found = scan_cameras()
    #if not found:
    #    print("No cameras detected! Check USB connection.")
   #     sys.exit(1)

    cam_idx = 1
    if args:
        cam_idx = int(args[0])
    print(f"\nUsing camera index {cam_idx}")

    cap = cv2.VideoCapture(cam_idx)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)

    if not cap.isOpened():
        print(f"ERROR: cannot open camera {cam_idx}")
        sys.exit(1)

    print_camera_props(cap)

    if use_board:
        run_board_mode(cap)
    else:
        run_quick_mode(cap)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
