"""
Camera Test & Calibration Utility
===================================
1) Scans camera indices 0-4 to find your USB camera.
2) Shows a live preview so you can verify it works.
3) Press 'C' to capture checkerboard frames for calibration
   (print a 9x6 checkerboard, hold it in front of the camera at
    various angles — aim for ~15-20 captures).
4) Press 'K' when done capturing to compute camera matrix &
   distortion coefficients.
5) Press 'Q' to quit.

The calibration output can be pasted straight into
ball_detection_krishiv.py to replace the placeholder values.
"""

import cv2
import numpy as np
import sys
import os

# Checkerboard inner-corner dimensions (columns x rows).
# A standard printable checkerboard is 10x7 squares → 9x6 inner corners.
BOARD_SIZE = (9, 6)
SQUARE_SIZE_MM = 25.0  # side length of one square on your printed board

FRAME_W, FRAME_H = 640, 480


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


def calibrate(obj_points, img_points, img_size):
    """Run OpenCV camera calibration and return results."""
    ret, cam_mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
        obj_points, img_points, img_size, None, None
    )
    print("\n" + "=" * 56)
    print("  CALIBRATION RESULTS  (reprojection error: {:.4f})".format(ret))
    print("=" * 56)

    fx = cam_mtx[0, 0]
    fy = cam_mtx[1, 1]
    cx = cam_mtx[0, 2]
    cy = cam_mtx[1, 2]

    print(f"\n  Focal length : fx={fx:.2f}  fy={fy:.2f} px")
    print(f"  Principal pt : cx={cx:.2f}  cy={cy:.2f} px")
    print(f"  Distortion   : {dist.ravel()}")

    print("\n  ---- Paste into ball_detection_krishiv.py ----\n")
    print(f"CAMERA_MATRIX = np.array(")
    print(f"    [[{fx:.1f}, 0.0, {cx:.1f}],")
    print(f"     [0.0, {fy:.1f}, {cy:.1f}],")
    print(f"     [0.0, 0.0, 1.0]], dtype=np.float64,")
    print(f")")
    d = dist.ravel()
    print(f"DIST_COEFFS = np.array([{d[0]:.6f}, {d[1]:.6f}, {d[2]:.6f}, {d[3]:.6f}, {d[4]:.6f}], dtype=np.float64)")
    print()
    return cam_mtx, dist


def main():
    # ---- Step 1: Scan for cameras ----
    print("\nScanning for cameras...")
    found = scan_cameras()
    if not found:
        print("No cameras detected! Check USB connection.")
        sys.exit(1)

    # Pick camera — prefer the first one, or let user override via arg
    cam_idx = found[0][0]
    if len(sys.argv) > 1:
        cam_idx = int(sys.argv[1])
    print(f"\nUsing camera index {cam_idx}")

    cap = cv2.VideoCapture(cam_idx)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)

    if not cap.isOpened():
        print(f"ERROR: cannot open camera {cam_idx}")
        sys.exit(1)

    print_camera_props(cap)

    # ---- Step 2: Live preview + checkerboard capture ----
    # Prepare object points (3D corners on a flat board)
    objp = np.zeros((BOARD_SIZE[0] * BOARD_SIZE[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:BOARD_SIZE[0], 0:BOARD_SIZE[1]].T.reshape(-1, 2)
    objp *= SQUARE_SIZE_MM

    obj_points = []   # 3D points in real-world space
    img_points = []   # 2D points in image plane

    print("\n  Live preview running.")
    print("  C = capture checkerboard frame")
    print("  K = calibrate with captured frames")
    print("  Q = quit\n")

    while True:
        ok, frame = cap.read()
        if not ok:
            continue

        display = frame.copy()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Try to find checkerboard corners for live feedback
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

        cv2.imshow("Camera Test", display)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break

        elif key == ord('c') and found_board:
            # Refine corners to sub-pixel accuracy
            criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            obj_points.append(objp)
            img_points.append(corners2)
            print(f"  Captured frame {len(img_points)}  ({len(img_points)} total)")

        elif key == ord('k'):
            if len(img_points) < 5:
                print(f"  Need at least 5 captures (have {len(img_points)}). Keep going!")
            else:
                calibrate(obj_points, img_points, (FRAME_W, FRAME_H))

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
