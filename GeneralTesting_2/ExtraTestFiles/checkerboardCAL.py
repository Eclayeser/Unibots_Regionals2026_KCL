#!/usr/bin/env python3
"""
calibrate_camera_headless.py
=============================
For Raspberry Pi with no monitor (SSH session).
Automatically captures a snapshot every 3 seconds for 60 seconds (20 shots).
Hold the chessboard at a different angle for each shot.
Results are printed to the terminal and saved to calibration_result.txt
"""

import sys
import time
import cv2
import numpy as np

CAMERA_INDEX     = 0
FRAME_W, FRAME_H = 640, 480
BOARD_COLS       = 9    # inner corners horizontally
BOARD_ROWS       = 6    # inner corners vertically
CAPTURE_INTERVAL = 3.0  # seconds between auto-captures
MAX_SHOTS        = 20

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
objp = np.zeros((BOARD_ROWS * BOARD_COLS, 3), np.float32)
objp[:, :2] = np.mgrid[0:BOARD_COLS, 0:BOARD_ROWS].T.reshape(-1, 2)

objpoints = []
imgpoints = []

cap = cv2.VideoCapture(CAMERA_INDEX)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  FRAME_W)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)

if not cap.isOpened():
    sys.exit("ERROR: Cannot open camera.")

print("=" * 55)
print("HEADLESS CAMERA CALIBRATION")
print("=" * 55)
print(f"Will attempt {MAX_SHOTS} captures every {CAPTURE_INTERVAL}s.")
print("Hold the chessboard at a DIFFERENT angle for each beep.\n")
print("Starting in 5 seconds — get the board ready...")
time.sleep(5)

shot_count  = 0
last_time   = time.time() - CAPTURE_INTERVAL  # capture immediately on first loop

while shot_count < MAX_SHOTS:
    ret, frame = cap.read()
    if not ret:
        time.sleep(0.1)
        continue

    now = time.time()
    if now - last_time < CAPTURE_INTERVAL:
        time.sleep(0.05)
        continue

    last_time = now
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    found, corners = cv2.findChessboardCorners(gray, (BOARD_COLS, BOARD_ROWS), None)

    if found:
        refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        objpoints.append(objp)
        imgpoints.append(refined)
        shot_count += 1
        # Save the captured frame so you can review it later
        cv2.imwrite(f"calib_shot_{shot_count:02d}.jpg", frame)
        print(f"  [{shot_count:02d}/{MAX_SHOTS}] Board found – shot saved. CHANGE ANGLE NOW.")
    else:
        print(f"  [--/{MAX_SHOTS}] No board detected – keep trying. ({CAPTURE_INTERVAL}s)")

cap.release()

if shot_count < 10:
    sys.exit(f"\nOnly {shot_count} valid shots captured – need at least 10. Run again.")

print(f"\nCalibrating from {shot_count} shots...")

ret_val, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
    objpoints, imgpoints, (FRAME_W, FRAME_H), None, None
)
optimal_mtx, _ = cv2.getOptimalNewCameraMatrix(
    mtx, dist, (FRAME_W, FRAME_H), alpha=0.0
)

fx = optimal_mtx[0, 0]
fy = optimal_mtx[1, 1]
cx = optimal_mtx[0, 2]
cy = optimal_mtx[1, 2]

total_error = 0.0
for i in range(len(objpoints)):
    projected, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
    total_error += cv2.norm(imgpoints[i], projected, cv2.NORM_L2) / len(projected)
mean_error = total_error / len(objpoints)

result = f"""
{'=' * 55}
CALIBRATION RESULTS  ({shot_count} shots)
{'=' * 55}
Reprojection error : {mean_error:.4f} px  {'(GOOD)' if mean_error < 1.0 else '(POOR – redo with more angle variety)'}

fx = {fx:.2f}
fy = {fy:.2f}
cx = {cx:.2f}
cy = {cy:.2f}

Paste into configTags_GT2.py:
  APRILTAG_CAMERA_PARAMS = (
      {fx:.2f},   # fx
      {fy:.2f},   # fy
      {cx:.2f},   # cx
      {cy:.2f},   # cy
  )

Paste into ball_detection_krishiv_GT2.py:
  CAMERA_MATRIX = np.array([
      [{fx:.2f}, 0.0, {cx:.2f}],
      [0.0, {fy:.2f}, {cy:.2f}],
      [0.0, 0.0, 1.0]], dtype=np.float64)
{'=' * 55}
"""

print(result)

with open("calibration_result.txt", "w") as f:
    f.write(result)
print("Results also saved to calibration_result.txt")