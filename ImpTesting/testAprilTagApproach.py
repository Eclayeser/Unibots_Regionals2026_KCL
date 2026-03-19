#!/usr/bin/env python3
"""
test_apriltag_approach.py  –  Standalone AprilTag approach & lock test
=======================================================================
Target platform: Raspberry Pi 5

What this script does
---------------------
1. Opens the camera.
2. Spins the robot left to search for an AprilTag.
3. Once a target tag is visible, steers toward it using apf_move()
   (APF steering).
4. When the tag is within CLOSE_RANGE_CM (35 cm) the robot performs a
   ONE-SHOT alignment:
       a. Rotate in place by the current yaw error  (rotate_to_align)
       b. Strafe sideways  by the current x-offset  (strafe_to_align)
   After this single correction the robot is marked 'manually_locked'
   regardless of whether the tag is still visible.
5. Executes slow_wall_approach() and stops.

Lock behaviour
--------------
There are two independent paths to 'locked':

  A. FAR-RANGE fallback  (APF phase, distance > 35 cm)
     AprilTagNavigator._is_locked() returns True when:
       • abs(yaw_deg)         ≤ LOCK_YAW_DEG  (≤ 10°)
       • abs(centre_error_px) ≤ LOCK_CENTER_TOL_PX (≤ 7 px)
     (Distance is NOT a condition.)

  B. CLOSE-RANGE one-shot  (primary path, distance ≤ CLOSE_RANGE_CM)
     The script reads yaw and x_cm ONCE from the current detection,
     applies open-loop corrections via rotate_to_align() and
     strafe_to_align(), then sets `manually_locked = True`.
     No further tag detection is needed; the robot is considered aligned.

Why one-shot instead of continuous close-range looping?
-------------------------------------------------------
The old close_range_align() was called every loop tick (~25 Hz).
Each tick computed a new correction and immediately executed it,
often oscillating: small overshoot → correction → opposite overshoot.
One-shot: read the error once, apply it open-loop, done.  More
predictable on a physical robot; avoids feedback oscillation at close
range where camera latency and motor lag are significant.

Tuning
------
  LOCK_YAW_DEG       : yaw tolerance for far-range lock (degrees).
  LOCK_CENTER_TOL_PX : pixel-centre tolerance for far-range lock.
  CLOSE_RANGE_CM     : distance below which one-shot alignment fires.
  MC.TIME_PER_DEGREE : calibrate so pivot_*_degrees() is accurate.
  MC.TIME_PER_CM_STRAFE : calibrate so strafe_to_align() is accurate.
  slow_wall_approach() duration/speed: in MotorsControllerIT.
"""

import cv2
import time
import logging

import configTag_IT as configTag
from AprilTagNavigator_IT import AprilTagNavigator
import MotorsControllerIT as MC

# ─────────────────────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────
CAMERA_INDEX = 0

# Far-range lock conditions (passed to AprilTagNavigator)
LOCK_YAW_DEG       = 10.0   # degrees
LOCK_CENTER_TOL_PX = 7      # pixels  ← was 320 (effectively disabled)

# Close-range one-shot alignment threshold
CLOSE_RANGE_CM = 35.0       # below this → one-shot correction + force lock

# Main loop pace
LOOP_SLEEP_S = 0.04         # ~25 Hz


# ─────────────────────────────────────────────────────────────────────────────
# Helper: flush stale frames from the camera driver buffer
# ─────────────────────────────────────────────────────────────────────────────
def _flush_camera(cap: cv2.VideoCapture, n: int = 1) -> None:
    for _ in range(n):
        cap.read()


# ─────────────────────────────────────────────────────────────────────────────
# One-shot close-range alignment
# ─────────────────────────────────────────────────────────────────────────────
def _one_shot_align(yaw_deg: float, x_cm: float) -> None:
    """
    Correct yaw then x-offset in a single open-loop sequence.

    Reads yaw_deg and x_cm ONCE (the values captured when the robot
    first enters CLOSE_RANGE_CM).  Applies the corrections without
    re-checking the camera — no feedback loop, no oscillation.

    Step order matters:
      1. Rotate first so the robot faces the wall head-on.
         This makes the subsequent strafe purely lateral.
      2. Strafe to centre on the tag.

    After this function returns, the caller sets manually_locked = True.
    """
    log.info(
        f"ONE-SHOT ALIGN START — yaw={yaw_deg:+.1f}°  x={x_cm:+.1f} cm"
    )

    # ── Step 1: rotate to zero yaw ────────────────────────────────────────
    if abs(yaw_deg) > MC.ALIGN_YAW_THRESHOLD:
        direction = "right" if yaw_deg > 0 else "left"
        log.info(
            f"  Rotating {direction} by {abs(yaw_deg):.1f}° "
            f"(~{abs(yaw_deg) * MC.TIME_PER_DEGREE:.3f} s)"
        )
        MC.rotate_to_align(yaw_deg)
        time.sleep(0.15)   # brief settle before strafe
    else:
        log.info(
            f"  Yaw {yaw_deg:+.1f}° within threshold "
            f"(±{MC.ALIGN_YAW_THRESHOLD}°) — no rotation needed."
        )

    # ── Step 2: strafe to correct lateral offset ──────────────────────────
    if abs(x_cm) > MC.ALIGN_X_THRESHOLD:
        direction = "right" if x_cm > 0 else "left"
        duration  = abs(x_cm) * MC.TIME_PER_CM_STRAFE
        log.info(
            f"  Strafing {direction} by ~{abs(x_cm):.1f} cm "
            f"(~{duration:.3f} s)"
        )
        MC.strafe_to_align(x_cm)
        time.sleep(0.15)   # brief settle before locked state
    else:
        log.info(
            f"  x-offset {x_cm:+.1f} cm within threshold "
            f"(±{MC.ALIGN_X_THRESHOLD} cm) — no strafe needed."
        )

    log.info("ONE-SHOT ALIGN DONE → manually_locked = True")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main() -> None:
    log.info("=" * 60)
    log.info("AprilTag approach test starting.")
    log.info(
        f"Far-range lock: yaw ≤ {LOCK_YAW_DEG}°, "
        f"centre ≤ {LOCK_CENTER_TOL_PX} px"
    )
    log.info(
        f"Close-range one-shot alignment fires at ≤ {CLOSE_RANGE_CM} cm."
    )
    log.info("=" * 60)

    navigator = AprilTagNavigator(
        target_tag_ids           = configTag.APRILTAG_TARGET_IDS,
        camera_params            = configTag.APRILTAG_CAMERA_PARAMS,
        tag_size_m               = configTag.APRILTAG_TAG_SIZE_M,
        frame_size               = configTag.APRILTAG_FRAME_SIZE,
        lock_distance_cm         = 20.0,               # unused by _is_locked; kept for API compat
        lock_center_tolerance_px = LOCK_CENTER_TOL_PX,
        lock_yaw_deg             = LOCK_YAW_DEG,
        tag_families             = configTag.APRILTAG_FAMILY,
    )

    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  configTag.APRILTAG_FRAME_SIZE[0])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, configTag.APRILTAG_FRAME_SIZE[1])
    cap.set(cv2.CAP_PROP_BUFFERSIZE,   1)

    if not cap.isOpened():
        log.error(f"Cannot open camera index {CAMERA_INDEX}. Exiting.")
        return

    log.info("Camera opened. Starting search loop. Press Ctrl-C to abort.")

    # ── State ────────────────────────────────────────────────────────────────
    # manually_locked is set to True after the one-shot alignment sequence.
    # Once True, the robot skips all detection logic and goes straight to
    # slow_wall_approach(), even if the tag is no longer visible.
    manually_locked = False

    try:
        while True:

            # ── If one-shot alignment already done: final approach ─────────
            if manually_locked:
                log.info("manually_locked = True → executing slow_wall_approach().")
                MC.stop_robot()
                time.sleep(1)
                MC.slow_wall_approach()
                log.info("Approach complete – robot stopped at wall.")
                break

            # ── Grab freshest frame ───────────────────────────────────────
            _flush_camera(cap, n=1)
            ret, frame = cap.read()

            if not ret or frame is None:
                log.warning("cap.read() failed – skipping frame.")
                time.sleep(LOOP_SLEEP_S)
                continue

            # ── AprilTag detection ────────────────────────────────────────
            try:
                result = navigator.process_frame(frame)
            except Exception as exc:
                log.warning(f"AprilTagNavigator raised: {exc}")
                time.sleep(LOOP_SLEEP_S)
                continue

            # ── FAR-RANGE LOCK (fallback path) ────────────────────────────
            # Fires only if the robot happens to satisfy yaw + centre
            # conditions before reaching CLOSE_RANGE_CM.
            if result["lockedTag"]:
                tag = result["chosen_tag"]
                log.info(
                    f"FAR-RANGE LOCK ✓  "
                    f"id={tag['tag_id']}  "
                    f"dist={tag['distance_cm']:.1f} cm  "
                    f"yaw={tag['yaw_deg']:.1f}°"
                )
                MC.stop_robot()
                time.sleep(1.5)
                log.info("Executing slow_wall_approach()...")
                MC.slow_wall_approach()
                log.info("Approach complete – robot stopped at wall.")
                break

            # ── TAG VISIBLE ───────────────────────────────────────────────
            elif result["found"]:
                tag = result["chosen_tag"]
                nav = result["moveTargetTag"]

                if tag["distance_cm"] <= CLOSE_RANGE_CM:
                    # ── ONE-SHOT ALIGNMENT ────────────────────────────────
                    # Capture yaw and x_cm once, apply open-loop corrections,
                    # then force locked state.  The camera is NOT consulted
                    # again during or after the correction sequence.
                    log.info(
                        f"Entered close range "
                        f"(dist={tag['distance_cm']:.1f} cm ≤ {CLOSE_RANGE_CM} cm) "
                        f"yaw={tag['yaw_deg']:+.1f}°  "
                        f"x_raw={tag['x_cm']:+.1f} cm  "
                        f"lateral={tag['lateral_cm']:+.1f} cm  "
                        f"→ one-shot alignment."
                    )
                    MC.stop_robot()
                    time.sleep(0.3)   # let the robot settle before reading pose

                    # Re-read the pose one more time after settling
                    _flush_camera(cap, n=2)
                    ret2, frame2 = cap.read()
                    if ret2 and frame2 is not None:
                        try:
                            result2 = navigator.process_frame(frame2)
                            if result2["found"]:
                                tag = result2["chosen_tag"]   # fresher reading
                        except Exception:
                            pass  # fall back to the tag we already have

                    _one_shot_align(yaw_deg=tag["yaw_deg"], x_cm=tag["lateral_cm"])
                    manually_locked = True
                    # Loop back to the top; the `if manually_locked` branch
                    # will handle the final wall approach on the next tick.

                else:
                    # ── FAR RANGE: APF arc-steering ───────────────────────
                    log.info(
                        f"TRACK  id={tag['tag_id']}  "
                        f"dist={tag['distance_cm']:.1f} cm  "
                        f"angle={nav['angle']:+.1f}°  "
                        f"mag={nav['magnitude']:.2f}"
                    )
                    MC.apf_move(
                        angle_deg = nav["angle"],
                        magnitude = nav["magnitude"],
                    )

            # ── NO TAG: pivot to search ───────────────────────────────────
            else:
                log.info("SEARCH – no tag visible, pivoting left.")
                MC.pivot_left(20)

            time.sleep(LOOP_SLEEP_S)

    except KeyboardInterrupt:
        log.info("Interrupted by user (Ctrl-C).")

    finally:
        MC.stop_robot()
        cap.release()
        log.info("Motors stopped. Camera released.")
        log.info("=" * 60)
        log.info("Test finished.")
        log.info("=" * 60)


if __name__ == "__main__":
    main()