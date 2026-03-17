#!/usr/bin/env python3

"""
main_integrated.py  –  Robot Ball Collection Algorithm  (integrated CV)
Target platform: Raspberry Pi 5

Based on GeneralTesting_1/mainGT1.py with the following changes:
  1. Imports updated to use KrishivBallDetection modules
     (ball_detector_runtime, MotorsController_mecanum, ServoController).
  2. ball_detector_process wired to STREAM_DEBUG_FEED / start_debug_mjpeg_server.
  3. BUTTON_GPIO = 7 (matches KrishivBallDetection hardware wiring).

Architecture
------------
Processes : MainTimer | BallDetector | AprilTagDetector
Threads   : MotorsHandler | ServoHandler | ButtonHandler (gpiozero) | ImageFrameCapture

Pause convention
----------------
  worker_pause_event   (set = RUNNING)  –  waited on by all workers EXCEPT MainTimer
  timer_pause_event    (set = RUNNING)  –  waited on by MainTimer only
  stop_event           (set = STOP)     –  terminates every loop when set

Button behaviour
----------------
  Hold > 5 s : full reset  – clears BOTH pause events, resets alg state
  Short press : if alg running  → toggle worker_pause_event only (MainTimer keeps going)
                if alg not yet started → sets initialAlgStart = True

Shared dict conventions
-----------------------
  moveTargetBall / moveTargetTag : {"angle": float, "magnitude": float}
    angle     – degrees relative to the robot's current heading.
                0° = straight ahead, positive = right, negative = left.
                Range: [-180, +180].  Must match MotorsController.apf_move().
    magnitude – normalised APF force strength in [0.0, 1.0].

  clawAdjusted : False → ServoHandler must reposition the claw before anything else.
                 True  → claw is in position; engageClaw / engageHandle may be used.

  clawBusy     : True  → a grab or unload is in progress; MotorsHandler must not
                          set engageClaw or engageHandle until this clears.
                 False → ready for the next action.
"""

import multiprocessing
import sys
import threading
import time
import logging
from pathlib import Path

import cv2
from gpiozero import Button

# ── Path setup: allow importing from FinalVersion/ and KrishivBallDetection/ ──
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "FinalVersion"))
sys.path.insert(0, str(_ROOT / "KrishivBallDetection"))

import ServoController as SC
import MotorsController_mecanum as MC
from ball_detector_runtime import (
    detect_balls,
    detect_obstacles,
    compute_navigation_vector,
    start_debug_mjpeg_server,
    STREAM_DEBUG_FEED,
)

# ─────────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s %(processName)s/%(threadName)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

BUTTON_GPIO = 7           # GPIO pin for the physical button
CAMERA_INDEX = 0          # cv2.VideoCapture device index (0 = first camera)
CAPTURE_Y_THRESHOLD = 0.88  # ball bottom pixel at this fraction of frame height → Capture mode

# Initial values for all shared state – used at startup and on full reset.
# Centralising them here ensures on_held() and main() are always in sync.
SHARED_INITIAL_STATE = {
    # ── Algorithm control ──────────────────────────────────────────────────
    "initialAlgStart":    False,    # set True by ButtonHandler to kick off timer
    "algHasBeenStarted":  False,    # set True by MainTimer once running
    "timeStarted":        0.0,      # epoch timestamp of algorithm start
    "finilisingState":    False,    # set True after 160 s; triggers unload+park

    # ── Ball detection / navigation ────────────────────────────────────────
    # angle: degrees from robot heading, positive = right. magnitude: 0.0–1.0.
    "lockedBall":         "",       # "" | "PingPong" | "steel"
    "moveTargetBall":     None,     # dict: {angle, magnitude}  (see convention above)

    # ── AprilTag detection / navigation ───────────────────────────────────
    # angle: degrees from robot heading, positive = right. magnitude: 0.0–1.0.
    "lockedTag":          False,    # True when our unload tag is directly ahead
    "moveTargetTag":      None,     # dict: {angle, magnitude}  (see convention above)

    # ── Claw / servo control ───────────────────────────────────────────────
    "engageClaw":         False,    # True → ServoHandler should grab a ball
    "engageHandle":       False,    # True → ServoHandler should unload balls
    "clawAdjusted":       False,    # False → ServoHandler must reposition the claw
                                    # True  → claw is in position; ready for actions
    "clawBusy":           False,    # True → grab/unload in progress; block re-trigger
    "currentlyHeldBalls": [0, 0],   # [pingpong_count, steel_count]
    "storageFull":        False,    # True → navigate to unload zone

    # ── Button ────────────────────────────────────────────────────────────
    "btnHeld":            False,    # True if button was held for >= 5 s
}


# ══════════════════════════════════════════════════════════════════════════════
# PROCESS  –  Main Timer
# ══════════════════════════════════════════════════════════════════════════════
def main_timer_process(shared, timer_pause_event, stop_event):
    """
    Tracks elapsed algorithm time.
    After 160 s of active runtime it sets finilisingState = True so
    the robot begins its end-of-game unload + park sequence.
    Also handles the initialisation handshake (initialAlgStart flag).
    """
    log.info("MainTimer started.")

    while not stop_event.is_set():
        timer_pause_event.wait()          # block if the timer itself is paused

        if shared["algHasBeenStarted"]:
            elapsed = time.time() - shared["timeStarted"]
            if elapsed > 160 and not shared["finilisingState"]:
                log.info("160 s elapsed – entering finalising state.")
                shared["finilisingState"] = True
        else:
            if shared["initialAlgStart"] and not stop_event.is_set():
                shared["initialAlgStart"]   = False
                shared["algHasBeenStarted"] = True
                shared["timeStarted"]       = time.time()
                log.info("Algorithm timer started.")

        time.sleep(0.02)

    log.info("MainTimer stopped.")


# ══════════════════════════════════════════════════════════════════════════════
# PROCESS  –  Ball Detector
# ══════════════════════════════════════════════════════════════════════════════
def ball_detector_process(frame_queue, shared, worker_pause_event, stop_event):
    """
    Reads the latest camera frame, detects balls, classifies them, and
    updates shared["lockedBall"] and shared["moveTargetBall"].

    Debug stream
    ------------
    When STREAM_DEBUG_FEED is True the MJPEG server is started once at
    process start (idempotent) and annotated frames are automatically
    pushed inside ball_detector_runtime.detect_balls().
    Open http://<pi-ip>:5000 on a connected laptop to view the feed.

    Three operating modes
    ---------------------
    SEARCH  – no balls visible:
        lockedBall=""  moveTargetBall=None  → MotorsHandler will pivot_left to scan.

    CAPTURE – highest-priority ball's bottom pixel (y + radius) has reached
              CAPTURE_Y_THRESHOLD of the frame height, meaning the ball is at the
              robot's intake zone:
        lockedBall="PingPong"|"steel"  moveTargetBall=None
        → MotorsHandler calls confident_approach_toGrab() then engages claw.

    TRACK   – ball visible but not yet in the intake zone:
        lockedBall=""  moveTargetBall={angle, magnitude}
        → MotorsHandler calls apf_move() to navigate toward the ball.

    moveTargetBall angle convention: 0° = straight ahead, positive = right,
    negative = left.  Range [-180, +180].  Must match apf_move() expectation.
    """
    log.info("BallDetector started.")

    # Start the MJPEG debug server once for the lifetime of this process.
    # The server runs as a daemon thread inside this process so it shares
    # the _debug_frame_queue with ball_detector_runtime without any IPC.
    if STREAM_DEBUG_FEED:
        start_debug_mjpeg_server()
        log.info(f"Debug stream active – connect at http://<pi-ip>:5000")

    while not stop_event.is_set():
        worker_pause_event.wait()

        # Run this process only while actively collecting balls.
        if shared["storageFull"] and shared["finilisingState"]:
            shared["moveTargetBall"] = None
            time.sleep(0.02)
            continue

        # ── Grab the latest frame (non-blocking) ──────────────────────────
        frame = None
        try:
            frame = frame_queue.get_nowait()
        except Exception:
            pass

        if frame is None:
            time.sleep(0.02)
            continue

        # ── Run calibrated ball detection pipeline ────────────────────────
        # detect_balls() automatically annotates and pushes to the debug
        # stream when STREAM_DEBUG_FEED is True.
        try:
            detection = detect_balls(frame)
        except Exception as exc:
            log.warning(f"detect_balls raised: {exc}")
            time.sleep(0.02)
            continue

        detected_balls = detection["detected_balls"]
        frame_height = detection["frame"].shape[0]

        # ── SEARCH mode ───────────────────────────────────────────────────
        if not detected_balls:
            shared["lockedBall"]     = ""
            shared["moveTargetBall"] = None
            time.sleep(0.02)
            continue

        # Highest-priority ball (sorted by type priority, then distance, then |angle|)
        target = detected_balls[0]

        # ── CAPTURE mode ──────────────────────────────────────────────────
        ball_bottom_y = target["y"] + target["radius"]
        if ball_bottom_y >= frame_height * CAPTURE_Y_THRESHOLD:
            shared["lockedBall"]     = target["type"]   # "PingPong" or "steel"
            shared["moveTargetBall"] = None
            time.sleep(0.02)
            continue

        # ── TRACK mode ────────────────────────────────────────────────────
        shared["lockedBall"] = ""

        obstacles = detect_obstacles(detection["frame"], detection["ball_mask"])
        nav = compute_navigation_vector(
            target, obstacles, detection["frame"].shape, detection["calibration"]
        )
        shared["moveTargetBall"] = nav

        time.sleep(0.02)

    log.info("BallDetector stopped.")


# ══════════════════════════════════════════════════════════════════════════════
# PROCESS  –  AprilTag Detector
# ══════════════════════════════════════════════════════════════════════════════
def apriltag_detector_process(frame_queue, shared, worker_pause_event, stop_event):
    """
    Reads the latest camera frame, detects AprilTags, and updates
    shared["lockedTag"] and shared["moveTargetTag"].

    moveTargetTag angle convention: 0° = straight ahead, positive = right,
    negative = left.  Range [-180, +180].  Must match apf_move() expectation.
    """
    log.info("AprilTagDetector started.")

    while not stop_event.is_set():
        worker_pause_event.wait()

        # Run this process only when navigating to the unload zone.
        if not shared["storageFull"] and not shared["finilisingState"]:
            shared["moveTargetTag"] = None
            time.sleep(0.02)
            continue

        frame = None
        try:
            frame = frame_queue.get_nowait()
        except Exception:
            pass

        if frame is None:
            time.sleep(0.02)
            continue

        # TODO: Run AprilTag detection on `frame` (e.g. with `apriltag` or
        #       `dt-apriltags`).  Return a list of dicts:
        #         {"tag_id": int, "our_side": bool, "centre_x": int,
        #          "centre_y": int, "distance": float}
        detected_tags = []   # placeholder ← replace with real detection

        if not detected_tags:
            shared["moveTargetTag"] = None

        else:
            # TODO: Determine whether an our-side AprilTag is:
            #         (a) centred horizontally in the frame, AND
            #         (b) within docking distance.
            #       Set tag_right_in_front = True / False.
            tag_right_in_front = False   # placeholder

            if tag_right_in_front:
                shared["lockedTag"] = True

            else:
                shared["lockedTag"] = False

                # TODO: Build obstacle list and compute navigation vector.
                obstacles = []   # placeholder
                shared["moveTargetTag"] = {"angle": 0, "magnitude": 0}

        time.sleep(0.02)

    log.info("AprilTagDetector stopped.")


# ══════════════════════════════════════════════════════════════════════════════
# THREAD  –  Image Frame Capture
# ══════════════════════════════════════════════════════════════════════════════
def image_capture_thread(ball_frame_q, tag_frame_q, worker_pause_event, stop_event):
    """
    Continuously captures camera frames and distributes the latest one to
    both detector processes via their respective single-slot queues.
    Stale frames are discarded so detectors always see the most recent image.
    """
    log.info("ImageFrameCapture started.")

    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)     # keep only the freshest frame in driver buffer

    if not cap.isOpened():
        log.error(f"ImageFrameCapture: could not open camera index {CAMERA_INDEX}.")
        return

    try:
        while not stop_event.is_set():
            worker_pause_event.wait()

            ret, new_frame = cap.read()
            if not ret or new_frame is None:
                log.warning("ImageFrameCapture: cap.read() failed – skipping frame.")
                time.sleep(0.02)
                continue

            # Push to BallDetector queue  (drop stale frame first)
            if not ball_frame_q.empty():
                try:
                    ball_frame_q.get_nowait()
                except Exception:
                    pass
            ball_frame_q.put(new_frame)

            # Push to AprilTagDetector queue  (drop stale frame first)
            if not tag_frame_q.empty():
                try:
                    tag_frame_q.get_nowait()
                except Exception:
                    pass
            tag_frame_q.put(new_frame)

            time.sleep(0.02)
    finally:
        cap.release()
        log.info("ImageFrameCapture: camera released.")

    log.info("ImageFrameCapture stopped.")


# ══════════════════════════════════════════════════════════════════════════════
# THREAD  –  Motors Handler
# ══════════════════════════════════════════════════════════════════════════════
def motors_handler_thread(shared, worker_pause_event, stop_event):
    """
    Reads moveTargetBall / moveTargetTag and the lockedBall / lockedTag flags
    to decide how to drive the robot's wheels.

    Priority:
      finilisingState=True  → navigate to AprilTag and unload + park
      storageFull=True      → navigate to AprilTag, unload, then resume
      otherwise             → navigate to / capture balls

    APF angle convention: 0° = straight ahead, positive = right, negative = left.
    apf_move() in MotorsController interprets positive angle as steer-right.
    """
    log.info("MotorsHandler started.")

    while not stop_event.is_set():
        worker_pause_event.wait()

        if shared["finilisingState"]:
            # ── End-of-game: unload and park ─────────────────────────────
            if shared["lockedTag"]:
                if shared["clawAdjusted"]:
                    shared["clawAdjusted"] = False

                MC.slow_wall_approach()

                shared["clawBusy"]     = True
                shared["engageHandle"] = True
                time.sleep(3)

                shared["algHasBeenStarted"] = False
                shared["initialAlgStart"]   = False
                stop_event.set()

            else:
                if shared["moveTargetTag"] is not None:
                    MC.apf_move(angle_deg=shared["moveTargetTag"]["angle"], magnitude=shared["moveTargetTag"]["magnitude"])
                else:
                    MC.pivot_left()

        else:
            if shared["storageFull"]:
                # ── Storage full: navigate to unload zone ─────────────────
                if shared["lockedTag"]:
                    if shared["clawAdjusted"]:
                        shared["clawAdjusted"] = False

                    MC.slow_wall_approach()

                    shared["clawBusy"]     = True
                    shared["engageHandle"] = True
                    time.sleep(3)

                    MC.reverse_from_wall()

                else:
                    if shared["moveTargetTag"] is not None:
                        MC.apf_move(angle_deg=shared["moveTargetTag"]["angle"], magnitude=shared["moveTargetTag"]["magnitude"])
                    else:
                        MC.pivot_left()

            else:
                # ── Normal operation: collect balls ───────────────────────
                if shared["lockedBall"] != "" and not shared["clawBusy"]:
                    MC.confident_approach_toGrab()
                    shared["engageClaw"] = True
                    shared["clawBusy"]   = True

                else:
                    if shared["moveTargetBall"] is not None:
                        MC.apf_move(angle_deg=shared["moveTargetBall"]["angle"], magnitude=shared["moveTargetBall"]["magnitude"])
                    else:
                        MC.pivot_left()

        time.sleep(0.02)

    log.info("MotorsHandler stopped.")


# ══════════════════════════════════════════════════════════════════════════════
# THREAD  –  Servo Handler
# ══════════════════════════════════════════════════════════════════════════════
def servo_handler_thread(shared, worker_pause_event, stop_event):
    """
    Controls:
      – Claw servo (ball capture): triggered by engageClaw flag
      – Handle / unload mechanism: triggered by engageHandle flag
    Also performs the one-time initial claw adjustment on startup.

    clawAdjusted convention
    -----------------------
      False → claw needs repositioning; this thread moves it then sets True.
      True  → claw is in position; this thread processes engageClaw/engageHandle.

    clawBusy convention
    -------------------
      True  → a grab or unload is in progress; MotorsHandler must not re-trigger.
      False → ready for the next action.
    """
    log.info("ServoHandler started.")

    while not stop_event.is_set():
        worker_pause_event.wait()

        if not shared["clawAdjusted"]:
            if shared["storageFull"] or shared["finilisingState"]:
                SC.clutch_set_angle(200)
                log.info("Claw retracted to 200° for delivery/end-of-game.")
            else:
                SC.clutch_down()
                log.info("Claw opened for collection.")
            shared["clawAdjusted"] = True

        else:
            # ── Unload mechanism (engageHandle) ───────────────────────────
            if shared["engageHandle"]:
                SC.mg90s_turn_by_180_up()
                time.sleep(2)
                SC.mg90s_turn_by_180_down()

                held = list(shared["currentlyHeldBalls"])
                log.info(f"Unloading {held[0]} PingPong + {held[1]} steel balls.")
                shared["currentlyHeldBalls"] = [0, 0]
                shared["engageHandle"]        = False
                shared["storageFull"]         = False
                shared["clawAdjusted"]        = False
                shared["clawBusy"]            = False

            # ── Ball capture claw (engageClaw) ────────────────────────────
            elif shared["engageClaw"]:
                SC.clutch_grabbing_motion()

                held = list(shared["currentlyHeldBalls"])
                ball_type = shared["lockedBall"]
                if ball_type == "PingPong":
                    held[0] += 1
                elif ball_type == "steel":
                    held[1] += 1
                shared["currentlyHeldBalls"] = held

                shared["engageClaw"]  = False
                shared["clawBusy"]    = False
                log.info(f"Ball captured ({ball_type}). Held: {held}")
                shared["lockedBall"]  = ""

                STORAGE_CAPACITY = 4  # PingPong=1 unit, steel=0.4 units; capacity=4 units
                totalVal = held[0] + 0.4 * held[1]

                if totalVal >= STORAGE_CAPACITY:
                    shared["storageFull"]  = True
                    shared["clawAdjusted"] = False
                    log.info("Storage full – switching to unload mode.")

        time.sleep(0.02)

    log.info("ServoHandler stopped.")


# ══════════════════════════════════════════════════════════════════════════════
# THREAD  –  Button Handler  (GPIO 7, via gpiozero)
# ══════════════════════════════════════════════════════════════════════════════
def setup_button_handler(shared, worker_pause_event, timer_pause_event, stop_event):
    """
    Configures gpiozero Button callbacks.
    Returns the Button object – the caller MUST keep a reference alive so
    gpiozero's background thread remains active.

    Button behaviour:
      Hold > 5 s  → full reset (pauses EVERYTHING incl. MainTimer, resets alg)
      Short press → if alg started: toggle pause for workers only (not MainTimer)
                    if alg not started: set initialAlgStart = True to kick off timer
    """
    button = Button(BUTTON_GPIO, hold_time=5)

    def on_held():
        """Triggered when button is held for longer than 5 seconds."""
        log.info("Button held 5 s – performing full reset.")
        shared["btnHeld"] = True

        worker_pause_event.clear()
        timer_pause_event.clear()

        for key, value in SHARED_INITIAL_STATE.items():
            if key == "btnHeld":
                continue
            shared[key] = value

        log.info("Full reset complete – all state restored to initial values.")

    def on_released():
        """Triggered on every button release (short or long)."""
        if shared["btnHeld"]:
            shared["btnHeld"] = False
            log.info("Long-hold released – robot paused and reset. Press again to start.")

        else:
            if shared["algHasBeenStarted"]:
                if worker_pause_event.is_set():
                    worker_pause_event.clear()
                    log.info("Workers paused (Main Timer continues).")
                else:
                    worker_pause_event.set()
                    log.info("Workers resumed.")
            else:
                shared["initialAlgStart"] = True
                worker_pause_event.set()
                timer_pause_event.set()
                log.info("Algorithm start requested.")

    button.when_held     = on_held
    button.when_released = on_released
    return button


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
def main():
    log.info("=== Robot Main File starting ===")

    # ── Create inter-process shared state via Manager ─────────────────────
    manager = multiprocessing.Manager()
    shared = manager.dict(SHARED_INITIAL_STATE)

    # ── Setup Events and Locks ────────────────────────────────────────────
    # Convention: event.set() = RUNNING, event.clear() = PAUSED
    worker_pause_event = multiprocessing.Event()   # workers (not MainTimer)
    timer_pause_event  = multiprocessing.Event()   # MainTimer only
    stop_event         = multiprocessing.Event()   # terminates all loops

    worker_pause_event.set()   # start in running state
    timer_pause_event.set()    # start in running state

    # ── Frame queues  (ImageFrameCapture → Detectors) ─────────────────────
    # maxsize=1 ensures detectors always consume the latest frame.
    ball_frame_queue = multiprocessing.Queue(maxsize=1)
    tag_frame_queue  = multiprocessing.Queue(maxsize=1)

    # ── Define and start processes ────────────────────────────────────────
    processes = [
        multiprocessing.Process(
            target=main_timer_process,
            args=(shared, timer_pause_event, stop_event),
            name="MainTimer",
            daemon=True,
        ),
        multiprocessing.Process(
            target=ball_detector_process,
            args=(ball_frame_queue, shared, worker_pause_event, stop_event),
            name="BallDetector",
            daemon=True,
        ),
        multiprocessing.Process(
            target=apriltag_detector_process,
            args=(tag_frame_queue, shared, worker_pause_event, stop_event),
            name="AprilTagDetector",
            daemon=True,
        ),
    ]
    for p in processes:
        p.start()
        log.info(f"Started process: {p.name}  (PID {p.pid})")

    # ── Define and start threads ──────────────────────────────────────────
    threads = [
        threading.Thread(
            target=motors_handler_thread,
            args=(shared, worker_pause_event, stop_event),
            name="MotorsHandler",
            daemon=True,
        ),
        threading.Thread(
            target=servo_handler_thread,
            args=(shared, worker_pause_event, stop_event),
            name="ServoHandler",
            daemon=True,
        ),
        threading.Thread(
            target=image_capture_thread,
            args=(ball_frame_queue, tag_frame_queue, worker_pause_event, stop_event),
            name="ImageFrameCapture",
            daemon=True,
        ),
    ]
    for t in threads:
        t.start()
        log.info(f"Started thread: {t.name}")

    # ── Button Handler (gpiozero manages its own internal thread) ─────────
    button = setup_button_handler(
        shared, worker_pause_event, timer_pause_event, stop_event
    )
    log.info(f"ButtonHandler active on GPIO {BUTTON_GPIO}.")

    # ── Main monitor loop ─────────────────────────────────────────────────
    try:
        while not stop_event.is_set():
            time.sleep(0.1)

    except KeyboardInterrupt:
        log.info("KeyboardInterrupt – shutting down.")

    except Exception as exc:
        log.error(f"Unexpected error in main loop: {exc}", exc_info=True)

    finally:
        log.info("Stopping all processes and threads...")
        stop_event.set()
        worker_pause_event.set()   # unblock any waiting workers
        timer_pause_event.set()    # unblock MainTimer

        for p in processes:
            p.join(timeout=3)
            if p.is_alive():
                p.terminate()
                log.warning(f"Force-terminated process: {p.name}")

        manager.shutdown()
        log.info("=== Robot Main File stopped ===")


if __name__ == "__main__":
    main()
