#!/usr/bin/env python3


"""
main_integrated.py  –  Robot Ball Collection Algorithm  (integrated CV)
Target platform: Raspberry Pi 5

Derived from TODO/mainToChange.py.
Changes vs the original:
  1. ball_detector_process – fully implemented with Search / Track / Capture modes.
  2. image_capture_thread  – fully implemented with cv2.VideoCapture.

All other processes and threads are unchanged.

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
    "finilisingState":    False,    # set True after 145 s; triggers unload+park

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
# PROCESS  –  Main Timer // FINISHED IMPLEMENTING THIS ONE
# ══════════════════════════════════════════════════════════════════════════════
def main_timer_process(shared, timer_pause_event, stop_event):
    """
    Tracks elapsed algorithm time.
    After 145 s of active runtime it sets finilisingState = True so
    the robot begins its end-of-game unload + park sequence.
    Also handles the initialisation handshake (initialAlgStart flag).
    """
    log.info("MainTimer started.")

    while not stop_event.is_set():
        timer_pause_event.wait()          # block if the timer itself is paused

        if shared["algHasBeenStarted"]:
            # ── Check how much time has passed ────────────────────────────
            elapsed = time.time() - shared["timeStarted"]
            if elapsed > 145 and not shared["finilisingState"]:
                log.info("145 s elapsed – entering finalising state.")
                shared["finilisingState"] = True
        else:
            # Guard against re-starting the timer after stop_event is set,
            # preventing a late-arriving initialAlgStart from reviving the
            # algorithm during shutdown.
            if shared["initialAlgStart"] and not stop_event.is_set():
                # ── First run: start the timer ────────────────────────────
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
    loop_count = 0
    last_state = None
    last_state_log_loop = -10_000
    state_log_heartbeat = 45

    def _log_state(state_name, detail=""):
        nonlocal last_state, last_state_log_loop
        should_log = (state_name != last_state) or ((loop_count - last_state_log_loop) >= state_log_heartbeat)
        if should_log:
            suffix = f" | {detail}" if detail else ""
            log.info(f"CV state: {state_name}{suffix}")
            last_state = state_name
            last_state_log_loop = loop_count

    while not stop_event.is_set():
        worker_pause_event.wait()
        loop_count += 1

        # Safety gate: never emit motion intent before the algorithm is started.
        if not shared["algHasBeenStarted"]:
            shared["lockedBall"] = ""
            shared["moveTargetBall"] = None
            _log_state("IDLE_WAIT_START")
            time.sleep(0.05)
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
            _log_state("SEARCHING")
            time.sleep(0.02)
            continue

        # Highest-priority ball (sorted by type priority, then distance, then |angle|)
        target = detected_balls[0]

        # ── CAPTURE mode ──────────────────────────────────────────────────
        ball_bottom_y = target["y"] + target["radius"]
        if ball_bottom_y >= frame_height * CAPTURE_Y_THRESHOLD:
            if shared["clawBusy"]:
                shared["lockedBall"] = ""
                obstacles = detect_obstacles(detection["frame"], detection["ball_mask"])
                nav = compute_navigation_vector(
                    target, obstacles, detection["frame"].shape, detection["calibration"]
                )
                shared["moveTargetBall"] = nav
                _log_state("WAITING FOR CLAW", f"target={target['type']} angle={target['angle']:.1f}deg")
                time.sleep(0.02)
                continue

            shared["lockedBall"]     = target["type"]   # "PingPong" or "steel"
            shared["moveTargetBall"] = None
            _log_state(f"CAPTURING {target['type']}")
            time.sleep(0.02)
            continue

        # ── TRACK mode ────────────────────────────────────────────────────
        shared["lockedBall"] = ""

        obstacles = detect_obstacles(detection["frame"], detection["ball_mask"])
        nav = compute_navigation_vector(
            target, obstacles, detection["frame"].shape, detection["calibration"]
        )
        # nav is None only when target is None – guaranteed non-None here.
        shared["moveTargetBall"] = nav
        if nav is not None:
            _log_state(
                f"TRACKING {target['type']}",
                f"angle={nav['angle']:.1f}deg mag={nav['magnitude']:.2f}",
            )
        else:
            _log_state(f"TRACKING {target['type']}", "nav=none")

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

        frame = None
        try:
            frame = frame_queue.get_nowait()
        except Exception:
            pass

        if frame is None:
            time.sleep(0.02)
            continue

        # TODO: Check latest_frame:
        #       – Run AprilTag detection (e.g. using the `apriltag` or
        #         `dt-apriltags` library) on `frame`.
        #       – Return a list of detected tags, each as a dict:
        #         {"tag_id": int, "our_side": bool, "centre_x": int,
        #          "centre_y": int, "distance": float}
        detected_tags = []   # placeholder  ← replace with real detection

        if not detected_tags:
            shared["moveTargetTag"] = None

        else:
            # ── Tags detected → check if our tag is directly in front ─────
            # TODO: Determine whether an our-side AprilTag is:
            #         (a) centred horizontally in the frame (centre_x ≈ frame_width/2), AND
            #         (b) within docking distance (distance < DOCK_THRESHOLD).
            #       Set tag_right_in_front = True / False accordingly.
            tag_right_in_front = False   # placeholder

            if tag_right_in_front:
                shared["lockedTag"] = True

            else:
                shared["lockedTag"] = False

                # TODO: Detect potential obstacles in the frame:
                #       walls and other robots (any object that is not white and
                #       not a ball).  Return a list of obstacle positions.
                obstacles = []   # placeholder

                # TODO: Update moveTargetTag using one of:
                #   (A) Our-side tag IS visible → set it as the navigation target;
                #       use its centre as the attraction point with obstacle
                #       repulsion from `obstacles`.
                #   (B) Only opponent tags are visible → rotate in place to search
                #       for our tag, OR use the known arena geometry to mathematically
                #       estimate our tag's position and navigate there.
                #   – angle: 0° = straight ahead, positive = right,
                #     negative = left.  magnitude: normalised 0.0–1.0.
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
# THREAD  –  Motors Handler // FINISHED IMPLEMENTING THIS ONE
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
                # Retract the claw before docking to prevent wall collision.
                # clawAdjusted=False triggers ServoHandler to retract on its
                # next loop tick. slow_wall_approach takes ~1 s, giving
                # ServoHandler time to complete retraction first.
                if shared["clawAdjusted"]:
                    shared["clawAdjusted"] = False

                MC.slow_wall_approach()  # approach the AprilTag target slowly and stop

                # Mark clawBusy before triggering engageHandle so that no other
                # path can set engageClaw while the unload is in progress.
                shared["clawBusy"]     = True
                shared["engageHandle"] = True
                time.sleep(3)   # wait for ServoHandler to complete unload

                # No reverse needed – end of match.

                # Full shutdown – algorithm is done.
                shared["algHasBeenStarted"] = False
                shared["initialAlgStart"]   = False
                stop_event.set()

            else:
                if shared["moveTargetTag"] is not None:
                    # angle: 0° ahead, positive = right, negative = left
                    MC.apf_move(angle_deg=shared["moveTargetTag"]["angle"], magnitude=shared["moveTargetTag"]["magnitude"])
                else:
                    # ── No AprilTag detected → rotate in place to search ──
                    MC.pivot_left()

        else:
            if shared["storageFull"]:
                # ── Storage full: navigate to unload zone ─────────────────
                if shared["lockedTag"]:
                    # Retract the claw before docking, symmetrically
                    # with the finilisingState path. Without this the claw
                    # was extended when the robot drove into the wall.
                    if shared["clawAdjusted"]:
                        shared["clawAdjusted"] = False

                    MC.slow_wall_approach()  # approach the AprilTag target slowly and stop

                    # Mark clawBusy before triggering engageHandle so
                    # that no other path can set engageClaw while unloading.
                    shared["clawBusy"]     = True
                    shared["engageHandle"] = True
                    time.sleep(3)   # wait for ServoHandler to complete unload

                    MC.reverse_from_wall()  # back away from the wall after unloading

                    # clawAdjusted is already False (set above before docking),
                    # so ServoHandler will reopen the claw automatically.

                else:
                    if shared["moveTargetTag"] is not None:
                        # angle: 0° ahead, positive = right, negative = left
                        MC.apf_move(angle_deg=shared["moveTargetTag"]["angle"], magnitude=shared["moveTargetTag"]["magnitude"])
                    else:
                        # ── No AprilTag detected → rotate in place to search ──
                        MC.pivot_left()

            else:
                # ── Normal operation: collect balls ───────────────────────
                # Guard with clawBusy so engageClaw is only set once per ball.
                # Without this guard, MotorsHandler would re-set engageClaw
                # every 20 ms while lockedBall persists, potentially triggering
                # a double-grab before ServoHandler clears lockedBall.
                if shared["lockedBall"] != "" and not shared["clawBusy"]:
                    MC.confident_approach_toGrab()  # approach the locked ball's position
                    shared["engageClaw"] = True
                    shared["clawBusy"]   = True     # lock until ServoHandler clears it

                else:
                    if shared["moveTargetBall"] is not None:
                        # angle: 0° ahead, positive = right, negative = left
                        MC.apf_move(angle_deg=shared["moveTargetBall"]["angle"], magnitude=shared["moveTargetBall"]["magnitude"])
                    else:
                        # ── No target ball detected → rotate in place to search ──
                        MC.pivot_left()

        time.sleep(0.02)

    log.info("MotorsHandler stopped.")


# ══════════════════════════════════════════════════════════════════════════════
# THREAD  –  Servo Handler // FINISHED IMPLEMENTING THIS ONE
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

        # clawAdjusted=False means the claw needs to be moved into position.
        # This covers three cases:
        #   (a) Initial startup  (clawAdjusted starts False, storageFull=False → open)
        #   (b) Storage just became full → hide claw for transit to unload zone
        #   (c) After unload completes  → reopen claw for next collection cycle
        # Also hides the claw when finilisingState is True so it is safely
        # retracted before the robot docks against the wall.
        if not shared["clawAdjusted"]:
            # ── Hide claw for delivery or end-of-game ────────────────────
            if shared["storageFull"] or shared["finilisingState"]:
                SC.clutch_set_angle(200)
                log.info("Claw retracted to 200° for delivery/end-of-game.")
            # ── Open claw for collection ──────────────────────────────────
            else:
                SC.clutch_down()
                log.info("Claw opened for collection.")
            shared["clawAdjusted"] = True

        else:
            # ── Unload mechanism (engageHandle) ───────────────────────────
            if shared["engageHandle"]:
                SC.mg90s_turn_by_180_up()
                time.sleep(2)   # wait for balls to slide out
                SC.mg90s_turn_by_180_down()

                held = list(shared["currentlyHeldBalls"])
                log.info(f"Unloading {held[0]} PingPong + {held[1]} steel balls.")
                shared["currentlyHeldBalls"] = [0, 0]
                shared["engageHandle"]        = False
                shared["storageFull"]         = False
                # Trigger claw to reopen on the next loop iteration.
                # storageFull is now False, so the adjustment block above
                # will call clutch_down() on the next tick.
                shared["clawAdjusted"]        = False
                # Release the grab lock so MotorsHandler can trigger a new
                # grab once the robot resumes ball collection.
                shared["clawBusy"]            = False

            # ── Ball capture claw (engageClaw) ────────────────────────────
            elif shared["engageClaw"]:
                SC.clutch_grabbing_motion()

                # ── Update held balls count ───────────────────────────────
                held = list(shared["currentlyHeldBalls"])
                ball_type = shared["lockedBall"]
                if ball_type == "PingPong":
                    held[0] += 1
                elif ball_type == "steel":
                    held[1] += 1
                shared["currentlyHeldBalls"] = held

                shared["engageClaw"]  = False
                # Release the grab lock so MotorsHandler can react to the
                # next ball once lockedBall is also cleared below.
                shared["clawBusy"]    = False
                log.info(f"Ball captured ({ball_type}). Held: {held}")
                shared["lockedBall"]  = ""

                STORAGE_CAPACITY = 4  # PingPong=1 unit, steel=0.4 units; capacity=4 units
                totalVal = held[0] + 0.4 * held[1]

                if totalVal >= STORAGE_CAPACITY:
                    shared["storageFull"]  = True
                    shared["clawAdjusted"] = False  # triggers claw hide on next loop
                    log.info("Storage full – switching to unload mode.")

        time.sleep(0.02)

    log.info("ServoHandler stopped.")


# ══════════════════════════════════════════════════════════════════════════════
# THREAD  –  Button Handler  (GPIO 7, via gpiozero) // FINISHED IMPLEMENTING THIS ONE
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

        # Pause ALL processes and threads, including the Main Timer.
        worker_pause_event.clear()
        timer_pause_event.clear()

        # Restore every shared flag to its initial value so the robot
        # cannot restart in a broken state (e.g. clawBusy=True blocking all
        # future grabs, storageFull=True sending it straight to the unload
        # zone, or stale engageClaw/engageHandle firing immediately on resume).
        # btnHeld is skipped here and preserved as True so on_released() can
        # correctly identify this event as a long-hold rather than a short press.
        for key, value in SHARED_INITIAL_STATE.items():
            if key == "btnHeld":
                continue
            shared[key] = value

        log.info("Full reset complete – all state restored to initial values.")

    def on_released():
        """Triggered on every button release (short or long)."""
        if shared["btnHeld"]:
            # ── Long-hold release: just clear the flag ───────────────────
            # The robot stays paused; pressing again will start fresh.
            shared["btnHeld"] = False
            log.info("Long-hold released – robot paused and reset. Press again to start.")

        else:
            # ── Short-press release ──────────────────────────────────────
            if shared["algHasBeenStarted"]:
                # Toggle pause for workers ONLY (Main Timer keeps running).
                if worker_pause_event.is_set():
                    # Currently running → pause workers.
                    worker_pause_event.clear()
                    log.info("Workers paused (Main Timer continues).")
                else:
                    # Currently paused → resume workers.
                    worker_pause_event.set()
                    log.info("Workers resumed.")
            else:
                # Algorithm has not been started yet → request start.
                shared["initialAlgStart"] = True
                # Ensure both pause events are set (running) so all threads wake up.
                worker_pause_event.set()
                timer_pause_event.set()
                log.info("Algorithm start requested.")

    button.when_held     = on_held
    button.when_released = on_released
    return button


# ══════════════════════════════════════════════════════════════════════════════
# MAIN FILE
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

    worker_pause_event.clear()   # start paused until first short-press starts algorithm
    timer_pause_event.clear()    # start paused until first short-press starts algorithm

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
