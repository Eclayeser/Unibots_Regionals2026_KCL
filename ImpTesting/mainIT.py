#!/usr/bin/env python3


"""
main_integrated.py  –  Robot Ball Collection Algorithm  (integrated CV)
Target platform: Raspberry Pi 5

Derived from TODO/mainToChange.py.
Changes vs the original:
  1. ball_detector_process  – fully implemented with Search / Track / Capture modes.
  2. image_capture_thread   – fully implemented with cv2.VideoCapture; frame
                              distribution freezes during atomic sequences.
  3. motors_handler_thread  – restructured into a priority-ordered elif chain;
                              pickingInProcess and dockingInProcess branches now
                              execute as blocking atomic sequences.
  4. servo_handler_thread   – pickingInProcess cleared as the final step of the
                              engageClaw branch so MotorsHandler unblocks only
                              after the claw is fully reset.
  5. apriltag_detector_process – clean TODO stub with full implementation contract.

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
  Short press : if alg running       → pause workers only (MainTimer keeps running)
                if workers paused     → resume workers
                if neither event set  → set both pause events to start the algorithm

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

  pickingInProcess : True → ball has entered capture zone; MotorsHandler is executing
                            the atomic grab sequence; image capture is frozen.
                    False → normal operation.

  dockingInProcess : True → AprilTag has been locked; MotorsHandler is executing
                            the atomic dock/unload/reverse sequence; image capture
                            is frozen.
                    False → normal operation.

Atomic sequence guarantees
--------------------------
  lockedBall  → BallDetector sets pickingInProcess=True and lockedBall=<type>.
                ImageFrameCapture stops distributing frames immediately.
                MotorsHandler executes: confident_approach → stop → engageClaw.
                MotorsHandler then blocks until ServoHandler clears pickingInProcess.
                ImageFrameCapture and detectors resume automatically.

  lockedTag   → AprilTagDetector sets dockingInProcess=True and lockedTag=True.
                ImageFrameCapture stops distributing frames immediately.
                MotorsHandler executes: slow_wall_approach → stop → engageHandle.
                MotorsHandler blocks until ServoHandler clears engageHandle.
                If storageFull: reverse_from_wall, clear dockingInProcess, resume.
                If finilisingState: stop_event.set() – end of match.
"""

import multiprocessing
import sys
import threading
import time
import logging
from pathlib import Path

import cv2
from gpiozero import Button

"""
# ── Path setup: allow importing from FinalVersion/ and KrishivBallDetection/ ──
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "FinalVersion"))
sys.path.insert(0, str(_ROOT / "KrishivBallDetection"))
"""

import ServoControllerIT as SC
import MotorsControllerIT as MC
from ball_detector_runtime_IT import process_ball_frame
from config_IT import (
    CAPTURE_BLOCKED_WAIT_S,
    CAPTURE_SKIP_COOLDOWN_S,
)

# ─────────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s %(processName)s/%(threadName)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

BUTTON_GPIO = 4           # GPIO pin for the physical button
CAMERA_INDEX = 0          # cv2.VideoCapture device index (0 = first camera)

# Initial values for all shared state – used at startup and on full reset.
# Centralising them here ensures on_held() and main() are always in sync.
SHARED_INITIAL_STATE = {
    # ── Algorithm control ──────────────────────────────────────────────────
    "algHasBeenStarted":  False,    # set True by MainTimer once running
    "timeStarted":        0.0,      # epoch timestamp of algorithm start
    "finilisingState":    False,    # set True after 160 s; triggers unload+park
    "startingPosReached": False,    # True once the robot has driven to the starting position

    # ── Ball detection / navigation ────────────────────────────────────────
    # angle: degrees from robot heading, positive = right. magnitude: 0.0–1.0.
    "lockedBall":         "",       # "" | "PingPong" | "steel"
    "moveTargetBall":     None,     # dict: {angle, magnitude}  (see convention above)
    "pickingInProcess":   False,    # True  → atomic grab sequence running; image
                                    #         capture frozen; set by BallDetector,
                                    #         cleared by ServoHandler after grab.

    # ── AprilTag detection / navigation ───────────────────────────────────
    # angle: degrees from robot heading, positive = right. magnitude: 0.0–1.0.
    "lockedTag":          False,    # True when our unload tag is directly ahead
    "moveTargetTag":      None,     # dict: {angle, magnitude}  (see convention above)
    "dockingInProcess":   False,    # True  → atomic dock sequence running; image
                                    #         capture frozen; set by AprilTagDetector,
                                    #         cleared by MotorsHandler after reverse.

    # ── Claw / servo control ───────────────────────────────────────────────
    "engageClaw":         False,    # True → ServoHandler should grab a ball
    "engageHandle":       False,    # True → ServoHandler should unload balls
    "clawAdjusted":       False,    # False → ServoHandler must reposition the claw
                                    # True  → claw is in position; ready for actions
    "clawBusy":           False,    # True → grab/unload in progress; block re-trigger
    "heldPingPong":        0,        # count of ping pong balls held
    "heldSteel":           0,        # count of steel balls held
    "storageFull":        False,    # True → navigate to unload zone

    # ── Button ────────────────────────────────────────────────────────────
    "btnHeld":            False,    # True if button was held for >= 5 s
}


# ══════════════════════════════════════════════════════════════════════════════
# PROCESS  –  Main Timer // FINISHED BY MARK
# ══════════════════════════════════════════════════════════════════════════════
def main_timer_process(shared, timer_pause_event, stop_event):
    """
    Tracks elapsed algorithm time.
    After 160 s of active runtime it sets finilisingState = True so
    the robot begins its end-of-game unload + park sequence.
    On the first iteration after both pause events are set, records
    timeStarted and sets algHasBeenStarted = True.
    """
    log.info("MainTimer started.")

    while not stop_event.is_set():
        timer_pause_event.wait()          # block if the timer itself is paused

        if shared["algHasBeenStarted"]:
            # ── Check how much time has passed ────────────────────────────
            elapsed = time.time() - shared["timeStarted"]
            if elapsed > 160 and not shared["finilisingState"]:
                log.info("160 s elapsed – entering finalising state.")
                shared["finilisingState"] = True
        else:
            # ── First iteration: record start timestamp ───────────────
            shared["algHasBeenStarted"] = True
            shared["timeStarted"]       = time.time()
            log.info("Algorithm timer started.")

        time.sleep(0.02)

    log.info("MainTimer stopped.")


# ══════════════════════════════════════════════════════════════════════════════
# PROCESS  –  Ball Detector // FINISHED BY KRISHIV
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
        lockedBall="PingPong"|"steel"  moveTargetBall=None  pickingInProcess=True
        → MotorsHandler starts atomic grab sequence; image capture freezes.

    TRACK   – ball visible but not yet in the intake zone:
        lockedBall=""  moveTargetBall={angle, magnitude}
        → MotorsHandler calls apf_move() to navigate toward the ball.

    moveTargetBall angle convention: 0° = straight ahead, positive = right,
    negative = left.  Range [-180, +180].  Must match apf_move() expectation.
    """
    log.info("BallDetector started.")

    blocked_since_s = 0.0
    skip_until_s = 0.0
    skip_anchor: tuple[float, float, float] | None = None

    while not stop_event.is_set():
        worker_pause_event.wait()

        now_s = time.monotonic()
        if skip_anchor is not None and now_s >= skip_until_s:
            skip_anchor = None

        # Run only when:  storage not full, not finalising, and no atomic
        # sequence is currently locking the robot.
        if (not shared["storageFull"]
                and not shared["finilisingState"]
                and not shared["pickingInProcess"]
                and not shared["dockingInProcess"]):

            # ── Grab the latest frame (non-blocking) ──────────────────────
            frame = None
            try:
                frame = frame_queue.get_nowait()
            except Exception:
                pass

            if frame is None:
                time.sleep(0.02)
                continue

            try:
                detection = process_ball_frame(
                    frame,
                    ping_pong_profile="orange",
                    skip_anchor=skip_anchor,
                )
            except Exception as exc:
                log.error(f"Vision pipeline crashed: {exc}", exc_info=True)
                time.sleep(0.02)
                continue

            target_ball = detection.get("target_ball")
            nav = detection.get("move_target")
            grab_ready = detection.get("grab_ready", False)
            is_blocked_in_zone = detection.get("is_blocked_in_zone", False)

            if target_ball is None:
                # SEARCHING
                blocked_since_s = 0.0
                shared["lockedBall"] = ""
                shared["moveTargetBall"] = None
            elif grab_ready:
                # CAPTURING
                blocked_since_s = 0.0
                shared["lockedBall"] = target_ball["type"]
                shared["moveTargetBall"] = None
                # Keep this trigger for motors_handler atomic grab sequence.
                shared["pickingInProcess"] = True
            elif is_blocked_in_zone:
                # TIMEOUT LOGIC
                if blocked_since_s == 0.0:
                    blocked_since_s = now_s

                if now_s - blocked_since_s >= CAPTURE_BLOCKED_WAIT_S:
                    skip_until_s = now_s + CAPTURE_SKIP_COOLDOWN_S
                    skip_radius = max(70.0, float(target_ball["radius"]) * 2.5)
                    skip_anchor = (float(target_ball["x"]), float(target_ball["y"]), skip_radius)
                    blocked_since_s = 0.0
                    shared["lockedBall"] = ""
                    shared["moveTargetBall"] = None
                else:
                    shared["lockedBall"] = ""
                    shared["moveTargetBall"] = nav
            else:
                # TRACKING
                blocked_since_s = 0.0
                shared["lockedBall"] = ""
                shared["moveTargetBall"] = nav

        else:
            # Inactive branch.
            # IMPORTANT: do NOT clear lockedBall while pickingInProcess=True –
            # ServoHandler still needs it to identify the ball type for the
            # held-ball count update.
            if not shared["pickingInProcess"]:
                shared["lockedBall"] = ""
            shared["moveTargetBall"] = None
            blocked_since_s = 0.0

        time.sleep(0.02)

    log.info("BallDetector stopped.")


# ══════════════════════════════════════════════════════════════════════════════
# PROCESS  –  AprilTag Detector // TODO (MANISH)
# ══════════════════════════════════════════════════════════════════════════════
def apriltag_detector_process(frame_queue, shared, worker_pause_event, stop_event):
    """
    TODO (Manish): implement AprilTag detection.

    Contract this process must fulfil
    ----------------------------------
    Gate condition – only run detection when ALL of these are true:
        (shared["storageFull"] or shared["finilisingState"])
        and not shared["dockingInProcess"]

    While SEARCHING (tag visible but not yet centred / close enough):
        shared["lockedTag"]    = False
        shared["moveTargetTag"] = {"angle": <degrees>, "magnitude": <0.0–1.0>}
            angle convention: 0° = straight ahead, positive = right, negative = left.

    On LOCK (tag centred and close enough to begin approach):
        shared["lockedTag"]        = True
        shared["moveTargetTag"]    = None       # no steering needed; MotorsHandler
                                                # will call slow_wall_approach()
        shared["dockingInProcess"] = True       # MUST be set last – this is the
                                                # trigger that freezes image capture
                                                # and starts the atomic dock sequence
                                                # in MotorsHandler.

    When NOT active (neither storageFull nor finilisingState):
        leave shared["lockedTag"] and shared["moveTargetTag"] unchanged
        (MotorsHandler does not read them in this state).

    After dockingInProcess is set to True, do NOT modify lockedTag or
    moveTargetTag again until MotorsHandler clears dockingInProcess.

    moveTargetTag angle convention: 0° = straight ahead, positive = right,
    negative = left.  Range [-180, +180].  Must match apf_move() expectation.
    """
    log.info("AprilTagDetector started (stub – awaiting implementation).")

    while not stop_event.is_set():
        worker_pause_event.wait()

        # TODO (Manish): replace the body of this block with real detection.
        if (shared["storageFull"] or shared["finilisingState"]) and not shared["dockingInProcess"]:
            log.info("AprilTagDetector: gate condition met – running detection (stub).")
            pass   # placeholder – real detector goes here

        time.sleep(0.5)

    log.info("AprilTagDetector stopped.")


# ══════════════════════════════════════════════════════════════════════════════
# THREAD  –  Image Frame Capture // FINISHED BY KRISHIV
# ══════════════════════════════════════════════════════════════════════════════
def image_capture_thread(ball_frame_q, tag_frame_q, shared, worker_pause_event, stop_event):
    """
    Continuously captures camera frames and distributes the latest one to
    both detector processes via their respective single-slot queues.
    Stale frames are discarded so detectors always see the most recent image.

    Frame distribution is suspended while pickingInProcess=True or
    dockingInProcess=True so that neither detector can write new targets
    into shared state while an atomic motor/servo sequence is in progress.
    The camera keeps reading (so the driver buffer stays fresh) but frames
    are simply not forwarded until the lock is released.
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

            # ── Freeze distribution during atomic sequences ────────────────
            # The camera continues reading so the driver buffer stays fresh,
            # but we do not forward frames to either detector while a pick or
            # dock sequence is executing.  This ensures no detector can
            # overwrite lockedBall / lockedTag / moveTarget* mid-sequence.
            if shared["pickingInProcess"] or shared["dockingInProcess"]:
                time.sleep(0.02)
                continue

            # ── Push to BallDetector queue  (drop stale frame first) ───────
            if not ball_frame_q.empty():
                try:
                    ball_frame_q.get_nowait()
                except Exception:
                    pass
            ball_frame_q.put(new_frame)

            # ── Push to AprilTagDetector queue  (drop stale frame first) ───
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
# THREAD  –  Motors Handler // FINISHED BY MARK, KRISHIV HELPED WITH MOTORSCONTROLLER
# ══════════════════════════════════════════════════════════════════════════════
def motors_handler_thread(shared, worker_pause_event, stop_event):
    """
    Reads moveTargetBall / moveTargetTag and the pickingInProcess /
    dockingInProcess flags to decide how to drive the robot's wheels.

    Priority (highest → lowest)
    ---------------------------
    1. startingPosReached=False  → drive to starting position (one-shot).
    2. pickingInProcess=True     → ATOMIC: approach → stop → signal claw →
                                   block until ServoHandler clears flag.
    3. dockingInProcess=True     → ATOMIC: approach wall → stop → signal handle →
                                   block until ServoHandler clears engageHandle →
                                   reverse (or stop_event if finilisingState).
    4. finilisingState=True      → navigate toward AprilTag (pivot if not found).
    5. storageFull=True          → navigate toward AprilTag (pivot if not found).
    6. default                   → navigate toward/search for balls.

    APF angle convention: 0° = straight ahead, positive = right, negative = left.
    apf_move() in MotorsController interprets positive angle as steer-right.
    """
    log.info("MotorsHandler started.")

    while not stop_event.is_set():
        worker_pause_event.wait()

        # ── ONE-SHOT: drive to starting position ──────────────────────────
        if not shared["startingPosReached"]:
            MC.move_forward_toStart()
            shared["startingPosReached"] = True

        # ── ATOMIC SEQUENCE: ball pick ─────────────────────────────────────
        # Triggered by BallDetector setting pickingInProcess=True + lockedBall.
        # Image capture has already stopped distributing frames.
        # Execute the full grab sequence here, then block until ServoHandler
        # clears pickingInProcess (confirming the claw motion is complete).
        elif shared["pickingInProcess"]:
            log.info("MotorsHandler: PICK sequence started.")
            MC.confident_approach_toGrab()   # brief final approach to ball
            MC.stop_robot()                  # hold still – robot must not move while claw engages
            shared["engageClaw"] = True
            shared["clawBusy"]   = True
            # Block until ServoHandler finishes the grab and clears pickingInProcess.
            # ServoHandler clears it as the very last step so the robot cannot
            # move again until the claw is fully reset.
            while not stop_event.is_set() and shared["pickingInProcess"]:
                time.sleep(0.02)
            MC.little_reverse()
            log.info("MotorsHandler: PICK sequence complete – resuming normal operation.")

        # ── ATOMIC SEQUENCE: tag dock / unload ────────────────────────────
        # Triggered by AprilTagDetector setting dockingInProcess=True + lockedTag.
        # Image capture has already stopped distributing frames.
        # Execute the full dock sequence here, then block until ServoHandler
        # clears engageHandle (confirming the unload motion is complete).
        elif shared["dockingInProcess"]:
            log.info("MotorsHandler: DOCK sequence started.")
            MC.slow_wall_approach()   # drive forward slowly until against the wall
            MC.stop_robot()
            shared["clawBusy"]     = True
            shared["engageHandle"] = True
            # Block until ServoHandler completes the unload and clears engageHandle.
            while not stop_event.is_set() and shared["engageHandle"]:
                time.sleep(0.02)
            log.info("MotorsHandler: unload complete.")

            if shared["finilisingState"]:
                # End of match – robot stays docked at the wall.
                log.info("MotorsHandler: finalising – triggering shutdown.")
                stop_event.set()
            else:
                MC.reverse_from_wall()
                # clawBusy and storageFull are already cleared by ServoHandler.
                # clawAdjusted was set to False on a prior iteration by the
                # finilisingState / storageFull nav branches (elif branches below
                # in this function), so ServoHandler will reopen the claw on its
                # next iteration.
                shared["dockingInProcess"] = False
                log.info("MotorsHandler: reversed from wall – resuming ball search.")

        # ── Navigating to AprilTag (end-of-game, tag not yet locked) ──────
        elif shared["finilisingState"]:
            # Retract claw before approaching the wall.
            if shared["clawAdjusted"]:
                shared["clawAdjusted"] = False   # ServoHandler will call clutch_up()
            if shared["moveTargetTag"] is not None:
                MC.apf_move(
                    angle_deg=shared["moveTargetTag"]["angle"],
                    magnitude=shared["moveTargetTag"]["magnitude"],
                )
            else:
                MC.pivot_left()   # no tag visible → rotate to search

        # ── Navigating to AprilTag (storage full, tag not yet locked) ─────
        elif shared["storageFull"]:
            # Retract claw before approaching the wall.
            if shared["clawAdjusted"]:
                shared["clawAdjusted"] = False   # ServoHandler will call clutch_up()
            if shared["moveTargetTag"] is not None:
                MC.apf_move(
                    angle_deg=shared["moveTargetTag"]["angle"],
                    magnitude=shared["moveTargetTag"]["magnitude"],
                )
            else:
                MC.pivot_left()   # no tag visible → rotate to search

        # ── Normal ball collection ─────────────────────────────────────────
        else:
            # pickingInProcess=False here, so no grab is in progress.
            # Just steer toward the current APF target or pivot to search.
            if shared["moveTargetBall"] is not None:
                MC.apf_move(
                    angle_deg=shared["moveTargetBall"]["angle"],
                    magnitude=shared["moveTargetBall"]["magnitude"],
                )
            else:
                MC.pivot_left()   # no ball visible → rotate to search

        time.sleep(0.02)

    log.info("MotorsHandler stopped.")


# ══════════════════════════════════════════════════════════════════════════════
# THREAD  –  Servo Handler // FINISHED BY MARK, RAEF HELPED WITH SERVOCONTROLLER
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

    pickingInProcess handoff
    ------------------------
      MotorsHandler sets engageClaw=True and clawBusy=True, then blocks on
      pickingInProcess.  This thread performs the grab, updates counts, and
      clears pickingInProcess as the LAST step so MotorsHandler unblocks only
      after the claw is fully reset and the robot is ready to move again.
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
                SC.clutch_up()
                log.info("Claw retracted to up position for delivery/end-of-game.")
            # ── Open claw for collection ──────────────────────────────────
            else:
                SC.clutch_down()
                log.info("Claw opened for collection.")
            shared["clawAdjusted"] = True

        else:
            # ── Unload mechanism (engageHandle) ───────────────────────────
            if shared["engageHandle"]:
                SC.mg90s_turn_by_180_up()
                time.sleep(2)                    # wait for balls to slide out
                SC.mg90s_turn_by_180_down()      # reset handle for next unload cycle

                log.info(
                    f"Unloaded {shared['heldPingPong']} PingPong "
                    f"+ {shared['heldSteel']} steel balls."
                )
                shared["heldPingPong"]  = 0
                shared["heldSteel"]     = 0
                shared["engageHandle"]  = False
                shared["storageFull"]   = False
                # clawAdjusted=False → ServoHandler will call clutch_down() on
                # the next iteration, reopening the claw for collection.
                shared["clawAdjusted"]  = False
                # Release grab lock; MotorsHandler unblocks from dockingInProcess
                # wait loop once engageHandle is False (cleared above), then
                # calls reverse_from_wall() and clears dockingInProcess itself.
                shared["clawBusy"]      = False

            # ── Ball capture claw (engageClaw) ────────────────────────────
            elif shared["engageClaw"]:
                SC.clutch_grabbing_motion()

                # ── Update held-ball count ────────────────────────────────
                # lockedBall is still set here because BallDetector is gated
                # out by pickingInProcess=True and ImageFrameCapture has
                # stopped distributing frames.
                ball_type = shared["lockedBall"]   # read BEFORE clearing
                if ball_type == "PingPong":
                    shared["heldPingPong"] = shared["heldPingPong"] + 1
                elif ball_type == "steel":
                    shared["heldSteel"] = shared["heldSteel"] + 1

                # Clear lockedBall and engageClaw before releasing locks.
                shared["lockedBall"]  = ""
                shared["engageClaw"]  = False
                shared["clawBusy"]    = False

                log.info(
                    f"Ball captured ({ball_type}). "
                    f"Held: PP={shared['heldPingPong']} steel={shared['heldSteel']}"
                )

                STORAGE_CAPACITY = 4  # PingPong=1 unit, steel=0.4 units;
                totalVal = shared["heldPingPong"] + 0.4 * shared["heldSteel"]

                if totalVal >= STORAGE_CAPACITY:
                    shared["storageFull"]  = True
                    shared["clawAdjusted"] = False   # triggers claw hide on next loop
                    log.info("Storage full – switching to unload mode.")

                # Clear pickingInProcess LAST – this is the signal that unblocks
                # MotorsHandler and re-enables image capture / detection.
                shared["pickingInProcess"] = False
                log.info("pickingInProcess cleared – robot resuming normal operation.")

        time.sleep(0.02)

    log.info("ServoHandler stopped.")


# ══════════════════════════════════════════════════════════════════════════════
# THREAD  –  Button Handler  (GPIO 4, via gpiozero) // FINISHED
# ══════════════════════════════════════════════════════════════════════════════
def setup_button_handler(shared, worker_pause_event, timer_pause_event, stop_event):
    """
    Configures gpiozero Button callbacks.
    Returns the Button object – the caller MUST keep a reference alive so
    gpiozero's background thread remains active.

    Button behaviour:
      Hold > 5 s  → full reset (pauses EVERYTHING incl. MainTimer, resets alg)
      Short press → if alg started: toggle pause for workers only (not MainTimer)
                    if alg not started: set both events to start the algorithm
    """
    button = Button(BUTTON_GPIO, hold_time=5, bounce_time=0.1)

    def on_held():
        """Triggered when button is held for longer than 5 seconds."""
        log.info("Button held 5 s – performing full reset.")
        shared["btnHeld"] = True

        # Pause ALL processes and threads, including the Main Timer.
        MC.stop_robot()
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
            if not worker_pause_event.is_set() and not timer_pause_event.is_set():
                # Neither event is set → first press; start the algorithm.
                worker_pause_event.set()
                timer_pause_event.set()
                log.info("Algorithm start requested.")
            else:
                if worker_pause_event.is_set():
                    # Currently running → pause workers.
                    MC.stop_robot()
                    worker_pause_event.clear()
                    log.info("Workers paused (Main Timer continues).")
                else:
                    # Currently paused → resume workers.
                    worker_pause_event.set()
                    log.info("Workers resumed.")

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

    # ── Setup Events ──────────────────────────────────────────────────────
    # Convention: event.set() = RUNNING, event.clear() = PAUSED
    worker_pause_event = multiprocessing.Event()   # workers (not MainTimer)
    timer_pause_event  = multiprocessing.Event()   # MainTimer only
    stop_event         = multiprocessing.Event()   # terminates all loops

    worker_pause_event.clear()   # start paused until button press
    timer_pause_event.clear()    # start paused until button press

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
            target=image_capture_thread,
            args=(ball_frame_queue, tag_frame_queue, shared, worker_pause_event, stop_event),
            name="ImageFrameCapture",
            daemon=True,
        ),
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