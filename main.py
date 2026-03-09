#!/usr/bin/env python3
"""
main.py  –  Robot Ball Collection Algorithm
Target platform: Raspberry Pi 5

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
"""

import multiprocessing
import threading
import time
import logging

from gpiozero import Button

# ─────────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s %(processName)s/%(threadName)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

BUTTON_GPIO = 7  # GPIO pin for the physical button


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
            # ── Check how much time has passed ────────────────────────────
            elapsed = time.time() - shared["timeStarted"]
            if elapsed > 160 and not shared["finilisingState"]:
                log.info("160 s elapsed – entering finalising state.")
                shared["finilisingState"] = True
        else:
            if shared["initialAlgStart"]:
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
    """
    log.info("BallDetector started.")

    while not stop_event.is_set():
        worker_pause_event.wait()

        # ── Grab the latest frame (non-blocking) ──────────────────────────
        frame = None
        try:
            frame = frame_queue.get_nowait()
        except Exception:
            pass

        # ── Validate frame ────────────────────────────────────────────────
        if frame is None:
            # latest_frame contains no image or corrupted image → skip
            time.sleep(0.02)
            continue

        # TODO: Check latest_frame:
        #       – Run colour-based ball detection (e.g. HSV thresholding or
        #         a trained model) on `frame`.
        #       – Return a priority-ordered list of detected balls, each as a
        #         dict: {"type": "PingPong"|"steel", "x": int, "y": int,
        #                "distance": float}
        #       – Sort by priority (e.g. steel first) then by distance
        #         (closest first).
        detected_balls = []   # placeholder  ← replace with real detection

        if not detected_balls:
            # ── No balls found → instruct robot to rotate and search ──────
            # TODO: Update moveTargetBall so the robot rotates to sweep for a ball.
            #       Populate with appropriate speed, angle, and direction values,
            #       e.g. {"speed": <slow>, "angle": 0, "direction": "rotate"}.
            shared["moveTargetBall"] = {"speed": 0, "angle": 0, "direction": "rotate"}

        else:
            # ── Balls detected → check if one is close enough to grab ─────
            # TODO: Determine whether the highest-priority ball is:
            #         (a) directly in front of the claw (centre-x ≈ frame centre), AND
            #         (b) close enough to be reliably captured (distance < GRAB_THRESHOLD).
            #       Set ball_almost_grabbed = True / False accordingly.
            #       Also set closest_ball_type = "PingPong" or "steel".
            ball_almost_grabbed  = False   # placeholder
            closest_ball_type    = ""      # placeholder

            if ball_almost_grabbed:
                # Ball is essentially in the claw – signal Motors to engage
                shared["lockedBall"] = closest_ball_type   # "PingPong" or "steel"

            else:
                shared["lockedBall"] = ""

                # TODO: Detect potential obstacles in the frame:
                #       walls and other robots (any object that is not white and
                #       not a ball).  Return a list of obstacle positions, e.g.
                #       [{"x": int, "y": int, "size": int}, ...].
                obstacles = []   # placeholder

                # TODO: Update moveTargetBall to navigate toward the ball:
                #       – Use the first entry in the priority list as the
                #         attraction point.
                #       – Treat every obstacle in `obstacles` as a repulsion
                #         point (e.g. potential-field or pure-pursuit with
                #         avoidance).
                #       – Populate speed, angle, and direction accordingly.
                shared["moveTargetBall"] = {"speed": 0, "angle": 0, "direction": "forward"}

        time.sleep(0.02)

    log.info("BallDetector stopped.")


# ══════════════════════════════════════════════════════════════════════════════
# PROCESS  –  AprilTag Detector
# ══════════════════════════════════════════════════════════════════════════════
def apriltag_detector_process(frame_queue, shared, worker_pause_event, stop_event):
    """
    Reads the latest camera frame, detects AprilTags, and updates
    shared["lockedTag"] and shared["moveTargetTag"].
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
            # latest_frame contains no image or corrupted image → skip
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
            # ── No tags found → instruct robot to rotate and search ───────
            # TODO: Update moveTargetTag so the robot rotates to sweep for a tag.
            #       Populate with appropriate speed, angle, and direction values,
            #       e.g. {"speed": <slow>, "angle": 0, "direction": "rotate"}.
            shared["moveTargetTag"] = {"speed": 0, "angle": 0, "direction": "rotate"}

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
                #   Populate speed, angle, and direction accordingly.
                shared["moveTargetTag"] = {"speed": 0, "angle": 0, "direction": "forward"}

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

    while not stop_event.is_set():
        worker_pause_event.wait()

        # TODO: Capture a new frame from the camera.
        #       Convert to BGR if using OpenCV:
        #           frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        new_frame = None   # placeholder  ← replace with real capture

        if new_frame is not None:
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
    """
    log.info("MotorsHandler started.")

    while not stop_event.is_set():
        worker_pause_event.wait()

        if shared["finilisingState"]:
            # ── End-of-game: unload and park ─────────────────────────────
            if shared["lockedTag"]:
                # TODO: Engage motors to approach the AprilTag target slowly
                #       and stop precisely in front of the unloading zone.
                shared["engageHandle"] = True
                time.sleep(3)   # wait for ServoHandler to complete unload

                # TODO: Execute parking manoeuvre (e.g. reverse to the
                #       designated parking spot and stop motors).
                log.info("Parking sequence initiated.")

                # Full shutdown – algorithm is done
                shared["algHasBeenStarted"] = False
                shared["initialAlgStart"]   = False
                stop_event.set()

            else:
                # TODO: Read shared["moveTargetTag"] (speed, angle, direction)
                #       and send the corresponding PWM values to the motor
                #       driver to navigate toward the unload tag.
                pass

        else:
            if shared["storageFull"]:
                # ── Storage full: navigate to unload zone ─────────────────
                if shared["lockedTag"]:
                    # TODO: Approach the AprilTag target slowly and stop
                    #       precisely in front of the unloading zone.
                    shared["engageHandle"] = True
                    time.sleep(3)   # wait for ServoHandler to complete unload

                    # TODO: Reverse a short distance away from the unloading
                    #       zone so the robot is clear to resume collection.
                    pass

                else:
                    # TODO: Read shared["moveTargetTag"] (speed, angle, direction)
                    #       and send the corresponding PWM values to the motor
                    #       driver to navigate toward the unload tag.
                    pass

            else:
                # ── Normal operation: collect balls ───────────────────────
                if shared["lockedBall"] != "":
                    # Ball is right in front of the claw – move in to capture
                    # TODO: Drive slowly forward until the claw is fully
                    #       around the ball (fine approach).
                    shared["engageClaw"] = True

                else:
                    # TODO: Read shared["moveTargetBall"] (speed, angle, direction)
                    #       and send the corresponding PWM values to the motor
                    #       driver to navigate toward the next ball.
                    pass

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
    """
    log.info("ServoHandler started.")

    while not stop_event.is_set():
        worker_pause_event.wait()

        # ── One-time initial claw adjustment ──────────────────────────────
        if not shared["clawAdjusted"]:
            # TODO: Send PWM signal to move the claw servo to 270 degrees
            #       (open / ready-to-capture position).
            #       Use RPi.GPIO or gpiozero Servo class on the claw's GPIO pin.
            shared["clawAdjusted"] = True
            log.info("Claw adjusted to initial position (270°).")

        # ── Unload mechanism (engageHandle) ───────────────────────────────
        if shared["engageHandle"]:
            # TODO: Move the handle / unload mechanism UP (PWM to upper limit),
            #       wait long enough for all held balls to slide out,
            #       then move it back DOWN (PWM to lower limit).
            held = list(shared["currentlyHeldBalls"])
            log.info(f"Unloading {held[0]} PingPong + {held[1]} steel balls.")
            shared["currentlyHeldBalls"] = [0, 0]
            shared["engageHandle"]        = False
            shared["storageFull"]         = False

        # ── Ball capture claw (engageClaw) ────────────────────────────────
        elif shared["engageClaw"]:
            # TODO: Move the claw servo DOWN (closed position) to grab the ball,
            #       hold briefly, then move it back UP (open position) so it is
            #       ready for the next ball.

            # TODO: Increment currentlyHeldBalls based on ball type:
            #         if shared["lockedBall"] == "PingPong" → currentlyHeldBalls[0] += 1
            #         if shared["lockedBall"] == "steel"    → currentlyHeldBalls[1] += 1
            #       Retrieve the mutable list, mutate it, then re-assign to shared
            #       so the Manager dict propagates the change.
            held = list(shared["currentlyHeldBalls"])
            ball_type = shared["lockedBall"]
            if ball_type == "PingPong":
                held[0] += 1
            elif ball_type == "steel":
                held[1] += 1
            shared["currentlyHeldBalls"] = held
            shared["engageClaw"]         = False
            log.info(f"Ball captured ({ball_type}). Held: {held}")

            # ── Storage-full check ────────────────────────────────────────
            # TODO: Replace STORAGE_CAPACITY with the real threshold value
            #       (maximum number of balls the robot can carry before
            #       it must visit the unload zone).
            STORAGE_CAPACITY = 6
            if sum(shared["currentlyHeldBalls"]) >= STORAGE_CAPACITY:
                shared["storageFull"] = True
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

        # Pause ALL processes and threads, including the Main Timer
        worker_pause_event.clear()
        timer_pause_event.clear()

        # Reset algorithm state so timer restarts fresh on next button press
        shared["algHasBeenStarted"] = False
        shared["initialAlgStart"]   = False
        shared["finilisingState"]   = False

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
                # Toggle pause for workers ONLY (Main Timer keeps running)
                if worker_pause_event.is_set():
                    # Currently running → pause workers
                    worker_pause_event.clear()
                    log.info("Workers paused (Main Timer continues).")
                else:
                    # Currently paused → resume workers
                    worker_pause_event.set()
                    log.info("Workers resumed.")
            else:
                # Algorithm has not been started yet → request start
                shared["initialAlgStart"] = True
                # Ensure both pause events are set (running) so all threads wake up
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
    shared = manager.dict({
        # ── Algorithm control ──────────────────────────────────────────────
        "initialAlgStart":    False,    # set True by ButtonHandler to kick off timer
        "algHasBeenStarted":  False,    # set True by MainTimer once running
        "timeStarted":        0.0,      # epoch timestamp of algorithm start
        "finilisingState":    False,    # set True after 160 s; triggers unload+park

        # ── Ball detection / navigation ────────────────────────────────────
        "lockedBall":         "",       # "" | "PingPong" | "steel"
        "moveTargetBall":     None,     # dict: {speed, angle, direction}

        # ── AprilTag detection / navigation ───────────────────────────────
        "lockedTag":          False,    # True when our unload tag is directly ahead
        "moveTargetTag":      None,     # dict: {speed, angle, direction}

        # ── Claw / servo control ───────────────────────────────────────────
        "engageClaw":         False,    # True → ServoHandler should grab a ball
        "engageHandle":       False,    # True → ServoHandler should unload balls
        "clawAdjusted":       False,    # True once initial 270° move is complete
        "currentlyHeldBalls": [0, 0],  # [pingpong_count, steel_count]
        "storageFull":        False,    # True → navigate to unload zone

        # ── Button ────────────────────────────────────────────────────────
        "btnHeld":            False,    # True if button was held for >= 5 s
    })

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