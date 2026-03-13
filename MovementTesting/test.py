"""
test.py — Interactive keyboard test for MotorsControllerTest & ServoControllerTest
====================================================================================
Works over SSH — no X11/display required.

Setup:
    pip install readchar gpiozero adafruit-circuitpython-servokit

Run:
    python test.py
"""

import sys
import time
import threading
import readchar   # SSH-safe, no X11 needed

# ─── Safe imports with friendly error messages ────────────────────────────────

try:
    import MotorsControllerTest as motors
    MOTORS_AVAILABLE = True
except ImportError as e:
    print(f"[WARN] MotorsControllerTest not found: {e}")
    MOTORS_AVAILABLE = False

try:
    import ServoControllerTest as servo
    SERVO_AVAILABLE = True
except ImportError as e:
    print(f"[WARN] ServoControllerTest not found: {e}")
    SERVO_AVAILABLE = False

# ─── Colour helpers (ANSI) ────────────────────────────────────────────────────

RESET  = "\033[0m"
BOLD   = "\033[1m"
CYAN   = "\033[96m"
GREEN  = "\033[92m"
YELLOW = "\033[93m"
RED    = "\033[91m"
DIM    = "\033[2m"

def ok(msg):    print(f"  {GREEN}✔  {msg}{RESET}")
def warn(msg):  print(f"  {YELLOW}⚠  {msg}{RESET}")
def err(msg):   print(f"  {RED}✖  {msg}{RESET}")
def info(msg):  print(f"  {CYAN}→  {msg}{RESET}")

# ─── Key map definition ───────────────────────────────────────────────────────

# Each entry: key_char → (label, callable, args_tuple)
KEY_MAP = {}

if MOTORS_AVAILABLE:
    KEY_MAP.update({
        "w": ("Move FORWARD",           motors.move_forward,             ()),
        "s": ("Move BACKWARD",          motors.move_backward,            ()),
        "a": ("Pivot LEFT (spot)",       motors.pivot_left,               ()),
        "d": ("Pivot RIGHT (spot)",      motors.pivot_right,              ()),
        "j": ("Pivot LEFT  45°",         motors.pivot_left_degrees,       (45,)),
        "l": ("Pivot RIGHT 45°",         motors.pivot_right_degrees,      (45,)),
        "k": ("STOP robot",             motors.stop_robot,               ()),
        "z": ("Slow wall approach",     motors.slow_wall_approach,       ()),
        "x": ("Reverse from wall",      motors.reverse_from_wall,        ()),
        "c": ("Confident approach/grab",motors.confident_approach_toGrab,()),
        "m": ("APF move (test vector)", None, None),  
    })

if SERVO_AVAILABLE:
    KEY_MAP.update({
        "u": ("MG90S turn 180° UP",      servo.mg90s_turn_by_180_up,   ()),
        "i": ("MG90S turn 180° DOWN",    servo.mg90s_turn_by_180_down,  ()),
        "o": ("MG90S STOP",              servo.mg90s_stop,              ()),
        "p": ("Clutch UP  (15°)",        servo.clutch_up,               ()),
        "[": ("Clutch DOWN (250°)",      servo.clutch_down,             ()),
        "]": ("Clutch GRAB motion",      servo.clutch_grabbing_motion,  ()),
        "n": ("Clutch set angle (test)", servo.clutch_set_angle,        (30)),   # special
    })

# ─── Print control table ──────────────────────────────────────────────────────[]

def print_help():
    print()
    print(f"{BOLD}{CYAN}{'═'*58}{RESET}")
    print(f"{BOLD}{CYAN}  ROBOT TEST CONSOLE  —  press Q to quit{RESET}")
    print(f"{BOLD}{CYAN}{'═'*58}{RESET}")

    if MOTORS_AVAILABLE:
        print(f"\n  {BOLD}── MOTORS ──────────────────────────────────{RESET}")
        motor_keys = ["w","s","a","d","j","l","k","z","x","c","m"]
        for k in motor_keys:
            if k in KEY_MAP:
                label = KEY_MAP[k][0]
                print(f"  {BOLD}{YELLOW}[{k.upper()}]{RESET}  {label}")
    else:
        warn("Motors module unavailable — motor keys disabled")

    if SERVO_AVAILABLE:
        print(f"\n  {BOLD}── SERVOS ──────────────────────────────────{RESET}")
        servo_keys = ["u","i","o","p","[","]","n"]
        for k in servo_keys:
            if k in KEY_MAP:
                label = KEY_MAP[k][0]
                print(f"  {BOLD}{YELLOW}[{k.upper()}]{RESET}  {label}")
    else:
        warn("Servo module unavailable — servo keys disabled")

    print(f"\n  {BOLD}{RED}[Q]{RESET}  Quit")
    print(f"  {DIM}[H]{RESET}  Show this help again")
    print(f"{BOLD}{CYAN}{'═'*58}{RESET}\n")

# ─── Action runner ────────────────────────────────────────────────────────────

_action_lock = threading.Lock()   # prevent key-mashing races

def run_action(char: str):
    """Dispatch a keypress to the appropriate function (non-blocking thread)."""
    if not _action_lock.acquire(blocking=False):
        warn("Still executing previous command — please wait.")
        return

    def _worker():
        try:
            entry = KEY_MAP.get(char)
            if entry is None:
                return

            label, fn, args = entry

            # ── Special cases that need interactive prompts ──────────────────
            
            if char == "m" and MOTORS_AVAILABLE:
                info("APF test — using angle=30°, magnitude=0.8")
                motors.apf_move(-45, 0.6)
                time.sleep(0.8)
                motors.stop_robot()
                ok(f"APF move complete")
                return
            """
            if char == "n" and SERVO_AVAILABLE:
                info("Clutch set-angle test — moving to 90°")
                servo.clutch_set_angle(90)
                ok("Clutch set to 90°")
                return
            """
            # ── Normal dispatch ──────────────────────────────────────────────
            info(f"Executing: {label}")
            fn(*args)
            ok(f"{label} — done")

        except Exception as exc:
            err(f"Error in '{label}': {exc}")
        finally:
            _action_lock.release()

    threading.Thread(target=_worker, daemon=True).start()

# ─── Shutdown helper ─────────────────────────────────────────────────────────

def safe_shutdown():
    print(f"\n{BOLD}{RED}Quitting — stopping all motors/servos…{RESET}", flush=True)
    if MOTORS_AVAILABLE:
        try: motors.stop_robot()
        except Exception: pass
    if SERVO_AVAILABLE:
        try: servo.mg90s_stop()
        except Exception: pass
    print("Goodbye.\n", flush=True)

# ─── Main loop ────────────────────────────────────────────────────────────────

def main():
    print_help()

    while True:
        try:
            key = readchar.readkey().lower()
        except KeyboardInterrupt:
            safe_shutdown()
            sys.exit(0)

        if key == "q":
            safe_shutdown()
            sys.exit(0)

        if key == "h":
            print_help()
            continue

        if key not in KEY_MAP:
            warn(f"Unknown key '{key}' — press H for help")
            continue

        run_action(key)

if __name__ == "__main__":
    main()