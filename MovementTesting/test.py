## test.py
## Remote keyboard test runner for MotorsControllerTest + ServoControllerTest
## Run on Raspberry Pi 5 via SSH from laptop.
## Uses pynput so keypresses are captured even over an SSH terminal session.
##
## Install once on the Pi:
##   pip install pynput --break-system-packages

from pynput import keyboard as kb
import time

# ── Import both controllers ────────────────────────────────────────────────────
from MotorsControllerTest import (
    move_forward, move_backward,
    pivot_right, pivot_left,
    pivot_right_degrees, pivot_left_degrees,
    slow_wall_approach, reverse_from_wall,
    confident_approach_toGrab,
    apf_move, stop_robot,
)
from ServoControllerTest import (
    mg90s_turn_by_180_up, mg90s_turn_by_180_down, mg90s_stop,
    clutch_up, clutch_down, clutch_set_angle, clutch_grabbing_motion,
)

# ── ANSI colour helpers ────────────────────────────────────────────────────────
_GRN  = "\033[92m"
_YLW  = "\033[93m"
_CYN  = "\033[96m"
_RST  = "\033[0m"
_BLD  = "\033[1m"

def _header(text: str) -> None:
    print(f"\n{_BLD}{_CYN}{'─'*55}{_RST}")
    print(f"{_BLD}{_CYN}  {text}{_RST}")
    print(f"{_BLD}{_CYN}{'─'*55}{_RST}")

def _ok(label: str) -> None:
    print(f"  {_GRN}✔  {label}{_RST}")

def _info(label: str) -> None:
    print(f"  {_YLW}→  {label}{_RST}")

# ── Key → (label, callable) map ───────────────────────────────────────────────
#
#  MOTORS (letter keys)          SERVOS (digit keys)
#  ─────────────────────         ──────────────────────────────────
#  W  move_forward               1  mg90s_turn_by_180_up
#  S  move_backward              2  mg90s_turn_by_180_down
#  D  pivot_right                3  mg90s_stop
#  A  pivot_left                 4  clutch_up
#  E  pivot_right 45°            5  clutch_down
#  Q  pivot_left  45°            6  clutch_grabbing_motion
#  R  reverse_from_wall          7  clutch_set_angle  90°
#  F  confident_approach_toGrab  8  clutch_set_angle 180°
#  Z  slow_wall_approach
#  T  apf_move (45°, 0.7)
#  X  stop_robot  (emergency)
#
#  ESC / Ctrl-C  → quit

KEY_MAP: dict[str, tuple[str, callable]] = {

    # ── Motors ──────────────────────────────────────────────
    'w': ("move_forward  (default speed, 1 s)",
          lambda: (move_forward(), time.sleep(1), stop_robot())),

    's': ("move_backward  (default speed, 1 s)",
          lambda: (move_backward(), time.sleep(1), stop_robot())),

    'd': ("pivot_right  (default speed, 1 s)",
          lambda: (pivot_right(), time.sleep(1), stop_robot())),

    'a': ("pivot_left  (default speed, 1 s)",
          lambda: (pivot_left(), time.sleep(1), stop_robot())),

    'e': ("pivot_right_degrees  (45°)",
          lambda: pivot_right_degrees(45)),

    'q': ("pivot_left_degrees  (45°)",
          lambda: pivot_left_degrees(45)),

    'r': ("reverse_from_wall",
          reverse_from_wall),

    'f': ("confident_approach_toGrab",
          confident_approach_toGrab),

    'z': ("slow_wall_approach",
          slow_wall_approach),

    't': ("apf_move  (angle=45°, magnitude=0.7)",
          lambda: apf_move(45.0, 0.7)),

    'x': ("stop_robot  ⚠  EMERGENCY STOP",
          stop_robot),

    # ── Servos ──────────────────────────────────────────────
    '1': ("mg90s_turn_by_180_up",
          mg90s_turn_by_180_up),

    '2': ("mg90s_turn_by_180_down",
          mg90s_turn_by_180_down),

    '3': ("mg90s_stop",
          mg90s_stop),

    '4': ("clutch_up  (15°)",
          clutch_up),

    '5': ("clutch_down  (250°)",
          clutch_down),

    '6': ("clutch_grabbing_motion  (up → pause → down)",
          clutch_grabbing_motion),

    '7': ("clutch_set_angle  90°",
          lambda: clutch_set_angle(90.0)),

    '8': ("clutch_set_angle  180°",
          lambda: clutch_set_angle(180.0)),
}

# ── Print the key map on startup ──────────────────────────────────────────────
def _print_help() -> None:
    _header("ROBOT TEST RUNNER  —  key bindings")

    print(f"\n  {_BLD}── MOTORS ──────────────────────────────────{_RST}")
    motor_keys = ['w','s','a','d','e','q','r','f','z','t','x']
    for k in motor_keys:
        label, _ = KEY_MAP[k]
        print(f"    {_BLD}{k.upper()}{_RST}  →  {label}")

    print(f"\n  {_BLD}── SERVOS ──────────────────────────────────{_RST}")
    servo_keys = ['1','2','3','4','5','6','7','8']
    for k in servo_keys:
        label, _ = KEY_MAP[k]
        print(f"    {_BLD}{k}{_RST}  →  {label}")

    print(f"\n  {_BLD}ESC{_RST}  →  quit\n")

# ── Listener callbacks ─────────────────────────────────────────────────────────
_running = True

def _on_press(key: kb.Key) -> None:
    global _running

    # Quit on Escape
    if key == kb.Key.esc:
        _info("ESC pressed — stopping all motors and exiting …")
        try:
            stop_robot()
            mg90s_stop()
        except Exception:
            pass
        _running = False
        return False          # stops the listener

    # Extract the character (handles KeyChar and special keys gracefully)
    try:
        char = key.char.lower() if key.char else None
    except AttributeError:
        return   # special key we don't care about

    if char in KEY_MAP:
        label, action = KEY_MAP[char]
        _info(f"Running: {label}")
        try:
            action()
            _ok("Done")
        except Exception as exc:
            print(f"  \033[91m✘  ERROR: {exc}\033[0m")

# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    _print_help()
    print("  Listening for keypresses …  (ESC to quit)\n")

    listener = kb.Listener(on_press=_on_press)
    listener.start()

    # Keep the main thread alive while the listener thread runs
    try:
        while _running and listener.is_alive():
            time.sleep(0.05)
    except KeyboardInterrupt:
        _info("Ctrl-C received — shutting down …")
        try:
            stop_robot()
            mg90s_stop()
        except Exception:
            pass

    listener.stop()
    print("\n  Goodbye 👋\n")