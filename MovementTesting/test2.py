"""
test2.py — Sequential individual wheel test
=============================================
Tests each wheel one at a time: forward then backward.
No keypresses needed — just watch and confirm each wheel spins.

Run:  python test2.py
"""

from gpiozero import Motor
import time

# ─── ANSI colours ─────────────────────────────────────────────────────────────
RESET  = "\033[0m"
BOLD   = "\033[1m"
CYAN   = "\033[96m"
GREEN  = "\033[92m"
YELLOW = "\033[93m"
RED    = "\033[91m"

def ok(msg):   print(f"  \033[92m✔  {msg}{RESET}", flush=True)
def info(msg): print(f"  {CYAN}→  {msg}{RESET}", flush=True)
def header(msg): print(f"\n{BOLD}{YELLOW}{'─'*50}\n  {msg}\n{'─'*50}{RESET}", flush=True)

# ─── Motor definitions (update GPIOs here if you changed any pins) ─────────────

MOTORS = [
    {
        "name": "Wheel A — Front Left  (L298N #1, Motor A)",
        "forward": 17,
        "backward": 27,
        "enable": 18,
    },
    {
        "name": "Wheel B — Rear Left   (L298N #1, Motor B)",
        "forward": 5,
        "backward": 6,
        "enable": 26,
    },
    {
        "name": "Wheel C — Front Right (L298N #2, Motor C)",
        "forward": 24,
        "backward": 25,
        "enable": 16,
    },
    {
        "name": "Wheel D — Rear Right  (L298N #2, Motor D)",
        "forward": 20,
        "backward": 21,
        "enable": 19,
    },
]

# ─── Timing config ─────────────────────────────────────────────────────────────

SPIN_DURATION  = 2.0   # seconds to spin in each direction
PAUSE_BETWEEN  = 1.0   # seconds to pause between forward/backward
GAP_AFTER      = 2.0   # seconds to pause between wheels

TEST_SPEED     = 0.6   # 0.0–1.0  (60% speed — safe for benchtop testing)

# ─── Test one wheel ────────────────────────────────────────────────────────────

def test_wheel(config: dict):
    header(config["name"])
    try:
        motor = Motor(
            forward=config["forward"],
            backward=config["backward"],
            enable=config["enable"],
        )

        info(f"FORWARD  for {SPIN_DURATION}s  (GPIO fwd={config['forward']})")
        motor.forward(TEST_SPEED)
        time.sleep(SPIN_DURATION)
        motor.stop()
        ok("Stopped")

        time.sleep(PAUSE_BETWEEN)

        info(f"BACKWARD for {SPIN_DURATION}s  (GPIO bwd={config['backward']})")
        motor.backward(TEST_SPEED)
        time.sleep(SPIN_DURATION)
        motor.stop()
        ok("Stopped")

        motor.close()   # release GPIO pins cleanly
        ok(f"{config['name'].split('—')[0].strip()} — test complete")

    except Exception as e:
        print(f"  {RED}✖  FAILED: {e}{RESET}", flush=True)

# ─── Main ──────────────────────────────────────────────────────────────────────

def main():
    print(f"\n{BOLD}{CYAN}{'═'*50}")
    print(f"  WHEEL SEQUENTIAL TEST")
    print(f"  Each wheel: {SPIN_DURATION}s forward → {SPIN_DURATION}s backward")
    print(f"  Speed: {int(TEST_SPEED*100)}%")
    print(f"{'═'*50}{RESET}\n")

    input(f"  {BOLD}Press ENTER to start…{RESET} ")

    for cfg in MOTORS:
        test_wheel(cfg)
        time.sleep(GAP_AFTER)

    print(f"\n{BOLD}{GREEN}{'═'*50}")
    print(f"  ALL WHEELS TESTED")
    print(f"{'═'*50}{RESET}\n")

if __name__ == "__main__":
    main()