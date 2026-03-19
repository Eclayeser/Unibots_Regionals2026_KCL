## Credits to Krishiv
## Rewritten for Artificial Potential Fields (APF) — smooth curved path support

from gpiozero import Motor
import time
import math

# ==========================================
# TUNING CONSTANTS  ← adjust these values
# ==========================================

DEFAULT_SPEED       = 35    # Base forward speed (0–100%)
MAX_TURN_ANGLE      = 90    # Angle (degrees) at which one side is fully stopped
TIME_PER_DEGREE     = 0.005 # Seconds to rotate 1° at 60% speed — needs physical calibration
DEAD_ZONE_MAGNITUDE = 0.15  # APF magnitudes below this are treated as "stop"
DEAD_ZONE_ANGLE     = 5     # Angles within ±5° are treated as straight ahead

# ==========================================
# HARDWARE SETUP (L298N Motor Drivers)
# ==========================================
# Motor(forward_pin, backward_pin, enable_pin)

# Left side  — share the same logical direction
motor_fl = Motor(forward=23, backward=27, enable=22)   # Front Left  (A)
motor_rl = Motor(forward=5, backward=6, enable=26)   # Rear  Left  (B)

# Right side — share the same logical direction
motor_fr = Motor(forward=24, backward=25, enable=16)   # Front Right (C)
motor_rr = Motor(forward=20, backward=21, enable=19)   # Rear  Right (D)

# ==========================================
# LOW-LEVEL WHEEL CONTROL
# ==========================================

def _set_side(motors: list, speed: float) -> None:
    """
    Drive a list of motors at a signed speed.
      speed > 0  → forward
      speed < 0  → backward
      speed = 0  → coast/stop
    speed is clamped to [-1.0, +1.0] before being sent to gpiozero.
    """
    speed = max(-1.0, min(1.0, speed))
    for m in motors:
        if speed > 0:
            m.forward(speed)
        elif speed < 0:
            m.backward(abs(speed))
        else:
            m.stop()

def set_left_speed(speed: float) -> None:
    """Drive both left wheels. speed ∈ [-1.0, +1.0]."""
    _set_side([motor_fl, motor_rl], speed)

def set_right_speed(speed: float) -> None:
    """Drive both right wheels. speed ∈ [-1.0, +1.0]."""
    _set_side([motor_fr, motor_rr], speed)

def stop_robot() -> None:
    """Immediately stop all four wheels."""
    set_left_speed(0)
    set_right_speed(0)

# ==========================================
# DIFFERENTIAL DRIVE  ← the core of curved motion
# ==========================================

def drive(base_speed: float, steering: float) -> None:
    """
    Differential drive mixer — produces curved paths.

    Parameters
    ----------
    base_speed : float
        Overall speed, signed.  +1.0 = full forward, -1.0 = full backward.
    steering : float
        Turn bias.  -1.0 = hard left, 0.0 = straight, +1.0 = hard right.

    How it works
    ------------
    The mixer scales left/right speeds proportionally so the robot follows
    an arc rather than stopping and pivoting:

        left_speed  = base_speed * (1 + steering)   ← clamped to [-1, 1]
        right_speed = base_speed * (1 - steering)   ← clamped to [-1, 1]

    A steering value of +0.5 with base_speed 0.6 gives:
        left  = 0.6 * 1.5 = 0.90  (outer wheel, faster)
        right = 0.6 * 0.5 = 0.30  (inner wheel, slower)
    → smooth right-hand curve.
    """
    left_speed  = base_speed * (1.0 + steering)
    right_speed = base_speed * (1.0 - steering)

    set_left_speed(left_speed)
    set_right_speed(right_speed)

# ==========================================
# APF VECTOR → MOTOR COMMAND
# ==========================================

def apf_move(angle_deg: float, magnitude: float) -> None:
    """
    Convert an APF resultant force vector into continuous curved motion.

    Parameters
    ----------
    angle_deg : float
        Direction of the APF force relative to the robot's heading.
        Positive = right of heading, negative = left of heading.
        Range: [-180, +180] degrees.

    magnitude : float
        Strength of the APF force, normalised to [0.0, 1.0].
        Values below DEAD_ZONE_MAGNITUDE are treated as zero (robot stops).

    Conversion logic
    ----------------
    1. Magnitude  → base_speed  (how fast to move)
    2. Angle      → steering    (how much to curve)

    steering is computed as a fraction of MAX_TURN_ANGLE so that
    small angle errors produce gentle curves and large angles produce
    tight arcs — all without stopping.
    """
    # --- Dead zone: stop if the APF force is negligible ---
    if magnitude < DEAD_ZONE_MAGNITUDE:
        stop_robot()
        return

    # --- Scale magnitude to a usable speed in [0, 1] ---
    base_speed = magnitude  # magnitude is already normalised 0–1

    # --- Convert angle to a steering bias in [-1, +1] ---
    # Clamp to ±MAX_TURN_ANGLE then normalise
    clamped_angle = max(-MAX_TURN_ANGLE, min(MAX_TURN_ANGLE, angle_deg))
    steering = clamped_angle / MAX_TURN_ANGLE   # ∈ [-1, +1]

    # --- Small-angle dead zone: ignore tiny heading errors ---
    if abs(angle_deg) < DEAD_ZONE_ANGLE:
        steering = 0.0

    drive(base_speed, steering)

# ==========================================
# CONVENIENCE WRAPPERS (unchanged behaviour)
# ==========================================

def move_forward(speed: int = DEFAULT_SPEED) -> None:
    """Drive straight forward at a given speed percentage (0–100)."""
    drive(speed / 100.0, steering=0.0)

def move_backward(speed: int = DEFAULT_SPEED) -> None:
    """Drive straight backward at a given speed percentage (0–100)."""
    drive(-(speed / 100.0), steering=0.0)

def pivot_right(speed: int = DEFAULT_SPEED) -> None:
    """Rotate clockwise on the spot (left fwd, right bwd)."""
    s = speed / 100.0
    set_left_speed(s)
    set_right_speed(-s)

def pivot_left(speed: int = DEFAULT_SPEED) -> None:
    """Rotate counter-clockwise on the spot (left bwd, right fwd)."""
    s = speed / 100.0
    set_left_speed(-s)
    set_right_speed(s)

def pivot_right_degrees(degrees: float, speed: int = 60) -> None:
    pivot_right(speed)
    time.sleep(degrees * TIME_PER_DEGREE)
    stop_robot()

def pivot_left_degrees(degrees: float, speed: int = 60) -> None:
    pivot_left(speed)
    time.sleep(degrees * TIME_PER_DEGREE)
    stop_robot()

# ==========================================
# SPECIALISED BEHAVIOURS
# ==========================================

# about 15 cm away from the wall, we want to slow down for a precise approach
def slow_wall_approach(speed: int = 30, duration: float = 0.45) -> None:
    """Creep forward slowly for precise wall alignment."""
    move_forward(speed)
    time.sleep(duration)
    stop_robot()

def reverse_from_wall(speed: int = DEFAULT_SPEED, duration: float = 0.30) -> None:
    """Back away from a wall after interaction."""
    move_backward(speed)
    time.sleep(duration)
    stop_robot()

def confident_approach_toGrab(speed: int = DEFAULT_SPEED, duration: float = 0.5) -> None:
    """Approach the ball confidently for a secure grab."""
    move_forward(speed)
    time.sleep(duration)
    stop_robot()

def move_forward_toStart(speed: int = 45, duration: float = 0.5) -> None:
    """Move forward to the starting position."""
    move_forward(speed)
    time.sleep(duration)
    stop_robot()

def little_reverse(speed: int = 30, duration: float = 0.2) -> None:
    """A short reverse to clear the claw after picking."""
    move_backward(speed)
    time.sleep(duration)
    stop_robot()