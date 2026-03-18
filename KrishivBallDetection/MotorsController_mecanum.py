## Credits to Krishiv
## Rewritten for Artificial Potential Fields (APF) with Mecanum kinematics

from gpiozero import Motor
import time
import math

# ==========================================
# TUNING CONSTANTS  ← adjust these values
# ==========================================

DEFAULT_SPEED = 60          # Base speed percentage (0-100)
TIME_PER_DEGREE = 0.015     # Seconds to rotate 1 degree at DEFAULT_SPEED
DEAD_ZONE_MAGNITUDE = 0.10  # APF magnitudes below this are treated as stop
STRAFE_DEAD_ZONE = 0.03     # Small strafe terms are suppressed to reduce jitter
FORWARD_DEAD_ZONE = 0.03    # Small forward terms are suppressed to reduce jitter

# ==========================================
# HARDWARE SETUP (L298N Motor Drivers)
# ==========================================
# Motor(forward_pin, backward_pin, enable_pin)

motor_fl = Motor(forward=17, backward=27, enable=18)   # Front Left  (A)
motor_rl = Motor(forward=22, backward=23, enable=12)   # Rear  Left  (B)
motor_fr = Motor(forward=6,  backward=26, enable=13)   # Front Right (C)
motor_rr = Motor(forward=20, backward=21, enable=19)   # Rear  Right (D)

# ==========================================
# LOW-LEVEL WHEEL CONTROL
# ==========================================

def _clamp_signed(speed: float) -> float:
    return max(-1.0, min(1.0, speed))


def _set_motor(motor: Motor, speed: float) -> None:
    speed = _clamp_signed(speed)
    if speed > 0:
        motor.forward(speed)
    elif speed < 0:
        motor.backward(abs(speed))
    else:
        motor.stop()


def set_wheel_speeds(fl: float, fr: float, rl: float, rr: float) -> None:
    """Set each wheel with signed speed in [-1.0, +1.0]."""
    _set_motor(motor_fl, fl)
    _set_motor(motor_fr, fr)
    _set_motor(motor_rl, rl)
    _set_motor(motor_rr, rr)


def stop_robot() -> None:
    """Immediately stop all four wheels."""
    set_wheel_speeds(0.0, 0.0, 0.0, 0.0)


# ==========================================
# MECANUM KINEMATICS
# ==========================================

def drive_vector(forward: float, strafe: float) -> None:
    """
    Drive the robot using a chassis vector with no commanded yaw.

    Parameters
    ----------
    forward : float
        +1.0 forward, -1.0 backward.
    strafe : float
        +1.0 right, -1.0 left.

    Mecanum wheel mix (no rotation term):
      FL = forward + strafe
      FR = forward - strafe
      RL = forward - strafe
      RR = forward + strafe

    The vector is normalized so no wheel exceeds +/-1.0.
    """
    fl = forward + strafe
    fr = forward - strafe
    rl = forward - strafe
    rr = forward + strafe

    max_mag = max(1.0, abs(fl), abs(fr), abs(rl), abs(rr))
    fl /= max_mag
    fr /= max_mag
    rl /= max_mag
    rr /= max_mag

    set_wheel_speeds(fl, fr, rl, rr)


# ==========================================
# APF VECTOR -> MECANUM COMMAND
# ==========================================

def apf_move(angle_deg: float, magnitude: float) -> None:
    """
    Convert APF output into mecanum forward/strafe motion.

    APF convention:
      angle_deg: 0 deg = straight ahead, positive = right, negative = left
      magnitude: [0.0, 1.0]
    """
    if magnitude < DEAD_ZONE_MAGNITUDE:
        stop_robot()
        return

    mag = max(0.0, min(1.0, magnitude))
    theta = math.radians(angle_deg)

    # Keep camera/chassis facing forward: translate in X/Y only (no yaw term).
    forward = mag * math.cos(theta)
    strafe = mag * math.sin(theta)

    if abs(forward) < FORWARD_DEAD_ZONE:
        forward = 0.0
    if abs(strafe) < STRAFE_DEAD_ZONE:
        strafe = 0.0

    drive_vector(forward, strafe)


# ==========================================
# CONVENIENCE WRAPPERS (API-compatible)
# ==========================================

def move_forward(speed: int = DEFAULT_SPEED) -> None:
    drive_vector(speed / 100.0, 0.0)


def move_backward(speed: int = DEFAULT_SPEED) -> None:
    drive_vector(-(speed / 100.0), 0.0)


def strafe_right(speed: int = DEFAULT_SPEED) -> None:
    drive_vector(0.0, speed / 100.0)


def strafe_left(speed: int = DEFAULT_SPEED) -> None:
    drive_vector(0.0, -(speed / 100.0))


def pivot_right(speed: int = DEFAULT_SPEED) -> None:
    """Rotate clockwise in place (used by search/fallback behaviours)."""
    s = speed / 100.0
    set_wheel_speeds(s, -s, s, -s)


def pivot_left(speed: int = DEFAULT_SPEED) -> None:
    """Rotate counter-clockwise in place (used by search/fallback behaviours)."""
    s = speed / 100.0
    set_wheel_speeds(-s, s, -s, s)


def pivot_right_degrees(degrees: float, speed: int = DEFAULT_SPEED) -> None:
    pivot_right(speed)
    time.sleep(degrees * TIME_PER_DEGREE)
    stop_robot()


def pivot_left_degrees(degrees: float, speed: int = DEFAULT_SPEED) -> None:
    pivot_left(speed)
    time.sleep(degrees * TIME_PER_DEGREE)
    stop_robot()


# ==========================================
# SPECIALISED BEHAVIOURS
# ==========================================

def slow_wall_approach(speed: int = 30, duration: float = 1.0) -> None:
    move_forward(speed)
    time.sleep(duration)
    stop_robot()


def reverse_from_wall(speed: int = DEFAULT_SPEED, duration: float = 0.75) -> None:
    move_backward(speed)
    time.sleep(duration)
    stop_robot()


def confident_approach_toGrab(speed: int = DEFAULT_SPEED, duration: float = 0.6) -> None:
    move_forward(speed)
    time.sleep(duration)
    stop_robot()
