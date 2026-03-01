import RPi.GPIO as GPIO
import time

# --- BCM Pin Mapping (from test code) ---
# L298N #1 (Front Wheels)
ENA_A, IN1_A, IN2_A = 18, 23, 24  # Front Left
ENB_B, IN3_B, IN4_B = 13, 27, 22  # Front Right

# L298N #2 (Rear Wheels)
ENA_C, IN1_C, IN2_C = 19, 5, 6    # Rear Left
ENB_D, IN3_D, IN4_D = 12, 16, 20  # Rear Right

# Setup
GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)

# Initialize all pins as outputs
pins = [ENA_A, IN1_A, IN2_A, ENB_B, IN3_B, IN4_B, ENA_C, IN1_C, IN2_C, ENB_D, IN3_D, IN4_D]
for pin in pins:
    GPIO.setup(pin, GPIO.OUT)
    GPIO.output(pin, GPIO.LOW)

# Initialize PWM for speed control (1000 Hz)
pwm_fl = GPIO.PWM(ENA_A, 1000) # Front Left
pwm_fr = GPIO.PWM(ENB_B, 1000) # Front Right
pwm_rl = GPIO.PWM(ENA_C, 1000) # Rear Left
pwm_rr = GPIO.PWM(ENB_D, 1000) # Rear Right

pwm_fl.start(0)
pwm_fr.start(0)
pwm_rl.start(0)
pwm_rr.start(0)

# ==========================================
# HELPER FUNCTION
# ==========================================
def set_motor(in1, in2, pwm_obj, direction, speed):
    """Controls a single motor's direction and speed (0-100%)."""
    pwm_obj.ChangeDutyCycle(speed)
    if direction == "forward":
        GPIO.output(in1, GPIO.HIGH)
        GPIO.output(in2, GPIO.LOW)
    elif direction == "backward":
        GPIO.output(in1, GPIO.LOW)
        GPIO.output(in2, GPIO.HIGH)
    else: # Stop
        GPIO.output(in1, GPIO.LOW)
        GPIO.output(in2, GPIO.LOW)
        pwm_obj.ChangeDutyCycle(0)

# ==========================================
# CORE MOVEMENT FUNCTIONS
# ==========================================

def stop_robot():
    """Stops all motors immediately."""
    set_motor(IN1_A, IN2_A, pwm_fl, "stop", 0)
    set_motor(IN3_B, IN4_B, pwm_fr, "stop", 0)
    set_motor(IN1_C, IN2_C, pwm_rl, "stop", 0)
    set_motor(IN3_D, IN4_D, pwm_rr, "stop", 0)

def move_forward(speed=60):
    set_motor(IN1_A, IN2_A, pwm_fl, "forward", speed)
    set_motor(IN3_B, IN4_B, pwm_fr, "forward", speed)
    set_motor(IN1_C, IN2_C, pwm_rl, "forward", speed)
    set_motor(IN3_D, IN4_D, pwm_rr, "forward", speed)

def move_backward(speed=60):
    set_motor(IN1_A, IN2_A, pwm_fl, "backward", speed)
    set_motor(IN3_B, IN4_B, pwm_fr, "backward", speed)
    set_motor(IN1_C, IN2_C, pwm_rl, "backward", speed)
    set_motor(IN3_D, IN4_D, pwm_rr, "backward", speed)

def turn_right_in_place(speed=60):
    """Left wheels forward, Right wheels backward"""
    set_motor(IN1_A, IN2_A, pwm_fl, "forward", speed)
    set_motor(IN1_C, IN2_C, pwm_rl, "forward", speed)
    set_motor(IN3_B, IN4_B, pwm_fr, "backward", speed)
    set_motor(IN3_D, IN4_D, pwm_rr, "backward", speed)

def turn_left_in_place(speed=60):
    """Left wheels backward, Right wheels forward"""
    set_motor(IN1_A, IN2_A, pwm_fl, "backward", speed)
    set_motor(IN1_C, IN2_C, pwm_rl, "backward", speed)
    set_motor(IN3_B, IN4_B, pwm_fr, "forward", speed)
    set_motor(IN3_D, IN4_D, pwm_rr, "forward", speed)

# ==========================================
# DEGREE-BASED TURNING (Time-Approximated)
# ==========================================
# You must calibrate 'time_per_degree_at_speed_60' through physical testing.
TIME_PER_DEGREE = 0.015 # Example: 0.015 seconds per degree

def turn_right_degrees(degrees, speed=60):
    turn_right_in_place(speed)
    # Note: If speed changes, time to turn changes. This is a simplified calculation.
    time.sleep(degrees * TIME_PER_DEGREE) 
    stop_robot()

def turn_left_degrees(degrees, speed=60):
    turn_left_in_place(speed)
    time.sleep(degrees * TIME_PER_DEGREE)
    stop_robot()

# ==========================================
# EXTRA FUNCTIONS: MECANUM STRAFING
# ==========================================

def strafe_right(speed=60):
    """Moves sideways to the right. 
    Front-Left & Rear-Right go forward. Front-Right & Rear-Left go backward."""
    set_motor(IN1_A, IN2_A, pwm_fl, "forward", speed)
    set_motor(IN3_D, IN4_D, pwm_rr, "forward", speed)
    set_motor(IN3_B, IN4_B, pwm_fr, "backward", speed)
    set_motor(IN1_C, IN2_C, pwm_rl, "backward", speed)

def strafe_left(speed=60):
    """Moves sideways to the left. 
    Front-Right & Rear-Left go forward. Front-Left & Rear-Right go backward."""
    set_motor(IN3_B, IN4_B, pwm_fr, "forward", speed)
    set_motor(IN1_C, IN2_C, pwm_rl, "forward", speed)
    set_motor(IN1_A, IN2_A, pwm_fl, "backward", speed)
    set_motor(IN3_D, IN4_D, pwm_rr, "backward", speed)

def cleanup():
    stop_robot()
    pwm_fl.stop()
    pwm_fr.stop()
    pwm_rl.stop()
    pwm_rr.stop()
    GPIO.cleanup()

def main():
    """Executes a set of movements to return the robot to its starting point."""
    print("Starting Mecanum movement sequence...")
    
    # Short delay before it starts
    time.sleep(2) 

    try:
        # 1. Move Forward
        print("Moving forward...")
        move_forward(speed=60)
        time.sleep(2)       # Run for 2 seconds
        stop_robot()
        time.sleep(0.5)     # Brief pause so the robot doesn't jerk

        # 2. Strafe Right
        print("Strafing right...")
        strafe_right(speed=60)
        time.sleep(2)       # Run for 2 seconds
        stop_robot()
        time.sleep(0.5)

        # 3. Move Backward
        print("Moving backward...")
        move_backward(speed=60)
        time.sleep(2)       # Run for 2 seconds
        stop_robot()
        time.sleep(0.5)

        # 4. Strafe Left (This should put it right back where it started)
        print("Strafing left back to start...")
        strafe_left(speed=60)
        time.sleep(2)       # Run for 2 seconds
        stop_robot()
        time.sleep(0.5)

        # 5. Victory Spin! (360 degrees)
        print("Spinning 360 degrees...")
        turn_right_degrees(360, speed=60)
        
        print("Sequence complete. Robot should be at the original position!")

    except KeyboardInterrupt:
        # This catches if you press Ctrl+C to emergency stop the program
        print("\nSequence interrupted by user! Stopping...")
        
    finally:
        # This guarantees that no matter what happens, the motors turn off
        print("Cleaning up GPIO pins...")
        cleanup()

if __name__ == "__main__":
    main()