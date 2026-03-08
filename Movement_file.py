from gpiozero import Motor
import time

# ==========================================
# HARDWARE SETUP (L298N Motor Drivers)
# ==========================================
# Using gpiozero's native Motor class optimized for Pi 5
# Format: Motor(forward_pin, backward_pin, enable_pin)

# L298N #1 (Front Wheels)
#Motor(IN1, IN2, ENA)
motor_fl = Motor(forward=11, backward=13, enable=12)  # Front Left
#Motor(IN3, IN4, ENB)
motor_fr = Motor(forward=15, backward=16, enable=32)  # Front Right

# L298N #2 (Rear Wheels)
motor_rl = Motor(forward=31,  backward=37,  enable=33)  # Rear Left
motor_rr = Motor(forward=38, backward=40, enable=35)  # Rear Right

# ==========================================
# HELPER FUNCTIONS
# ==========================================

def get_speed(speed_percentage):
    """Converts a 0-100 speed value into the 0.0-1.0 format required by gpiozero."""
    return max(0, min(100, speed_percentage)) / 100.0

def stop_robot():
    """Immediately stops all motor activity."""
    motor_fl.stop()
    motor_fr.stop()
    motor_rl.stop()
    motor_rr.stop()

# ==========================================
# CORE MOVEMENT FUNCTIONS
# ==========================================

def move_forward(speed=60):
    s = get_speed(speed)
    motor_fl.forward(s)
    motor_fr.forward(s)
    motor_rl.forward(s)
    motor_rr.forward(s)

def move_backward(speed=60):
    s = get_speed(speed)
    motor_fl.backward(s)
    motor_fr.backward(s)
    motor_rl.backward(s)
    motor_rr.backward(s)

def turn_right_in_place(speed=60):
    """Left wheels go forward, right wheels go backward."""
    s = get_speed(speed)
    motor_fl.forward(s)
    motor_rl.forward(s)
    motor_fr.backward(s)
    motor_rr.backward(s)

def turn_left_in_place(speed=60):
    """Left wheels go backward, right wheels go forward."""
    s = get_speed(speed)
    motor_fl.backward(s)
    motor_rl.backward(s)
    motor_fr.forward(s)
    motor_rr.forward(s)

# ==========================================
# DEGREE-BASED TURNING
# ==========================================
TIME_PER_DEGREE = 0.015  # Time required to turn 1 degree at 60% speed. Needs physical testing!

def turn_right_degrees(degrees, speed=60):
    turn_right_in_place(speed)
    time.sleep(degrees * TIME_PER_DEGREE)
    stop_robot()

def turn_left_degrees(degrees, speed=60):
    turn_left_in_place(speed)
    time.sleep(degrees * TIME_PER_DEGREE)
    stop_robot()

# ==========================================
# MECANUM STRAFING FUNCTIONS
# ==========================================

def strafe_right(speed=60):
    """Slides right: Front-Left & Rear-Right forward, Front-Right & Rear-Left backward."""
    s = get_speed(speed)
    motor_fl.forward(s)
    motor_rr.forward(s)
    motor_fr.backward(s)
    motor_rl.backward(s)

def strafe_left(speed=60):
    """Slides left: Front-Right & Rear-Left forward, Front-Left & Rear-Right backward."""
    s = get_speed(speed)
    motor_fr.forward(s)
    motor_rl.forward(s)
    motor_fl.backward(s)
    motor_rr.backward(s)

# ==========================================
# MAIN EXECUTION SEQUENCE
# ==========================================

def main():
    """Executes a diagnostic square pattern using Mecanum capabilities."""
    print("Starting optimized Pi 5 Mecanum sequence...")
    time.sleep(2) 

    try:
        print("Moving forward...")
        move_forward(speed=60)
        time.sleep(2)       
        stop_robot()
        time.sleep(0.5)     

        print("Strafing right...")
        strafe_right(speed=60)
        time.sleep(2)       
        stop_robot()
        time.sleep(0.5)

        print("Moving backward...")
        move_backward(speed=60)
        time.sleep(2)       
        stop_robot()
        time.sleep(0.5)

        print("Strafing left back to start...")
        strafe_left(speed=60)
        time.sleep(2)       
        stop_robot()
        time.sleep(0.5)

        print("Victory Spin! Spinning 360 degrees...")
        turn_right_degrees(360, speed=60)
        time.sleep(0.5)

        print("Sequence complete. Robot should be back where it started!")

    except KeyboardInterrupt:
        print("\nEmergency stop triggered by user!")
        stop_robot()
        
    finally:
        # gpiozero automatically handles all pin cleanup upon exit!
        stop_robot()
        print("Hardware safely shut down.")

if __name__ == "__main__":
    main()