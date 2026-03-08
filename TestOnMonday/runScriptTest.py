"""
Script to test button response, servo and basic movement

Functions to test:

Foward, Backward, Left, Right, Stop:
    - move.move_forward(speed=60) (from 0 to 100 only)
    - move.move_backward(speed=60)
    - move.turn_right_in_place(speed=60)
    - move.turn_left_in_place(speed=60)
    - move.strafe_right(speed=60)
    - move.strafe_left(speed=60)

    - move.stop_robot()

Servo Claw:
    - servo.clutch_set_angle(deg: float) (try 15, 200, 250, 270)

Servo Handle:
    - servo.mg90s_turn_by_180_up()
    - servo.mg90s_turn_by_180_down()

    *may need to adjust THROTTLE
"""

import time
import Movement_testing_file as move
import ServoCodeTest as servo
from gpiozero import Button

is_alg_running = False
was_held = False

def button_held():
    global was_held, is_alg_running
    was_held = True
    print("-> [BUTTON] 5-second hold!")
    is_alg_running = True 

def button_released():
    global was_held, is_alg_running
    if was_held:
        was_held = False
    else:
        is_alg_running = not is_alg_running

def test_button():
    print("-> [TEST] Button Pressed!")


if __name__ == '__main__':
    

    # Initialize the Button Thread
    btn = Button(7, hold_time=5, bounce_time=0.05)
    btn.when_held = button_held
    btn.when_released = button_released


    try:
        while True:
            if is_alg_running:
                #PLACE FUNCTION HERE
                test_button()
                is_alg_running = False
                    
            else:
                pass
                
            time.sleep(0.02) # Keep the main loop ticking at 50Hz
            
    except KeyboardInterrupt:
        print("Shutting down...")
        