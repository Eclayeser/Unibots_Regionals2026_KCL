"""
To make this run as background service on Pi

sudo nano /etc/systemd/system/robot_button.service

---------------------------------------------------------
[Unit]
Description=Robot Button Listener Service
After=network.target

[Service]
ExecStart=/usr/bin/python3 #PATH_TO_YOUR_SCRIPT#/runScriptButton.py
WorkingDirectory=#PATH_TO_YOUR_SCRIPT#
StandardOutput=inherit
StandardError=inherit
Restart=always
User=#MY_USERNAME#

[Install]
WantedBy=multi-user.target
---------------------------------------------------------

CTRL+O, Enter, CTRL+X to save and exit.

TO ENABLE AND START THE SERVICE:

sudo systemctl daemon-reload
sudo systemctl enable robot_button.service
sudo systemctl start robot_button.service

TO CHECK STATUS:

sudo systemctl status robot_button.service

IMPORTANT

Called functions should be non-blocking and return quickly to avoid missing button events.
Example: change a variable

"""


import time
import threading
import multiprocessing
import cv2
from gpiozero import Button

# ==========================================
# 1. THREAD SHARED MEMORY (The Kitchen)
# ==========================================
# Threads share memory, so we can use standard globals and Locks
is_alg_running = False
was_held = False

latest_frame = None
frame_lock = threading.Lock()

# ==========================================
# 2. THREADS (Handling I/O & Hardware)
# ==========================================
def button_held():
    global was_held, is_alg_running
    was_held = True
    print("-> [BUTTON] 5-second hold! Forcing Restart...")
    is_alg_running = True 

def button_released():
    global was_held, is_alg_running
    if was_held:
        was_held = False
    else:
        is_alg_running = not is_alg_running
        state = "RUNNING" if is_alg_running else "PAUSED"
        print(f"-> [BUTTON] Short press! Alg is now: {state}")

def camera_worker():
    """Thread: Grabs frames and ensures they are a manageable resolution."""
    global latest_frame
    
    cap = cv2.VideoCapture(0)
    
    # --- 1. ATTEMPT HARDWARE DOWNSCALING ---
    # We ask the camera hardware to switch to 640x480.
    DESIRED_WIDTH = 640
    DESIRED_HEIGHT = 480
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, DESIRED_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, DESIRED_HEIGHT)
    
    # Read one frame to see if the camera actually listened to us
    ret, test_frame = cap.read()
    if ret:
        actual_height, actual_width, _ = test_frame.shape
        print(f"-> [CAMERA] Requested {DESIRED_WIDTH}x{DESIRED_HEIGHT}.")
        print(f"-> [CAMERA] Actual resolution is {actual_width}x{actual_height}.")
        
        # Check if the camera ignored our hardware request
        needs_software_resize = (actual_width != DESIRED_WIDTH)
    else:
        print("-> [CAMERA] Error: Could not read from webcam.")
        return

    print("-> [CAMERA THREAD] Started capturing.")
    
    while True:
        ret, frame = cap.read()
        if ret:
            
            # --- 2. BACKUP SOFTWARE DOWNSCALING ---
            # If the camera ignored cap.set(), squish the image manually
            if needs_software_resize:
                frame = cv2.resize(frame, (DESIRED_WIDTH, DESIRED_HEIGHT))
            
            # Safely hand the frame to the global variable
            with frame_lock:
                latest_frame = frame
                
        time.sleep(0.01)

# ==========================================
# 3. MULTIPROCESSING (Heavy CPU Math)
# ==========================================
def heavy_math_worker(in_queue, out_queue):
    """Process: Runs on a totally separate CPU core. No globals allowed!"""
    print("-> [MATH PROCESS] Started on Core 2.")
    
    while True:
        # .get() will wait here until the main loop sends it a frame
        frame_to_process = in_queue.get() 
        
        # --- SIMULATE HEAVY MATH ---
        # (Put your AprilTag detector or Neural Network code here)
        # If this took 0.5 seconds in a thread, the robot would stutter.
        # In a Process, the robot drives smoothly while this thinks!
        time.sleep(0.5) 
        
        # Pretend we found an AprilTag at these coordinates
        fake_tag_x = 320
        fake_tag_y = 240
        
        # Package the results into a dictionary and tube it back to the main loop
        result = {"x": fake_tag_x, "y": fake_tag_y, "found": True}
        out_queue.put(result)

# ==========================================
# 4. MAIN SCRIPT (The Brain)
# ==========================================
# Rule: Multiprocessing setup MUST be inside this __name__ block
if __name__ == '__main__':
    
    # 1. Create the pneumatic tubes (Queues) for cross-process communication
    frame_tube = multiprocessing.Queue()
    result_tube = multiprocessing.Queue()
    
    # 2. Start the Heavy Math Process
    math_process = multiprocessing.Process(
        target=heavy_math_worker, 
        args=(frame_tube, result_tube), # Pass the tubes as arguments!
        daemon=True
    )
    math_process.start()

    # 3. Start the Camera Thread
    threading.Thread(target=camera_worker, daemon=True).start()

    # 4. Initialize the Button Thread
    btn = Button(14, hold_time=5, bounce_time=0.05)
    btn.when_held = button_held
    btn.when_released = button_released

    print("-> [MAIN] Robot booted. Press button to start.")

    try:
        while True:
            if is_alg_running:
                
                # --- A. GRAB LATEST FRAME ---
                current_frame = None
                with frame_lock:
                    if latest_frame is not None:
                        current_frame = latest_frame.copy()
                
                # --- B. SEND TO MATH PROCESS (If it's ready) ---
                # We only send a frame if the tube is empty, meaning the 
                # math process has finished its last job and is waiting for a new one.
                if current_frame is not None and frame_tube.empty():
                    frame_tube.put(current_frame)
                    
                # --- C. CHECK FOR MATH RESULTS ---
                # .empty() is non-blocking. It checks the tube instantly and moves on.
                if not result_tube.empty():
                    tag_data = result_tube.get()
                    print(f"-> [MAIN] Received tag data: X:{tag_data['x']} Y:{tag_data['y']}")
                    
                    # You now have fresh coordinates! Move your motors!
                    # drive_motors_towards(tag_data['x'], tag_data['y'])
                    
            else:
                # Button is paused. Stop motors instantly.
                pass
                
            time.sleep(0.02) # Keep the main loop ticking at 50Hz
            
    except KeyboardInterrupt:
        print("Shutting down...")
        # Always clean up your Processes so they don't turn into zombie background tasks
        math_process.terminate()
        math_process.join()