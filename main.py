import threading
import cv2
import recognition
import UI
import time
import system  # Import the new system module
import keyboard
import json

with open('config.json', 'r') as config_file:
    config = json.load(config_file)

pac_rec = config['pac_rec']
pac_UI = config['pac_UI']*pac_rec

# Shared data structures
pos_dict = {}
pix_dict = {}

# Apply the timer decorator to the functions
recognition.get_landmarks_and_pixels = system.timer(pac_rec)(recognition.get_landmarks_and_pixels)
UI.display_landmarks_on_image = system.timer(pac_UI)(UI.display_landmarks_on_image)

def recognition_thread():
    global pos_dict, pix_dict
    cap = recognition.initialize_camera()  # Initialize the camera using 'recognition' module
    while True:
        pos_dict, pix_dict, _ = recognition.get_landmarks_and_pixels(cap)
        
def ui_thread():
    global pos_dict, pix_dict
    while True:
        UI.display_landmarks_on_image(pos_dict, pix_dict)

def main():
    # Start the recognition thread
    rec_thread = threading.Thread(target=recognition_thread)
    rec_thread.start()

    # Start the UI thread
    ui_thread_instance = threading.Thread(target=ui_thread)
    ui_thread_instance.start()

    try:
        # Main loop to handle "q" button press
        while True:
            if keyboard.is_pressed('q'):  # Check if "q" is pressed
                system.quit()  # Use the quit function from system.py to close the application
            time.sleep(0.1)  # Small sleep to avoid overloading the CPU
    except KeyboardInterrupt:
        system.quit()

if __name__ == "__main__":
    main()
