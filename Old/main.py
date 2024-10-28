# main.py

import recognition
import Old.UI as UI
import cv2
import threading

cap = cv2.VideoCapture(0)

def main():
    # Start the UI thread
    ui_thread = threading.Thread(target=UI.display_landmarks_on_image, args=(cap,))
    ui_thread.start()

if __name__ == "__main__":
    main()
