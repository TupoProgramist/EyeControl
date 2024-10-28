# UI.py

import cv2
import recognition  # Import the whole camera module
import numpy as np

def display_landmarks_on_image(cap):
    while True:
        pos_dict, pix_dict, image = recognition.get_landmarks_and_pixels(cap)

        if pos_dict and pix_dict:
            # Define circle drawing information for each point
            cir_dict = {
                "right": {
                    "left": cv2.circle(image, pix_dict["right"]["left"], 2, (0, 0, 255), -1),
                    "right": cv2.circle(image, pix_dict["right"]["right"], 2, (255, 255, 0), -1),
                    "upper": cv2.circle(image, pix_dict["right"]["upper"], 2, (255, 0, 0), -1),
                    "lower": cv2.circle(image, pix_dict["right"]["lower"], 2, (0, 255, 0), -1),
                    "center": cv2.circle(image, pix_dict["right"]["center"], 2, (255, 0, 255), -1),
                },
                "left": {
                    "left": cv2.circle(image, pix_dict["left"]["left"], 2, (0, 0, 255), -1),
                    "right": cv2.circle(image, pix_dict["left"]["right"], 2, (255, 255, 0), -1),
                    "upper": cv2.circle(image, pix_dict["left"]["upper"], 2, (255, 0, 0), -1),
                    "lower": cv2.circle(image, pix_dict["left"]["lower"], 2, (0, 255, 0), -1),
                    "center": cv2.circle(image, pix_dict["left"]["center"], 2, (255, 0, 255), -1),
                },
                "face": {
                    "chin": cv2.circle(image, pix_dict["face"]["lower"], 2, (255, 255, 255), -1),
                    "nose_tip": cv2.circle(image, pix_dict["face"]["center"], 2, (0, 255, 255), -1),
                    "forehead_top": cv2.circle(image, pix_dict["face"]["upper"], 2, (255, 0, 255), -1),
                    "left_face_edge": cv2.circle(image, pix_dict["face"]["left"], 2, (255, 255, 0), -1),
                    "right_face_edge": cv2.circle(image, pix_dict["face"]["right"], 2, (0, 255, 0), -1),
                }
            }

            # Execute the drawing commands from cir_dict
            for side in ["right", "left"]:
                for key in cir_dict[side]:
                    cir_dict[side][key]

            for key in cir_dict["face"]:
                cir_dict["face"][key]

            # Show the image
            cv2.imshow('Landmarks', image)

        if cv2.waitKey(5) & 0xFF == ord('r'):
            break

    cap.release()
    cv2.destroyAllWindows()
