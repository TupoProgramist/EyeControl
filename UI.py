# UI.py

import cv2
import sys

def display_landmarks_on_image(pos_dict, pix_dict):
    # Use the frame from the camera instead of a blank image
    if pos_dict and pix_dict and "image" in pix_dict:
        image = pix_dict["image"]
    else:
        return  # Exit if no image is available

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
                "left": cv2.circle(image, pix_dict["face"]["left"], 2, (255, 255, 0), -1),
                "right": cv2.circle(image, pix_dict["face"]["right"], 2, (0, 255, 0), -1),
                "upper": cv2.circle(image, pix_dict["face"]["upper"], 2, (255, 0, 255), -1),
                "lower": cv2.circle(image, pix_dict["face"]["lower"], 2, (255, 255, 255), -1),
                "center": cv2.circle(image, pix_dict["face"]["center"], 2, (0, 255, 255), -1),
            }
        }

        # Execute the drawing commands from cir_dict
        for side in ["right", "left"]:
            for key in cir_dict[side]:
                cir_dict[side][key]

        for key in cir_dict["face"]:
            cir_dict["face"][key]

        # Show the image with landmarks
        cv2.imshow('Landmarks', image)

    if cv2.waitKey(5) & 0xFF == ord('r'):
        cv2.destroyAllWindows()
        sys.exit()  # Immediately exit the program
