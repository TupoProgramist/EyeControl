import cv2
import mediapipe as mp
import numpy as np

# Define land_dict with landmarks
land_dict = {
    "right": {
        "left": 133,  # Right side of the right eye
        "right": 33,  # Left side of the right eye
        "upper": 27,   # Upper point of the right eye
        "lower": 23,  # Lower point of the right eye
        "center": 168  # Center (pupil) of the right eye
    },
    "left": {
        "left": 263,   # Right side of the left eye
        "right": 362,  # Left side of the left eye
        "upper": 257,  # Upper point of the left eye
        "lower": 253,  # Lower point of the left eye
        "center": 473  # Center (pupil) of the left eye
    },
    "face": {
        "left": 234,   # Left edge of the face
        "right": 454,  # Right edge of the face
        "upper": 10,   # Forehead top
        "lower": 152,  # Chin
        "center": 1    # Nose tip
    }
}

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

# Function to convert normalized coordinates to pixel coordinates
def to_pixel_coords(landmark, image_shape):
    return int(landmark.x * image_shape[1]), int(landmark.y * image_shape[0])

def process_video():
    cap = cv2.VideoCapture(0)

    with mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.1,
            min_tracking_confidence=0.1) as face_mesh:

        while cap.isOpened():
            success, image = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                continue

            # Convert the BGR image to RGB for processing
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Process the image and detect face mesh
            results = face_mesh.process(image_rgb)

            # Convert back to BGR for OpenCV
            image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:                    

                    # Create pos_dict using land_dict
                    pos_dict = {
                        "right": {
                            "left": face_landmarks.landmark[land_dict["right"]["left"]],
                            "right": face_landmarks.landmark[land_dict["right"]["right"]],
                            "upper": face_landmarks.landmark[land_dict["right"]["upper"]],
                            "lower": face_landmarks.landmark[land_dict["right"]["lower"]],
                            "center": face_landmarks.landmark[land_dict["right"]["center"]]
                        },
                        "left": {
                            "left": face_landmarks.landmark[land_dict["left"]["left"]],
                            "right": face_landmarks.landmark[land_dict["left"]["right"]],
                            "upper": face_landmarks.landmark[land_dict["left"]["upper"]],
                            "lower": face_landmarks.landmark[land_dict["left"]["lower"]],
                            "center": face_landmarks.landmark[land_dict["left"]["center"]]
                        },
                        "face": {
                            "left": face_landmarks.landmark[land_dict["face"]["left"]],
                            "right": face_landmarks.landmark[land_dict["face"]["right"]],
                            "upper": face_landmarks.landmark[land_dict["face"]["upper"]],
                            "lower": face_landmarks.landmark[land_dict["face"]["lower"]],
                            "center": face_landmarks.landmark[land_dict["face"]["center"]]
                        }
                    }

                    # Create pix_dict using pos_dict
                    pix_dict = {
                        "right": {
                            "left": to_pixel_coords(pos_dict["right"]["left"], image.shape),
                            "right": to_pixel_coords(pos_dict["right"]["right"], image.shape),
                            "upper": to_pixel_coords(pos_dict["right"]["upper"], image.shape),
                            "lower": to_pixel_coords(pos_dict["right"]["lower"], image.shape),
                            "center": to_pixel_coords(pos_dict["right"]["center"], image.shape)
                        },
                        "left": {
                            "left": to_pixel_coords(pos_dict["left"]["left"], image.shape),
                            "right": to_pixel_coords(pos_dict["left"]["right"], image.shape),
                            "upper": to_pixel_coords(pos_dict["left"]["upper"], image.shape),
                            "lower": to_pixel_coords(pos_dict["left"]["lower"], image.shape),
                            "center": to_pixel_coords(pos_dict["left"]["center"], image.shape)
                        },
                        "face": {
                            "left": to_pixel_coords(pos_dict["face"]["left"], image.shape),
                            "right": to_pixel_coords(pos_dict["face"]["right"], image.shape),
                            "upper": to_pixel_coords(pos_dict["face"]["upper"], image.shape),
                            "lower": to_pixel_coords(pos_dict["face"]["lower"], image.shape),
                            "center": to_pixel_coords(pos_dict["face"]["center"], image.shape)
                        }
                    }

                    # Create cir_dict to store images with circles drawn
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

                    # Draw circles on the image
                    for side in ["right", "left"]:
                        for key in cir_dict[side]:
                            cir_dict[side][key]  # This applies the drawing onto the image

                    for key in cir_dict["face"]:
                        cir_dict["face"][key]  # This applies the drawing onto the image

            # Display the image
            cv2.imshow('Eye and Face Landmarks Tracking', image)

            # Check for 'R' key to exit
            if cv2.waitKey(5) & 0xFF == ord('r'):
                break

    cap.release()
    cv2.destroyAllWindows()

# Run the video processing
if __name__ == "__main__":
    process_video()
