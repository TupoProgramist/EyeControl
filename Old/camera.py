# camera.py

import cv2
import mediapipe as mp

# Initialize Mediapipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh

# Define land_dict with landmark indices
land_dict = {
    "right": {
        "left": 133,
        "right": 33,
        "upper": 27,
        "lower": 23,
        "center": 468
    },
    "left": {
        "left": 263,
        "right": 362,
        "upper": 257,
        "lower": 253,
        "center": 473
    },
    "face": {
        "left": 234,
        "right": 454,
        "upper": 10,
        "lower": 152,
        "center": 1
    }
}

def to_pixel_coords(landmark, image_shape):
    return int(landmark.x * image_shape[1]), int(landmark.y * image_shape[0])

def get_landmarks_and_pixels(cap):
    with mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.1,
            min_tracking_confidence=0.1) as face_mesh:

        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            return {}, {}, None

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

                return pos_dict, pix_dict, image

    return {}, {}, None
