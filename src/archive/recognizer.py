import cv2
import mediapipe as mp
import numpy as np
import os

import utils.helpers as helpers
import numpy as np
from algorithm.eigenface_enhanced import EnhancedEigenface
from algorithm.eigenface_orig import Eigenface
from sklearn.model_selection import train_test_split

label = 8

# Initialize Mediapipe Face Detection and Face Mesh modules
mp_face_detection = mp.solutions.face_detection
mp_face_mesh = mp.solutions.face_mesh

# Parameters
WINDOW_WIDTH = 1920  # Fixed width of the output window
WINDOW_HEIGHT = 1080  # Fixed height of the output window
FIXED_FACE_SIZE = 900  # Desired face size in pixels
SMOOTHING_FACTOR = 0.4  # Smoothing factor for translation
SMOOTHING_SCALE = 0.4  # Smoothing factor for scaling
THRESHOLD_POSITION = 15  # Threshold for position changes
THRESHOLD_SCALE = 0.2  # Threshold for scale changes
EYE_TO_FACE_CENTER_OFFSET = 50  # How much to move the face upward relative to the eyes

# Initialize previous translation values and size
prev_translate_x = 0
prev_translate_y = 0
prev_scale_factor = 1.0  # Initial scale factor
rt_prev_translate_x = 0
rt_prev_translate_y = 0
rt_prev_scale_factor = 1.0  # Initial scale factor
prev_bbox = None

# Initialize the person's name
person_surname = "SURNAME"
person_name = "FIRSTNAME"

N = 100
train_images, train_labels = helpers.load_images_labels("dataset/captured_faces", N, N)
# M = int(len(train_images) * 0.15)
M = len(train_images)
# enhanced_eigenface = EnhancedEigenface(train_images, train_labels, M)
enhanced_eigenface = Eigenface(train_images, train_labels, M)
enhanced_eigenface.fit()

# face_dict = {
#     0: "Mabel",
#     1: "Amyr",
#     2: "Virginia",
#     3: "Eddie",
#     4: "Ianh",
#     5: "Jared",
#     6: "James",
#     7: "Gelo",
#     8: "Lyka",
# }

face_dict = {
    0: "Amyr",
    1: "Gelo",
    2: "Lyka",
}


def predict(image):
    preprocessed_image = helpers.preprocess_image(image, N, N)
    predicted_labels, min_weight_distances, projection_distances = (
        enhanced_eigenface.predict(np.array([preprocessed_image]))
    )

    return predicted_labels[0]


def smooth_coordinates(current, previous, smoothing):
    """Smooths the coordinates using a moving average."""
    return previous + (current - previous) * smoothing


def apply_threshold(value, previous, threshold, smoothing):
    """Applies a threshold to ignore small changes."""
    if abs(value - previous) > threshold:
        return previous + (value - previous) * smoothing
    return previous


def change_name(new_surname, new_name):
    """Function to change the person's name."""
    global person_surname, person_name
    person_surname = new_surname
    person_name = new_name


def get_face(image):
    """Captures and saves the center of the face."""
    # Crop the image on the center box
    face_image = image[
        (WINDOW_HEIGHT - FIXED_FACE_SIZE) // 2 : (WINDOW_HEIGHT + FIXED_FACE_SIZE) // 2,
        (WINDOW_WIDTH - FIXED_FACE_SIZE) // 2 : (WINDOW_WIDTH + FIXED_FACE_SIZE) // 2,
    ]

    return face_image


def save_face(face_image):
    """Saves the face image to the specified filename."""
    global label
    # Save the face image
    face_dir = "dataset/captured_faces"
    os.makedirs(face_dir, exist_ok=True)
    # Generate a random number
    random_number = np.random.randint(1000)
    face_filename = os.path.join(face_dir, f"{label}_{random_number}.png")
    cv2.imwrite(face_filename, face_image)
    print(f"Face saved as {face_filename}")


def main():
    global prev_translate_x, prev_translate_y, prev_scale_factor, rt_prev_translate_x, rt_prev_translate_y, rt_prev_scale_factor, prev_bbox

    # Start video capture
    cap = cv2.VideoCapture(0)

    with mp_face_detection.FaceDetection(
        min_detection_confidence=0.5
    ) as face_detection, mp_face_mesh.FaceMesh(
        min_detection_confidence=0.5
    ) as face_mesh:
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                continue

            # Flip the image horizontally for a later selfie-view display
            image = cv2.flip(image, 1)

            # Convert the image to RGB format
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            face_detection_results = face_detection.process(image_rgb)
            face_mesh_results = face_mesh.process(image_rgb)

            # Get the height and width of the original image
            h, w, _ = image.shape

            # Initialize translation variables
            translate_x = 0
            translate_y = 0

            # If faces are detected
            if (
                face_detection_results.detections
                and face_mesh_results.multi_face_landmarks
            ):
                for detection, landmarks in zip(
                    face_detection_results.detections,
                    face_mesh_results.multi_face_landmarks,
                ):
                    # Get the bounding box
                    bboxC = detection.location_data.relative_bounding_box
                    x_min = int(bboxC.xmin * w)
                    y_min = int(bboxC.ymin * h)
                    width = int(bboxC.width * w)
                    height = int(bboxC.height * h)

                    # Extract the left and right eye positions (landmarks 33 and 263 for the eyes)
                    left_eye = landmarks.landmark[33]
                    right_eye = landmarks.landmark[263]

                    # Get eye coordinates in pixel space
                    left_eye_x = int(left_eye.x * w)
                    left_eye_y = int(left_eye.y * h)
                    right_eye_x = int(right_eye.x * w)
                    right_eye_y = int(right_eye.y * h)

                    # Average the positions to get the midpoint of the eyes
                    eye_center_x = (left_eye_x + right_eye_x) // 2
                    eye_center_y = (left_eye_y + right_eye_y) // 2

                    # Calculate the translation for stabilization using the eye center
                    translate_x = (WINDOW_WIDTH // 2) - eye_center_x
                    translate_y = (
                        (WINDOW_HEIGHT // 2) - eye_center_y - EYE_TO_FACE_CENTER_OFFSET
                    )

                    # Apply threshold and smooth the translations using a moving average
                    prev_translate_x = smooth_coordinates(
                        apply_threshold(
                            translate_x,
                            prev_translate_x,
                            THRESHOLD_POSITION,
                            SMOOTHING_FACTOR,
                        ),
                        prev_translate_x,
                        SMOOTHING_FACTOR,
                    )
                    prev_translate_y = smooth_coordinates(
                        apply_threshold(
                            translate_y,
                            prev_translate_y,
                            THRESHOLD_POSITION,
                            SMOOTHING_FACTOR,
                        ),
                        prev_translate_y,
                        SMOOTHING_FACTOR,
                    )
                    rt_prev_translate_x = smooth_coordinates(
                        apply_threshold(
                            translate_x, rt_prev_translate_x, THRESHOLD_POSITION, 0.6
                        ),
                        rt_prev_translate_x,
                        0.8,
                    )
                    rt_prev_translate_y = smooth_coordinates(
                        apply_threshold(
                            translate_y, rt_prev_translate_y, THRESHOLD_POSITION, 0.6
                        ),
                        rt_prev_translate_y,
                        0.8,
                    )

                    # Calculate the scale factor based on the bounding box height
                    face_distance = (
                        height  # Using the height of the bounding box for distance
                    )
                    new_scale_factor = FIXED_FACE_SIZE / (
                        face_distance + 1
                    )  # Avoid division by zero

                    # Apply threshold and smooth the scale factor
                    prev_scale_factor = smooth_coordinates(
                        apply_threshold(
                            new_scale_factor,
                            prev_scale_factor,
                            THRESHOLD_SCALE,
                            SMOOTHING_SCALE,
                        ),
                        prev_scale_factor,
                        SMOOTHING_SCALE,
                    )
                    rt_prev_scale_factor = smooth_coordinates(
                        apply_threshold(
                            new_scale_factor, rt_prev_scale_factor, 0.15, 0.8
                        ),
                        rt_prev_scale_factor,
                        0.8,
                    )

                    # Center the resized image in the output window without black borders
                    def transform(tx, ty, sf):
                        # Create a transformation matrix for translation
                        M = np.float32([[1, 0, tx], [0, 1, ty]])

                        # Apply translation and resize the image
                        translated_image = cv2.warpAffine(
                            image, M, (WINDOW_WIDTH, WINDOW_HEIGHT)
                        )

                        # Resize the translated image based on the smoothed scale factor
                        resized_image = cv2.resize(translated_image, None, fx=sf, fy=sf)

                        resized_height, resized_width = resized_image.shape[:2]
                        if (
                            resized_height < WINDOW_HEIGHT
                            or resized_width < WINDOW_WIDTH
                        ):
                            im = np.zeros(
                                (WINDOW_HEIGHT, WINDOW_WIDTH, 3), dtype=np.uint8
                            )
                            y_offset = (WINDOW_HEIGHT - resized_height) // 2
                            x_offset = (WINDOW_WIDTH - resized_width) // 2
                            im[
                                y_offset : y_offset + resized_height,
                                x_offset : x_offset + resized_width,
                            ] = resized_image
                        else:
                            # Crop to fit in the window
                            im = resized_image[
                                (resized_height - WINDOW_HEIGHT)
                                // 2 : (resized_height + WINDOW_HEIGHT)
                                // 2,
                                (resized_width - WINDOW_WIDTH)
                                // 2 : (resized_width + WINDOW_WIDTH)
                                // 2,
                            ]
                        return im

                    canvas = transform(
                        prev_translate_x, prev_translate_y, prev_scale_factor
                    )
                    rt_canvas = transform(
                        rt_prev_translate_x, rt_prev_translate_y, rt_prev_scale_factor
                    )
                    cropped_face = get_face(rt_canvas)
                    cropped_face = cv2.resize(cropped_face, (300, 300))
                    canvas[-cropped_face.shape[0] :, -cropped_face.shape[1] :] = (
                        cropped_face
                    )

                    # Save the captured face image
                    if cv2.waitKey(1) & 0xFF == ord("s"):  # Press 's' to save the face
                        rt_canvas = transform(
                            rt_prev_translate_x,
                            rt_prev_translate_y,
                            rt_prev_scale_factor,
                        )
                        face_image = get_face(rt_canvas)
                        save_face(face_image)

                    # Predict the person's name
                    label = predict(get_face(rt_canvas))
                    person_surname = face_dict[label]

                    # Draw the person's name on the canvas
                    cv2.putText(
                        canvas,
                        person_surname,
                        (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.5,
                        (255, 255, 255),
                        2,
                        cv2.LINE_AA,
                    )
                    cv2.putText(
                        canvas,
                        person_name,
                        (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (255, 255, 255),
                        2,
                        cv2.LINE_AA,
                    )

                    # Display the stabilized image on a fixed-size window
                    cv2.imshow("Stabilized Face", canvas)
                    break  # Draw only the first detected face

            else:
                # If no face is detected, display the original image on the fixed-size window
                resized_image = cv2.resize(image, (WINDOW_WIDTH, WINDOW_HEIGHT))
                cv2.imshow("Stabilized Face", resized_image)

            # Exit on pressing 'q'
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
