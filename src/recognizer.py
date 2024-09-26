import cv2
import mediapipe as mp
import numpy as np

# Initialize Mediapipe Face Detection and Face Mesh modules
mp_face_detection = mp.solutions.face_detection
mp_face_mesh = mp.solutions.face_mesh

# Parameters
WINDOW_WIDTH = 640  # Fixed width of the output window
WINDOW_HEIGHT = 360  # Fixed height of the output window
FIXED_FACE_SIZE = 300  # Desired face size in pixels
SMOOTHING_FACTOR = 0.5  # Smoothing factor for translation
SMOOTHING_SCALE = 0.3  # Smoothing factor for scaling
THRESHOLD_POSITION = 20  # Threshold for position changes
THRESHOLD_SCALE = 0.08  # Threshold for scale changes

# Initialize previous translation values and size
prev_translate_x = 0
prev_translate_y = 0
prev_scale_factor = 1.0  # Initial scale factor

prev_bbox = None


def smooth_coordinates(current, previous, smoothing):
    """Smooths the coordinates using a moving average."""
    return previous + (current - previous) * smoothing


def apply_threshold(value, previous, threshold, smoothing):
    """Applies a threshold to ignore small changes."""
    if abs(value - previous) > threshold:
        return previous + (value - previous) * smoothing
    return previous


def main():
    global prev_translate_x, prev_translate_y, prev_scale_factor, prev_bbox

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
            scale_factor = 1.0

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

                    # Smooth the bounding box coordinates
                    current_bbox = (x_min, y_min, width, height)
                    if prev_bbox is None:
                        prev_bbox = current_bbox
                    else:
                        smoothed_x_min = smooth_coordinates(
                            apply_threshold(
                                x_min,
                                prev_bbox[0],
                                THRESHOLD_POSITION,
                                SMOOTHING_FACTOR,
                            ),
                            prev_bbox[0],
                            SMOOTHING_FACTOR,
                        )
                        smoothed_y_min = smooth_coordinates(
                            apply_threshold(
                                y_min,
                                prev_bbox[1],
                                THRESHOLD_POSITION,
                                SMOOTHING_FACTOR,
                            ),
                            prev_bbox[1],
                            SMOOTHING_FACTOR,
                        )
                        smoothed_width = smooth_coordinates(
                            apply_threshold(
                                width,
                                prev_bbox[2],
                                THRESHOLD_POSITION,
                                SMOOTHING_SCALE,
                            ),
                            prev_bbox[2],
                            SMOOTHING_SCALE,
                        )
                        smoothed_height = smooth_coordinates(
                            apply_threshold(
                                height,
                                prev_bbox[3],
                                THRESHOLD_POSITION,
                                SMOOTHING_SCALE,
                            ),
                            prev_bbox[3],
                            SMOOTHING_SCALE,
                        )
                        prev_bbox = (
                            smoothed_x_min,
                            smoothed_y_min,
                            smoothed_width,
                            smoothed_height,
                        )

                    # Draw the smoothed bounding box around the face
                    cv2.rectangle(
                        image,
                        (int(prev_bbox[0]), int(prev_bbox[1])),
                        (
                            int(prev_bbox[0] + prev_bbox[2]),
                            int(prev_bbox[1] + prev_bbox[3]),
                        ),
                        (0, 255, 0),
                        2,
                    )

                    # Calculate the center of the bounding box
                    bbox_center_x = x_min + width // 2
                    bbox_center_y = y_min + height // 2

                    # Calculate translation for stabilization using the center of the bounding box
                    translate_x = (WINDOW_WIDTH // 2) - bbox_center_x
                    translate_y = (WINDOW_HEIGHT // 2) - bbox_center_y

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

                    # Create a transformation matrix for translation
                    M = np.float32([[1, 0, prev_translate_x], [0, 1, prev_translate_y]])

                    # Apply translation and resize the image
                    translated_image = cv2.warpAffine(
                        image, M, (WINDOW_WIDTH, WINDOW_HEIGHT)
                    )

                    # Resize the translated image based on the smoothed scale factor
                    resized_image = cv2.resize(
                        translated_image,
                        None,
                        fx=prev_scale_factor,
                        fy=prev_scale_factor,
                    )

                    # Center the resized image in the output window without black borders
                    resized_height, resized_width = resized_image.shape[:2]
                    if resized_height < WINDOW_HEIGHT or resized_width < WINDOW_WIDTH:
                        canvas = np.zeros(
                            (WINDOW_HEIGHT, WINDOW_WIDTH, 3), dtype=np.uint8
                        )
                        y_offset = (WINDOW_HEIGHT - resized_height) // 2
                        x_offset = (WINDOW_WIDTH - resized_width) // 2
                        canvas[
                            y_offset : y_offset + resized_height,
                            x_offset : x_offset + resized_width,
                        ] = resized_image
                    else:
                        # Crop to fit in the window
                        canvas = resized_image[
                            (resized_height - WINDOW_HEIGHT)
                            // 2 : (resized_height + WINDOW_HEIGHT)
                            // 2,
                            (resized_width - WINDOW_WIDTH)
                            // 2 : (resized_width + WINDOW_WIDTH)
                            // 2,
                        ]

                    # Display the stabilized image on a fixed-size window
                    cv2.imshow("Stabilized Face", canvas)
                    break  # Draw only the first detected face

            else:
                # If no face is detected, display the original image on the fixed-size window
                resized_image = cv2.resize(image, (WINDOW_WIDTH, WINDOW_HEIGHT))
                cv2.imshow("Stabilized Face", resized_image)

            # Exit on pressing 'q'
            if cv2.waitKey(5) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
