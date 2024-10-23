import cv2
import dlib
from datetime import timedelta
import numpy as np

# Load dlib face detector
detector_dlib = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Load OpenCV Haar Cascade face detector
haar_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

cap = cv2.VideoCapture("howard_video.mp4")

# Get FPS
fps = cap.get(cv2.CAP_PROP_FPS)
process_interval = 0.5  # Process every 0.5 seconds
frame_interval = int(fps * process_interval)

# Variables for timestamp recording
not_talking_start_time = None
talking = False
no_face_duration = 0
max_no_face_duration = 4.0  # Set to 4 seconds to maintain a segment during screen transitions
min_talking_duration = 3.0
last_lip_movement_time = None  # The time when the last lip movement was detected
lip_movement_timeout = 5.0  # End the timestamp if no lip movement for 5 seconds (increased to 5 seconds)
last_valid_time = 0.0  # The time of the last valid frame
prev_frame_gray = None  # Store the previous frame

def detect_lip_movement(landmarks):
    lip_points = []
    for n in range(48, 68):
        x = landmarks.part(n).x
        y = landmarks.part(n).y
        lip_points.append((x, y))
    total_movement = 0
    for (x1, y1), (x2, y2) in zip(lip_points[:-1], lip_points[1:]):
        total_movement += abs(x1 - x2) + abs(y1 - y2)
    return total_movement

def is_frontal_face(landmarks):
    left_eye_x = landmarks.part(36).x
    right_eye_x = landmarks.part(45).x
    nose_x = landmarks.part(30).x
    eye_distance = abs(left_eye_x - right_eye_x)
    if eye_distance == 0:
        return False
    nose_eye_diff = abs(nose_x - (left_eye_x + right_eye_x) / 2)
    if nose_eye_diff < eye_distance * 0.1:  # If it's a frontal face
        return True
    else:  # If it's a side face
        return False

def is_scene_change(current_frame, prev_frame, threshold=30):
    if prev_frame is None or current_frame is None:
        return False
    diff = cv2.absdiff(current_frame, prev_frame)
    non_zero_count = np.count_nonzero(diff)
    total_pixels = diff.size
    change_percentage = (non_zero_count / total_pixels) * 100
    return change_percentage > threshold

def seconds_to_hhmmss(seconds):
    return str(timedelta(seconds=int(seconds)))

frame_count = 0

with open("output.txt", "w") as f:
        
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_interval == 0:
            current_time = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            last_valid_time = current_time  # Save the time of the last valid frame to use when the video ends

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Detect scene change
            scene_changed = is_scene_change(gray, prev_frame_gray)
            prev_frame_gray = gray.copy()

            # Face detection using dlib
            faces_dlib = detector_dlib(gray)

            # Face detection using OpenCV Haar Cascade
            faces_haar = haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

            faces_detected = []

            # Add dlib detection results to the list
            for face in faces_dlib:
                faces_detected.append(face)

            # Convert Haar Cascade detection results to dlib's rectangle format and add to the list
            for (x, y, w, h) in faces_haar:
                face = dlib.rectangle(int(x), int(y), int(x + w), int(y + h))
                faces_detected.append(face)

            if len(faces_detected) > 0:
                no_face_duration = 0  # Reset since faces were detected
                for face in faces_detected:
                    landmarks = predictor(gray, face)
                    lip_movement = detect_lip_movement(landmarks)

                    is_frontal = is_frontal_face(landmarks)

                    # Apply different lip movement thresholds based on whether it's a frontal or side face
                    if is_frontal:
                        lip_threshold = 400  # Threshold for frontal faces
                    else:
                        lip_threshold = 100  # Threshold for side faces

                    print(f"Lip movement: {lip_movement}, Frontal face: {is_frontal} at {seconds_to_hhmmss(current_time)}")

                    if lip_movement > lip_threshold:
                        if not talking:
                            talking = True
                            if not_talking_start_time is not None:
                                not_talking_end_time = current_time
                                print(f"Non-talking segment: {seconds_to_hhmmss(not_talking_start_time)} -> {seconds_to_hhmmss(not_talking_end_time)}")
                                f.write(f"{seconds_to_hhmmss(not_talking_start_time)} -> {seconds_to_hhmmss(not_talking_end_time)}\n")
                                not_talking_start_time = None  # Reset
                        last_lip_movement_time = current_time  # Update the last lip movement time
                    else:
                        # No processing if there's no lip movement
                        pass

            else:
                no_face_duration += process_interval
                print(f"Current no_face_duration: {no_face_duration} last_lip_movement_time: {last_lip_movement_time} talking:{talking} at {seconds_to_hhmmss(current_time)}")

                # If no face is detected, determine whether to end the timestamp based on scene change
                if talking:
                    # End the timestamp if no face is detected for too long
                    if no_face_duration >= max_no_face_duration:
                        talking = False
                        not_talking_start_time = current_time
                else:
                    if not_talking_start_time is None:
                        not_talking_start_time = current_time  # Record the start of the non-talking segment

        frame_count += 1

    # When the video ends, if there's still a non-talking segment left, output the final timestamp
    if not_talking_start_time is not None:
        not_talking_end_time = last_valid_time
        print(f"Non-talking segment: {seconds_to_hhmmss(not_talking_start_time)} -> {seconds_to_hhmmss(not_talking_end_time)}")
        f.write(f"{seconds_to_hhmmss(not_talking_start_time)} -> {seconds_to_hhmmss(not_talking_end_time)}\n")

cap.release()
cv2.destroyAllWindows()
