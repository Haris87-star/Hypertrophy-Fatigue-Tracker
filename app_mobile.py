import os
# Suppress TensorFlow and absl warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['GLOG_minloglevel'] = '2'

import absl.logging
absl.logging.set_verbosity(absl.logging.ERROR)

import cv2
import mediapipe as mp
import time
import numpy as np
import csv
from datetime import datetime

from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import PoseLandmarker, PoseLandmarkerOptions, PoseLandmark

# Video capture: webcam or replace with DroidCam IP stream
WINDOW_NAME = 'Fatigue & Form Detector - Mobile Vision'
cap = cv2.VideoCapture('http://10.203.212.130:4747/video')  # Use 0 for webcam, or IP URL for DroidCam
print(f"Camera opened: {cap.isOpened()}")
if not cap.isOpened():
    print('Failed to open video stream. Check your DroidCam URL and network, and ensure both devices are on the same Wi-Fi.')
    exit(1)

# Fatigue and rep counting variables
right_previous_y = None
right_previous_time = None
right_velocity = 0
right_rep_count = 0
right_stage = 'unknown'
right_rep_velocities = []
right_baseline_velocity = 0
right_status = 'Baseline Calibration'

left_previous_y = None
left_previous_time = None
left_velocity = 0
left_rep_count = 0
left_stage = 'unknown'
left_rep_velocities = []
left_baseline_velocity = 0
left_status = 'Baseline Calibration'

fatigue_threshold = 0.80  # 80% of baseline velocity
form_status = 'Unknown'

# Calibration parameters - adjust these based on your setup
CONFIDENCE_THRESHOLD = 0.5  # Minimum confidence for pose landmarks
SPEED_SMOOTHING_WINDOW = 5  # Average speed over last N frames
right_speed_history = []  # Track right hand speeds for smoothing
left_speed_history = []  # Track left hand speeds for smoothing
right_last_rep_time = 0
left_last_rep_time = 0
MIN_REP_INTERVAL = 0.5  # Minimum time between reps (prevents double counting)
NOISE_THRESHOLD = 0.02  # Ignore tiny wrist jitter
UP_ANGLE_THRESHOLD = 80  # Top of curl / upward motion
DOWN_ANGLE_THRESHOLD = 140  # Bottom of curl / extended arm
MIN_REP_AMPLITUDE = 0.08  # Minimum normalized wrist travel to count a rep

right_wrist_y_history = []
left_wrist_y_history = []
right_angle_history = []
left_angle_history = []
right_rep_ready = False
left_rep_ready = False

filename = f"workout_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
with open(filename, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Rep Number', 'Speed', 'Target Speed', 'Status', 'Form', 'Arm'])

print(f"Starting Mobile Version... Data will be saved to {filename}")
print("Press 'q' to exit.")

cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
# Keep standard window mode to ensure the display appears reliably

# Download model if not present
# Get the directory where this script is located
script_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(script_dir, 'pose_landmarker_lite.task')

if not os.path.exists(model_path):
    print("\n" + "="*70)
    print("MODEL FILE NOT FOUND")
    print("="*70)
    print(f"\nRequired file: {model_path}")
    print(f"\nPlease download 'pose_landmarker_lite.task' from:")
    print("  https://github.com/google-mediapipe/mediapipe/releases")
    print(f"\nDetailed instructions available in: DOWNLOAD_MODEL.md")
    print("="*70 + "\n")
    exit(1)

# Initialize Pose Estimator with tasks API
options = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=vision.RunningMode.VIDEO
)
pose_detector = PoseLandmarker.create_from_options(options)
print("Pose detector initialized successfully.")


def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    if angle > 180.0:
        angle = 360 - angle
    return angle


def normalize_speed(delta_y, delta_t):
    if delta_t <= 0:
        return 0
    return abs(delta_y / delta_t) * 100


def evaluate_form(elbow_angle):
    if elbow_angle > 160:
        return 'Extend more at top'
    if elbow_angle < 45:
        return 'Do not overflex shoulder'
    return 'Good Form'

def process_arm(
    wrist, elbow, shoulder, 
    prev_y, prev_time, 
    state, rep_count, velocities, baseline, ready_flag,
    speed_history, wrist_history, angle_history,
    last_rep_time, current_time, filename, arm_name
):
    # Combine all the angle/speed calculation logic here
    # Return updated values as a tuple
    pass


# Main processing loop
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print('Failed to capture frame. Check the camera or DroidCam connection.')
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    current_time = time.time()
    results = pose_detector.detect_for_video(mp_image, int(current_time * 1000))
    display_frame = frame.copy()
    cv2.rectangle(display_frame, (10, 10), (330, 390), (0, 0, 0), -1)

    if results.pose_landmarks:
        landmarks = results.pose_landmarks[0]
        
        # Get the landmarks we need for both arms
        right_wrist = landmarks[PoseLandmark.RIGHT_WRIST.value]
        right_elbow = landmarks[PoseLandmark.RIGHT_ELBOW.value]
        right_shoulder = landmarks[PoseLandmark.RIGHT_SHOULDER.value]
        left_wrist = landmarks[PoseLandmark.LEFT_WRIST.value]
        left_elbow = landmarks[PoseLandmark.LEFT_ELBOW.value]
        left_shoulder = landmarks[PoseLandmark.LEFT_SHOULDER.value]

        right_valid = right_wrist.presence >= CONFIDENCE_THRESHOLD and right_elbow.presence >= CONFIDENCE_THRESHOLD and right_shoulder.presence >= CONFIDENCE_THRESHOLD
        left_valid = left_wrist.presence >= CONFIDENCE_THRESHOLD and left_elbow.presence >= CONFIDENCE_THRESHOLD and left_shoulder.presence >= CONFIDENCE_THRESHOLD

        if not right_valid and not left_valid:
            right_wrist_y_history.clear()
            left_wrist_y_history.clear()
            right_angle_history.clear()
            left_angle_history.clear()
            right_previous_y = None
            right_previous_time = None
            left_previous_y = None
            left_previous_time = None
            right_velocity = 0
            left_velocity = 0
            cv2.putText(display_frame, 'Low confidence - reposition', (20, 420), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            cv2.imshow('Fatigue & Form Detector - Mobile Vision', display_frame)
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
            continue

        if right_valid:
            right_elbow_angle = calculate_angle(
                [right_shoulder.x, right_shoulder.y],
                [right_elbow.x, right_elbow.y],
                [right_wrist.x, right_wrist.y]
            )
            right_form_status = evaluate_form(right_elbow_angle)
            right_wrist_y_history.append(right_wrist.y)
            if len(right_wrist_y_history) > 10:
                right_wrist_y_history.pop(0)
            right_wrist_y_smooth = sum(right_wrist_y_history) / len(right_wrist_y_history)

            right_angle_history.append(right_elbow_angle)
            if len(right_angle_history) > 10:
                right_angle_history.pop(0)
            right_smoothed_angle = sum(right_angle_history) / len(right_angle_history)

            right_movement_amplitude = max(right_wrist_y_history) - min(right_wrist_y_history) if len(right_wrist_y_history) > 1 else 0

            if right_previous_y is not None and right_previous_time is not None:
                right_delta_y = right_previous_y - right_wrist_y_smooth
                right_delta_t = current_time - right_previous_time
                if right_delta_t > 0:
                    right_current_speed = normalize_speed(right_delta_y, right_delta_t)
                    if abs(right_delta_y) < NOISE_THRESHOLD:
                        right_current_speed = 0
                    right_speed_history.append(right_current_speed)
                    if len(right_speed_history) > SPEED_SMOOTHING_WINDOW:
                        right_speed_history.pop(0)
                    right_smoothed_speed = sum(right_speed_history) / len(right_speed_history) if right_speed_history else 0
                    right_velocity = int(right_smoothed_speed)
                    if right_stage == 'unknown':
                        if right_smoothed_angle > DOWN_ANGLE_THRESHOLD:
                            right_stage = 'down'
                    elif right_stage == 'down':
                        if right_smoothed_angle < UP_ANGLE_THRESHOLD and right_movement_amplitude > MIN_REP_AMPLITUDE:
                            right_stage = 'up'
                            right_rep_ready = True
                    elif right_stage == 'up':
                        if right_smoothed_angle > DOWN_ANGLE_THRESHOLD and right_rep_ready:
                            if current_time - right_last_rep_time > MIN_REP_INTERVAL:
                                right_rep_count += 1
                                right_last_rep_time = current_time
                                right_rep_velocities.append(right_current_speed)
                                if len(right_rep_velocities) == 3:
                                    right_baseline_velocity = sum(right_rep_velocities) / len(right_rep_velocities)
                                right_target_speed = int(right_baseline_velocity * fatigue_threshold) if right_rep_count >= 3 else 0
                                right_status = 'Baseline Calibration' if right_rep_count < 3 else ('Fatigued' if right_current_speed < right_target_speed else 'Optimal')
                                with open(filename, mode='a', newline='') as file:
                                    writer = csv.writer(file)
                                    writer.writerow([right_rep_count, int(right_current_speed), right_target_speed, right_status, right_form_status, 'Right'])
                            right_rep_ready = False
                            right_stage = 'down'
            else:
                right_delta_y = 0

            right_previous_y = right_wrist_y_smooth
            right_previous_time = current_time
            right_elbow_angle = right_smoothed_angle
            right_form_status = evaluate_form(right_elbow_angle)

        else:
            right_wrist_y_history.clear()
            right_angle_history.clear()
            right_previous_y = None
            right_previous_time = None
            right_velocity = 0
            right_form_status = '--'
            right_elbow_angle = 0
            right_status = 'Right not visible'

        if left_valid:
            left_elbow_angle = calculate_angle(
                [left_shoulder.x, left_shoulder.y],
                [left_elbow.x, left_elbow.y],
                [left_wrist.x, left_wrist.y]
            )
            left_form_status = evaluate_form(left_elbow_angle)
            left_wrist_y_history.append(left_wrist.y)
            if len(left_wrist_y_history) > 10:
                left_wrist_y_history.pop(0)
            left_wrist_y_smooth = sum(left_wrist_y_history) / len(left_wrist_y_history)

            left_angle_history.append(left_elbow_angle)
            if len(left_angle_history) > 10:
                left_angle_history.pop(0)
            left_smoothed_angle = sum(left_angle_history) / len(left_angle_history)

            left_movement_amplitude = max(left_wrist_y_history) - min(left_wrist_y_history) if len(left_wrist_y_history) > 1 else 0

            if left_previous_y is not None and left_previous_time is not None:
                left_delta_y = left_previous_y - left_wrist_y_smooth
                left_delta_t = current_time - left_previous_time
                if left_delta_t > 0:
                    left_current_speed = normalize_speed(left_delta_y, left_delta_t)
                    if abs(left_delta_y) < NOISE_THRESHOLD:
                        left_current_speed = 0
                    left_speed_history.append(left_current_speed)
                    if len(left_speed_history) > SPEED_SMOOTHING_WINDOW:
                        left_speed_history.pop(0)
                    left_smoothed_speed = sum(left_speed_history) / len(left_speed_history) if left_speed_history else 0
                    left_velocity = int(left_smoothed_speed)
                    if left_stage == 'unknown':
                        if left_smoothed_angle > DOWN_ANGLE_THRESHOLD:
                            left_stage = 'down'
                    elif left_stage == 'down':
                        if left_smoothed_angle < UP_ANGLE_THRESHOLD and left_movement_amplitude > MIN_REP_AMPLITUDE:
                            left_stage = 'up'
                            left_rep_ready = True
                    elif left_stage == 'up':
                        if left_smoothed_angle > DOWN_ANGLE_THRESHOLD and left_rep_ready:
                            if current_time - left_last_rep_time > MIN_REP_INTERVAL:
                                left_rep_count += 1
                                left_last_rep_time = current_time
                                left_rep_velocities.append(left_current_speed)
                                if len(left_rep_velocities) == 3:
                                    left_baseline_velocity = sum(left_rep_velocities) / len(left_rep_velocities)
                                left_target_speed = int(left_baseline_velocity * fatigue_threshold) if left_rep_count >= 3 else 0
                                left_status = 'Baseline Calibration' if left_rep_count < 3 else ('Fatigued' if left_current_speed < left_target_speed else 'Optimal')
                                with open(filename, mode='a', newline='') as file:
                                    writer = csv.writer(file)
                                    writer.writerow([left_rep_count, int(left_current_speed), left_target_speed, left_status, left_form_status, 'Left'])
                            left_rep_ready = False
                            left_stage = 'down'
            else:
                left_delta_y = 0

            left_previous_y = left_wrist_y_smooth
            left_previous_time = current_time
            left_elbow_angle = left_smoothed_angle
            left_form_status = evaluate_form(left_elbow_angle)

        else:
            left_wrist_y_history.clear()
            left_angle_history.clear()
            left_previous_y = None
            left_previous_time = None
            left_velocity = 0
            left_form_status = '--'
            left_elbow_angle = 0
            left_status = 'Left not visible'

        line_x = 25
        line_y = 45
        line_height = 36

        cv2.putText(display_frame, 'HYPERTROPHY TRACKER', (line_x, line_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220, 220, 220), 2)
        line_y += line_height
        cv2.putText(display_frame, f'Right Reps: {right_rep_count}', (line_x, line_y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        line_y += line_height
        cv2.putText(display_frame, f'Right Speed: {right_velocity}', (line_x, line_y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        line_y += line_height
        cv2.putText(display_frame, f'Right Elbow: {int(right_elbow_angle) if right_valid else 0}', (line_x, line_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        line_y += line_height
        cv2.putText(display_frame, f'Right Status: {right_status}', (line_x, line_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        line_y += line_height
        cv2.putText(display_frame, f'Left Reps: {left_rep_count}', (line_x, line_y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        line_y += line_height
        cv2.putText(display_frame, f'Left Speed: {left_velocity}', (line_x, line_y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        line_y += line_height
        cv2.putText(display_frame, f'Left Elbow: {int(left_elbow_angle) if left_valid else 0}', (line_x, line_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        line_y += line_height
        cv2.putText(display_frame, f'Left Status: {left_status}', (line_x, line_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        line_y += line_height * 2

        if right_valid or left_valid:
            cv2.putText(display_frame, 'Tracking active', (line_x, line_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            cv2.putText(display_frame, 'No strong hand detected', (line_x, line_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    else:
        right_wrist_y_history.clear()
        left_wrist_y_history.clear()
        right_angle_history.clear()
        left_angle_history.clear()
        right_previous_y = None
        right_previous_time = None
        left_previous_y = None
        left_previous_time = None
        right_velocity = 0
        left_velocity = 0
        line_x = 25
        line_y = 45
        line_height = 36
        cv2.putText(display_frame, 'HYPERTROPHY TRACKER', (line_x, line_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220, 220, 220), 2)
        line_y += line_height
        cv2.putText(display_frame, 'Right Reps: 0', (line_x, line_y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        line_y += line_height
        cv2.putText(display_frame, 'Right Speed: 0', (line_x, line_y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        line_y += line_height
        cv2.putText(display_frame, 'Right Elbow: --', (line_x, line_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        line_y += line_height
        cv2.putText(display_frame, 'Right Status: Waiting', (line_x, line_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        line_y += line_height
        cv2.putText(display_frame, 'Left Reps: 0', (line_x, line_y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        line_y += line_height
        cv2.putText(display_frame, 'Left Speed: 0', (line_x, line_y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        line_y += line_height
        cv2.putText(display_frame, 'Left Elbow: --', (line_x, line_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        line_y += line_height
        cv2.putText(display_frame, 'Left Status: Waiting', (line_x, line_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        line_y += line_height
        cv2.putText(display_frame, 'No pose detected. Reposition camera.', (line_x, line_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    cv2.imshow(WINDOW_NAME, display_frame)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
# Properly close the pose detector
try:
    pose_detector.close()
except Exception as e:
    print(f"Note: {e}")

print(f"\nWorkout session saved to: {filename}")
print("Thank you for using Hypertrophy Tracker!")