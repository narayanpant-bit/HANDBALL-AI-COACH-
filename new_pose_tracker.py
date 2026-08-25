import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import winsound
import threading

def play_sound(frequency, duration=150):
    threading.Thread(target=winsound.Beep, args=(frequency, duration), daemon=True).start()

def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    return 360.0 - angle if angle > 180.0 else angle

# Setup Tasks API Landmarker
base_options = python.BaseOptions(model_asset_path='pose_landmarker_heavy.task')
options = vision.PoseLandmarkerOptions(
    base_options=base_options,
    output_segmentation_masks=False)
detector = vision.PoseLandmarker.create_from_options(options)

cap = cv2.VideoCapture(0)
shot_count = 0
stage = None
last_sound_played = None

print("New MediaPipe Tasks Pose Tracker Started!")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    detection_result = detector.detect(mp_image)

    if detection_result.pose_landmarks:
        landmarks = detection_result.pose_landmarks[0]

        # Right Arm: Shoulder (12), Elbow (14), Wrist (16)
        shoulder = [landmarks[12].x * w, landmarks[12].y * h]
        elbow = [landmarks[14].x * w, landmarks[14].y * h]
        wrist = [landmarks[16].x * w, landmarks[16].y * h]

        angle = calculate_angle(shoulder, elbow, wrist)

        if angle < 90:
            stage = "READY"
        if angle > 130 and stage == "READY":
            stage = "RELEASED"
            shot_count += 1

        if 85 <= angle <= 125:
            feedback, color = "PERFECT FORM!", (0, 255, 0)
            if last_sound_played != "PERFECT":
                play_sound(900, 120)
                last_sound_played = "PERFECT"
        elif angle < 85:
            feedback, color = "ARM TOO BENT!", (0, 0, 255)
            if last_sound_played != "BENT":
                play_sound(350, 200)
                last_sound_played = "BENT"
        else:
            feedback, color = "EXTEND CONTROLLED!", (0, 165, 255)
            if last_sound_played != "EXTEND":
                play_sound(450, 200)
                last_sound_played = "EXTEND"

        cv2.circle(frame, (int(elbow[0]), int(elbow[1])), 8, (0, 255, 255), -1)
        cv2.putText(frame, f"{int(angle)} deg", (int(elbow[0])+15, int(elbow[1])), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        cv2.rectangle(frame, (10, 10), (420, 85), (0, 0, 0), -1)
        cv2.putText(frame, f"SHOTS: {shot_count}", (20, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
        cv2.putText(frame, f"Status: {feedback}", (20, 70), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    cv2.imshow('Handball AI - New Tasks API', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
