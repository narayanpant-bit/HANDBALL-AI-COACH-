import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import winsound
import threading

def play_sound(frequency, duration=100):
    threading.Thread(target=winsound.Beep, args=(frequency, duration), daemon=True).start()

# Tasks API Setup
base_options = python.BaseOptions(model_asset_path='pose_landmarker_heavy.task')
options = vision.PoseLandmarkerOptions(
    base_options=base_options,
    output_segmentation_masks=False)
detector = vision.PoseLandmarker.create_from_options(options)

cap = cv2.VideoCapture(0)
last_action = ""

print("Keeper All-Actions AI Tracker Started!")

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

        # Extract Key Coordinates
        l_shoulder = [landmarks[11].x * w, landmarks[11].y * h]
        r_shoulder = [landmarks[12].x * w, landmarks[12].y * h]
        l_wrist = [landmarks[15].x * w, landmarks[15].y * h]
        r_wrist = [landmarks[16].x * w, landmarks[16].y * h]
        l_hip = [landmarks[23].x * w, landmarks[23].y * h]
        r_hip = [landmarks[24].x * w, landmarks[24].y * h]
        l_ankle = [landmarks[27].x * w, landmarks[27].y * h]
        r_ankle = [landmarks[28].x * w, landmarks[28].y * h]

        action = "NEUTRAL / READY"
        color = (255, 255, 255)

        # 1. STAR SAVE (JUMP SPREAD) - Hands high & feet wide apart
        leg_span = np.abs(l_ankle[0] - r_ankle[0])
        arm_span = np.abs(l_wrist[0] - r_wrist[0])
        
        if (l_wrist[1] < l_shoulder[1] and r_wrist[1] < r_shoulder[1]) and leg_span > 200:
            action = "STAR SAVE (FULL COVERAGE)"
            color = (0, 255, 255)
            if last_action != "STAR":
                play_sound(1200, 150)
                last_action = "STAR"

        # 2. HIGH BLOCK - Hands above shoulders
        elif l_wrist[1] < l_shoulder[1] and r_wrist[1] < r_shoulder[1]:
            action = "HIGH BLOCK SAVE"
            color = (0, 255, 0)
            if last_action != "HIGH":
                play_sound(900, 100)
                last_action = "HIGH"

        # 3. LOW SAVE - Either wrist below hip level
        elif l_wrist[1] > l_hip[1] or r_wrist[1] > r_hip[1]:
            action = "LOW BLOCK / LEG SAVE"
            color = (0, 0, 255)
            if last_action != "LOW":
                play_sound(400, 100)
                last_action = "LOW"

        # 4. WING SIDE SAVE - Extended arm to far left or right
        elif l_wrist[0] < l_shoulder[0] - 60 or r_wrist[0] > r_shoulder[0] + 60:
            action = "WING REACH SAVE"
            color = (255, 165, 0)
            if last_action != "WING":
                play_sound(650, 100)
                last_action = "WING"

        else:
            last_action = "READY"

        # UI Visual Overlay
        cv2.rectangle(frame, (10, 10), (480, 75), (0, 0, 0), -1)
        cv2.putText(frame, f"ACTION: {action}", (20, 45), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    cv2.imshow('Keeper All-Actions AI Coach', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
