import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import winsound
import threading

def play_sound(frequency, duration=100):
    threading.Thread(target=winsound.Beep, args=(frequency, duration), daemon=True).start()

# Setup Tasks API Landmarker
base_options = python.BaseOptions(model_asset_path='pose_landmarker_heavy.task')
options = vision.PoseLandmarkerOptions(
    base_options=base_options,
    output_segmentation_masks=False)
detector = vision.PoseLandmarker.create_from_options(options)

cap = cv2.VideoCapture(0)
last_pose = ""

print("Keeper AI Tracker Started!")

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

        # Key Landmarks
        l_shoulder = [landmarks[11].x * w, landmarks[11].y * h]
        r_shoulder = [landmarks[12].x * w, landmarks[12].y * h]
        l_wrist = [landmarks[15].x * w, landmarks[15].y * h]
        r_wrist = [landmarks[16].x * w, landmarks[16].y * h]
        
        # Calculate Arm Span (हाथों की चौड़ाई)
        arm_span = np.linalg.norm(np.array(l_wrist) - np.array(r_wrist))
        shoulder_width = np.linalg.norm(np.array(l_shoulder) - np.array(r_shoulder))
        
        status = "NEUTRAL"
        color = (255, 255, 255)

        # 1. READY / GUARD STANCE (हाथ कंधे के पास या ऊपर फैले हैं)
        if l_wrist[1] < l_shoulder[1] and r_wrist[1] < r_shoulder[1]:
            status = "HIGH BLOCK / SAVE"
            color = (0, 255, 0)
            if last_pose != "HIGH":
                play_sound(1000, 100) # Quick High Beep
                last_pose = "HIGH"
                
        elif l_wrist[0] < l_shoulder[0] - 40 or r_wrist[0] > r_shoulder[0] + 40:
            status = "WIDE WING COVERAGE"
            color = (255, 255, 0)
            if last_pose != "WIDE":
                play_sound(700, 100)
                last_pose = "WIDE"
        else:
            status = "READY STANCE"
            color = (0, 165, 255)
            last_pose = "READY"

        # Points visual draw
        cv2.circle(frame, (int(l_wrist[0]), int(l_wrist[1])), 10, (0, 255, 0), -1)
        cv2.circle(frame, (int(r_wrist[0]), int(r_wrist[1])), 10, (0, 255, 0), -1)
        cv2.line(frame, (int(l_wrist[0]), int(l_wrist[1])), (int(r_wrist[0]), int(r_wrist[1])), (255, 0, 0), 2)

        # UI Screen Overlay
        cv2.rectangle(frame, (10, 10), (450, 80), (0, 0, 0), -1)
        cv2.putText(frame, f"KEEPER POSE: {status}", (20, 45), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.putText(frame, f"Reach Ratio: {round(arm_span/shoulder_width, 2)}x", (20, 70), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    cv2.imshow('Handball AI - Keeper Coach', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
