import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import streamlit.components.v1 as components

# Page Setup & Branding
st.set_page_config(page_title="Handball AI Coach", layout="wide")
st.title("🤾‍♂️ Handball AI Coach")
st.caption("✨ **Created by Narayan Pant** ✨")

# Audio Beep Alert Function
def trigger_sound(sound_type):
    freq = 600 if sound_type == "pass" else (1000 if sound_type == "shot" else 850)
    components.html(f"""<script>
        var ctx = new (window.AudioContext || window.webkitAudioContext)();
        var osc = ctx.createOscillator();
        osc.frequency.value = {freq};
        osc.connect(ctx.destination);
        osc.start();
        osc.stop(ctx.currentTime + 0.15);
    </script>""", height=0)

# Sidebar Selection & Branding
st.sidebar.title("App Navigation")
st.sidebar.markdown("**Developer:** Narayan Pant")
st.sidebar.markdown("---")
mode = st.sidebar.selectbox("Choose Tracking Mode:", ["Goalkeeper Mode", "Player Tracking Mode"])

# Tasks API Setup
base_options = python.BaseOptions(model_asset_path='pose_landmarker_heavy.task')
options = vision.PoseLandmarkerOptions(base_options=base_options)
detector = vision.PoseLandmarker.create_from_options(options)

def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    return 360.0 - angle if angle > 180.0 else angle

class HandballAITransformer(VideoTransformerBase):
    def __init__(self):
        self.shot_count = 0
        self.pass_count = 0
        self.r_shot_stage = None
        self.l_shot_stage = None
        self.r_pass_stage = None
        self.l_pass_stage = None
        self.last_sound = None

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)
        h, w, _ = img.shape
        
        rgb_frame = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        detection_result = detector.detect(mp_image)

        if detection_result.pose_landmarks:
            landmarks = detection_result.pose_landmarks[0]

            # Key Landmarks
            l_shoulder = [landmarks[11].x * w, landmarks[11].y * h]
            r_shoulder = [landmarks[12].x * w, landmarks[12].y * h]
            l_elbow = [landmarks[13].x * w, landmarks[13].y * h]
            r_elbow = [landmarks[14].x * w, landmarks[14].y * h]
            l_wrist = [landmarks[15].x * w, landmarks[15].y * h]
            r_wrist = [landmarks[16].x * w, landmarks[16].y * h]

            # ---------------- 1. GOALKEEPER MODE ----------------
            if mode == "Goalkeeper Mode":
                action = "READY STANCE"
                detail = "Keep Knees Bent & Hands Open"
                color = (0, 165, 255)

                if l_wrist[1] < l_shoulder[1] or r_wrist[1] < r_shoulder[1]:
                    action = "HIGH BLOCK SAVE"
                    detail = "Upper Guard Position Active"
                    color = (0, 255, 0)
                    if self.last_sound != "SAVE":
                        self.last_sound = "SAVE"
                        
                elif l_wrist[0] < l_shoulder[0] - 40 or r_wrist[0] > r_shoulder[0] + 40:
                    action = "WING REACH SAVE"
                    detail = "Lateral Wing Coverage"
                    color = (255, 165, 0)
                    if self.last_sound != "SAVE":
                        self.last_sound = "SAVE"
                else:
                    self.last_sound = "READY"

                # UI Display
                cv2.rectangle(img, (10, 10), (460, 85), (0, 0, 0), -1)
                cv2.putText(img, f"KEEPER: {action}", (20, 45), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                cv2.putText(img, f"POS: {detail}", (20, 72), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            # ---------------- 2. PLAYER TRACKING MODE ----------------
            elif mode == "Player Tracking Mode":
                r_angle = calculate_angle(r_shoulder, r_elbow, r_wrist)
                l_angle = calculate_angle(l_shoulder, l_elbow, l_wrist)

                pos_text = "NEUTRAL / WAITING"

                # Right Hand Logic
                if r_wrist[1] < r_shoulder[1]:
                    pos_text = "THROW POSITION: Arm Cocked High"
                    if r_angle < 100: self.r_shot_stage = "READY"
                    if r_angle > 125 and self.r_shot_stage == "READY":
                        self.shot_count += 1
                        self.r_shot_stage = None
                        self.last_sound = "SHOT"
                        pos_text = "SHOT RELEASED!"
                else:
                    if r_angle < 90: 
                        self.r_pass_stage = "READY"
                        pos_text = "PASS POSITION: Chest Ready"
                    if r_angle > 130 and self.r_pass_stage == "READY":
                        self.pass_count += 1
                        self.r_pass_stage = None
                        self.last_sound = "PASS"
                        pos_text = "PASS COMPLETED!"

                # Left Hand Logic
                if l_wrist[1] < l_shoulder[1]:
                    pos_text = "THROW POSITION: Left Arm Cocked"
                    if l_angle < 100: self.l_shot_stage = "READY"
                    if l_angle > 125 and self.l_shot_stage == "READY":
                        self.shot_count += 1
                        self.l_shot_stage = None
                        self.last_sound = "SHOT"
                        pos_text = "SHOT RELEASED!"
                else:
                    if l_angle < 90: 
                        self.l_pass_stage = "READY"
                        pos_text = "PASS POSITION: Chest Ready"
                    if l_angle > 130 and self.l_pass_stage == "READY":
                        self.pass_count += 1
                        self.l_pass_stage = None
                        self.last_sound = "PASS"
                        pos_text = "PASS COMPLETED!"

                # Joint Dots
                cv2.circle(img, (int(r_wrist[0]), int(r_wrist[1])), 7, (0, 255, 0), -1)
                cv2.circle(img, (int(l_wrist[0]), int(l_wrist[1])), 7, (0, 255, 255), -1)

                # Dashboard Overlay
                cv2.rectangle(img, (10, 10), (450, 125), (0, 0, 0), -1)
                cv2.putText(img, f"PASSES: {self.pass_count}", (20, 40), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
                cv2.putText(img, f"SHOTS:  {self.shot_count}", (20, 75), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                cv2.putText(img, f"STATUS: {pos_text}", (20, 110), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            # Footer Watermark inside Frame
            cv2.putText(img, "Created by Narayan Pant", (w - 240, h - 20), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)

        return img

# Streamer Setup
ctx = webrtc_streamer(
    key=f"handball-stream-{mode}", 
    video_transformer_factory=HandballAITransformer
)

# Play Audio Alerts
if ctx.video_processor:
    if ctx.video_processor.last_sound == "PASS":
        trigger_sound("pass")
        ctx.video_processor.last_sound = None
    elif ctx.video_processor.last_sound == "SHOT":
        trigger_sound("shot")
        ctx.video_processor.last_sound = None
    elif ctx.video_processor.last_sound == "SAVE":
        trigger_sound("save")
        ctx.video_processor.last_sound = None

# App Bottom Branding
st.markdown("---")
# ✅ नई फिक्स की हुई लाइन:
st.html("<h4 style='text-align: center; color: #4CAF50;'>Handball AI Coach System | Created by Narayan Pant</h4>")
