import cv2
import pyttsx3
import threading

class HandballAnalyzer:
    """
    EXPERIMENTAL MVP - NOT FOR MEDICAL OR PROFESSIONAL USE.
    """

    def __init__(self):
        self.status = "Idle"
        self.feedback = "Camera ke samne aayein"
        self.throw_count = 0
        self.pass_count = 0
        
        # States
        self.r_stage = "down"
        self.l_stage = "down"
        self.pass_stage = "idle"
        
        # Jump tracking
        self.base_hip_y = None
        self.max_jump_height = 0
        self.is_jumping = False

        # Voice Engine Initialization (Runs in background thread so video doesn't lag)
        try:
            self.engine = pyttsx3.init()
            self.engine.setProperty('rate', 160)  # Speaking speed
        except Exception as e:
            self.engine = None

    def speak(self, text: str):
        """Asynchronous voice feedback to prevent camera lag."""
        if self.engine:
            def run_speech():
                try:
                    engine = pyttsx3.init()
                    engine.setProperty('rate', 160)
                    engine.say(text)
                    engine.runAndWait()
                except:
                    pass
            threading.Thread(target=run_speech, daemon=True).start()

    def analyze_pose(self, angles: dict, keypoints_px: dict):
        if not keypoints_px:
            self.status = "Koi player nahi dikha"
            self.feedback = "Frame me aayein"
            return

        r_shoulder = keypoints_px.get("r_shoulder")
        r_elbow = keypoints_px.get("r_elbow")
        r_wrist = keypoints_px.get("r_wrist")

        l_shoulder = keypoints_px.get("l_shoulder")
        l_elbow = keypoints_px.get("l_elbow")
        l_wrist = keypoints_px.get("l_wrist")

        r_hip = keypoints_px.get("r_hip")
        l_hip = keypoints_px.get("l_hip")

        r_elbow_angle = angles.get("Right Elbow", 0)
        l_elbow_angle = angles.get("Left Elbow", 0)

        # 1. JUMP DETECTOR
        if r_hip and l_hip and r_hip != (0, 0) and l_hip != (0, 0):
            current_hip_y = (r_hip[1] + l_hip[1]) / 2.0

            if self.base_hip_y is None:
                self.base_hip_y = current_hip_y
            else:
                self.base_hip_y = 0.95 * self.base_hip_y + 0.05 * current_hip_y

            jump_pixels = self.base_hip_y - current_hip_y

            if jump_pixels > 35:
                self.is_jumping = True
                if jump_pixels > self.max_jump_height:
                    self.max_jump_height = int(jump_pixels)
            else:
                self.is_jumping = False

        # 2. PASS DETECTOR
        if r_wrist and l_wrist and r_shoulder and l_shoulder:
            both_hands_at_chest = (r_wrist[1] > r_shoulder[1]) and (l_wrist[1] > l_shoulder[1]) and (r_elbow_angle < 100) and (l_elbow_angle < 100)
            both_hands_extended = (r_elbow_angle > 150) and (l_elbow_angle > 150)

            if both_hands_at_chest:
                self.pass_stage = "ready"
            elif self.pass_stage == "ready" and both_hands_extended:
                self.pass_stage = "idle"
                self.pass_count += 1
                self.status = "CHEST PASS DETECTED!"
                self.feedback = f"Pass Complete! Total Passes: {self.pass_count}"
                self.speak("Nice Pass!")
                return

        # 3. THROW DETECTOR
        r_released = False
        l_released = False

        if r_wrist and r_elbow and r_shoulder and r_wrist != (0, 0):
            if r_wrist[1] < r_shoulder[1] and 70 <= r_elbow_angle <= 120:
                self.r_stage = "cocked"
            elif self.r_stage == "cocked" and r_elbow_angle > 140 and r_wrist[1] > r_elbow[1]:
                self.r_stage = "down"
                r_released = True

        if l_wrist and l_elbow and l_shoulder and l_wrist != (0, 0):
            if l_wrist[1] < l_shoulder[1] and 70 <= l_elbow_angle <= 120:
                self.l_stage = "cocked"
            elif self.l_stage == "cocked" and l_elbow_angle > 140 and l_wrist[1] > l_elbow[1]:
                self.l_stage = "down"
                l_released = True

        # Status & Voice Feedback Logic
        if r_released or l_released:
            self.throw_count += 1
            if self.is_jumping:
                self.status = "JUMP SHOT RELEASED!"
                self.feedback = f"Awesome Jump-Shot! Max Elevation: {self.max_jump_height}px"
                self.speak("Great Jump Shot!")
            else:
                self.status = "STANDING SHOT RELEASED!"
                self.feedback = f"Shot Counted! Total Throws: {self.throw_count}"
                self.speak("Good Shot!")
        elif self.is_jumping:
            self.status = "IN THE AIR (JUMPING)"
            self.feedback = f"Jumping... Elevation: {int(self.max_jump_height)}px"
        elif self.r_stage == "cocked" or self.l_stage == "cocked":
            self.status = "Throwing Preparation"
            self.feedback = "Arm cocked high. Ready to throw!"
        else:
            self.status = "Neutral Stance"
            self.feedback = "Ready stance banayein."
