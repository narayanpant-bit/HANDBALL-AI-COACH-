class HandballAnalyzer:
    """
    EXPERIMENTAL MVP - NOT FOR MEDICAL OR PROFESSIONAL USE.
    """

    def __init__(self):
        self.status = "Idle"
        self.feedback = "Camera ke samne aayein"

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

        r_elbow_angle = angles.get("Right Elbow", 0)
        l_elbow_angle = angles.get("Left Elbow", 0)

        # Right / Left Arm Throwing posture detection
        r_raised = (r_wrist and r_elbow and r_shoulder) and (r_wrist[1] < r_elbow[1] < r_shoulder[1]) and (70 <= r_elbow_angle <= 130)
        l_raised = (l_wrist and l_elbow and l_shoulder) and (l_wrist[1] < l_elbow[1] < l_shoulder[1]) and (70 <= l_elbow_angle <= 130)

        if r_raised or l_raised:
            self.status = "Throwing / Shooting Motion"
            self.feedback = "Haath upar hai. Throwing position acchhi hai."
        elif (r_wrist and r_shoulder and r_wrist[1] < r_shoulder[1]) or (l_wrist and l_shoulder and l_wrist[1] < l_shoulder[1]):
            self.status = "Guard / Catch Position"
            self.feedback = "Haath guard position me hain."
        else:
            self.status = "Neutral / Running Stance"
            self.feedback = "Ready stance banaye rakhein."
