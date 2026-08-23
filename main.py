import cv2
import time
from ultralytics import YOLO
from utils import calculate_angle
from analyzer import HandballAnalyzer

DISCLAIMER_TEXT = "EXPERIMENTAL MVP - NOT FOR MEDICAL/PROFESSIONAL USE"

def main():
    # YOLOv8 Pose Model load karein (No MediaPipe needed)
    model = YOLO("yolov8n-pose.pt")

    analyzer = HandballAnalyzer()
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    prev_time = time.time()

    # YOLO Pose landmark connection map
    connections = [
        (5, 7), (7, 9),      # Right Arm (Shoulder->Elbow->Wrist)
        (6, 8), (8, 10),     # Left Arm
        (5, 6),              # Shoulders
        (5, 11), (6, 12),    # Torso (Shoulder->Hip)
        (11, 12),            # Hips
        (11, 13), (13, 15),  # Right Leg (Hip->Knee->Ankle)
        (12, 14), (14, 16)   # Left Leg
    ]

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Camera error")
            break

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape

        # Run YOLO Tracking
        results = model(frame, verbose=False)
        angles = {}
        keypoints_px = {}

        if results and len(results[0].keypoints) > 0:
            # First detected person
            kp = results[0].keypoints.xy[0].cpu().numpy()

            if len(kp) >= 17:
                # Keypoints index mapping for YOLO:
                # 0: Nose, 5: L_Shoulder, 6: R_Shoulder, 7: L_Elbow, 8: R_Elbow,
                # 9: L_Wrist, 10: R_Wrist, 11: L_Hip, 12: R_Hip, 13: L_Knee, 14: R_Knee, 15: L_Ankle, 16: R_Ankle
                keypoints_px = {
                    "nose": (int(kp[0][0]), int(kp[0][1])),
                    "l_shoulder": (int(kp[5][0]), int(kp[5][1])),
                    "r_shoulder": (int(kp[6][0]), int(kp[6][1])),
                    "l_elbow": (int(kp[7][0]), int(kp[7][1])),
                    "r_elbow": (int(kp[8][0]), int(kp[8][1])),
                    "l_wrist": (int(kp[9][0]), int(kp[9][1])),
                    "r_wrist": (int(kp[10][0]), int(kp[10][1])),
                    "l_hip": (int(kp[11][0]), int(kp[11][1])),
                    "r_hip": (int(kp[12][0]), int(kp[12][1])),
                    "l_knee": (int(kp[13][0]), int(kp[13][1])),
                    "r_knee": (int(kp[14][0]), int(kp[14][1])),
                    "l_ankle": (int(kp[15][0]), int(kp[15][1])),
                    "r_ankle": (int(kp[16][0]), int(kp[16][1])),
                }

                # Skeleton Connections Draw Karein
                for p1, p2 in connections:
                    pt1 = (int(kp[p1][0]), int(kp[p1][1]))
                    pt2 = (int(kp[p2][0]), int(kp[p2][1]))
                    if pt1 != (0, 0) and pt2 != (0, 0):
                        cv2.line(frame, pt1, pt2, (0, 255, 0), 2)

                # Keypoint Dots Draw Karein
                for name, pt in keypoints_px.items():
                    if pt != (0, 0):
                        cv2.circle(frame, pt, 5, (0, 0, 255), -1)

                # Calculate Angles
                if keypoints_px["l_shoulder"] != (0, 0) and keypoints_px["l_elbow"] != (0, 0) and keypoints_px["l_wrist"] != (0, 0):
                    angles["Left Elbow"] = calculate_angle(keypoints_px["l_shoulder"], keypoints_px["l_elbow"], keypoints_px["l_wrist"])
                    cv2.putText(frame, f"{angles['Left Elbow']:.0f} deg", keypoints_px["l_elbow"], cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

                if keypoints_px["r_shoulder"] != (0, 0) and keypoints_px["r_elbow"] != (0, 0) and keypoints_px["r_wrist"] != (0, 0):
                    angles["Right Elbow"] = calculate_angle(keypoints_px["r_shoulder"], keypoints_px["r_elbow"], keypoints_px["r_wrist"])
                    cv2.putText(frame, f"{angles['Right Elbow']:.0f} deg", keypoints_px["r_elbow"], cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

                if keypoints_px["l_hip"] != (0, 0) and keypoints_px["l_knee"] != (0, 0) and keypoints_px["l_ankle"] != (0, 0):
                    angles["Left Knee"] = calculate_angle(keypoints_px["l_hip"], keypoints_px["l_knee"], keypoints_px["l_ankle"])
                    cv2.putText(frame, f"{angles['Left Knee']:.0f} deg", keypoints_px["l_knee"], cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

                if keypoints_px["r_hip"] != (0, 0) and keypoints_px["r_knee"] != (0, 0) and keypoints_px["r_ankle"] != (0, 0):
                    angles["Right Knee"] = calculate_angle(keypoints_px["r_hip"], keypoints_px["r_knee"], keypoints_px["r_ankle"])
                    cv2.putText(frame, f"{angles['Right Knee']:.0f} deg", keypoints_px["r_knee"], cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

                analyzer.analyze_pose(angles, keypoints_px)
        else:
            analyzer.analyze_pose(angles, None)

        # UI Overlay
        curr_time = time.time()
        fps = 1.0 / (curr_time - prev_time + 1e-6)
        prev_time = curr_time

        # Header
        cv2.rectangle(frame, (0, 0), (w, 60), (30, 30, 30), -1)
        cv2.putText(frame, "HANDBALL AI COACH (YOLO POSE)", (20, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.putText(frame, f"FPS: {int(fps)}", (w - 130, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Footer Panel
        cv2.rectangle(frame, (20, h - 110), (w - 20, h - 20), (20, 20, 20), -1)
        cv2.rectangle(frame, (20, h - 110), (w - 20, h - 20), (0, 255, 255), 2)

        cv2.putText(frame, f"State: {analyzer.status}", (40, h - 75), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"Feedback: {analyzer.feedback}", (40, h - 45), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

        cv2.putText(frame, DISCLAIMER_TEXT, (20, h - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

        cv2.imshow("Handball AI Coach (No MediaPipe)", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
