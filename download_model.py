import urllib.request
import os

model_url = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task"
filename = "pose_landmarker_heavy.task"

print("Downloading pose_landmarker_heavy.task file... Please wait.")

try:
    urllib.request.urlretrieve(model_url, filename)
    print("✅ File downloaded successfully!")
    print(f"Saved in: {os.path.abspath(filename)}")
except Exception as e:
    print(f"❌ Download failed: {e}")
