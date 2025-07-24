import cv2
import random
import os
from pathlib import Path

# === Parameters ===
video_path = Path("video/Auklab1_FAR6_2024-07-12_13.00.00.mp4")
output_dir = "images"
num_samples = 50

# === Setup ===
os.makedirs(output_dir, exist_ok=True)
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    raise IOError(f"Cannot open video: {video_path}")

total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(f"Total frames in video: {total_frames}")

# === Sample unique frame indices ===
random_indices = sorted(random.sample(range(total_frames), num_samples))

# === Grab and save frames ===
for i, frame_idx in enumerate(random_indices):
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    
    if not ret:
        print(f"Warning: Could not read frame {frame_idx}")
        continue

    filename = os.path.join(output_dir, f"{video_path.stem}{frame_idx:06d}.jpg")
    cv2.imwrite(filename, frame)

    print(f"[{i+1}/{num_samples}] Saved {filename}")

cap.release()
print("Done.")
