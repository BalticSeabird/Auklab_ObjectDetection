import os
import glob
from pathlib import Path
from roboflow import Roboflow

# Authenticate
# Authenticate with your API key
rf = Roboflow(api_key="2Z8LedwxqBlKAbVYyz8T")

# Connect to your workspace and project
project = rf.workspace("ai-course-2024").project("fish_seabirds_combined-625bd")

# Dataset base paths (adjust to your case)
image_dir = Path("images")
label_dir = Path("labels")

# Accepted extensions
extensions = [".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"]

# Loop over all images
for img_path in image_dir.iterdir():
    if img_path.suffix not in extensions:
        continue

    label_path = label_dir / f"{img_path.stem}.txt"

    if label_path.exists():
        print(f"Uploading {img_path.name} with label → train split")
        project.upload(
            image_path=str(img_path),
            annotation_path=str(label_path),
            split="train"
        )
    else:
        print(f"Uploading {img_path.name} without label → train split")
        project.upload(
            image_path=str(img_path),
            split="train"
        )
        