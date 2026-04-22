import pandas as pd
from pathlib import Path 
from ultralytics import YOLO
import os 
import sys
import cv2

# Load a pretrained YOLO model
#modelpath = Path("models/eider_model_nano_v5852.pt")
modelpath = Path("runs/detect/train24/weights/tag_detection_nano1347.pt")

model = YOLO(modelpath)
modelname = modelpath.stem

ims = list(Path("../../../../../../mnt/BSP_NAS2_work/ring_reader/test_results/step1_ring_crops/").rglob("*.png"))
print(f"Found {len(ims)} images")

# Create output directory for annotated images
output_dir = Path("./output_predictions")
output_dir.mkdir(exist_ok=True)

# Run inference using the pretrained model
for idx, im in enumerate(ims):
    print(f"Processing image {idx + 1}/{len(ims)}: {im.name}")

    # Run YOLOv8 inference on the image
    results = model.predict(im, conf=0.5)

    # Get the annotated image with predictions drawn
    annotated_frame = results[0].plot()

    # Save the annotated image
    output_path = output_dir / f"pred_{im.name}"
    cv2.imwrite(str(output_path), annotated_frame)
    print(f"  Saved to: {output_path}")

print(f"\nAll predictions saved to {output_dir.resolve()}")

    



