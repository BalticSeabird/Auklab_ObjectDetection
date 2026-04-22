

from ultralytics import YOLO
import torch

torch.cuda.empty_cache()

prefix = "seabird_fish"
dataset_version = 7190

#dataset_version = input("Enter dataset version (e.g., 6080): ").strip()

# Load a COCO-pretrained YOLO11m model
model = YOLO("models/yolo26x.pt")

# Train the model on the dataset
results = model.train(data=f"dataset/dataset_{prefix}_{dataset_version}.yaml", batch=32, epochs=100, imgsz=680, device = [0, 1])

# Save the model
model.save(f'models/auklab_yolo26xl_{prefix}_{dataset_version}.pt')
