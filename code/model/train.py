

from ultralytics import YOLO
import torch

torch.cuda.empty_cache()

# Empty cache of images (in shell...)
#find /home/jonas/Documents/vscode/Auklab_OD -name "*.cache" -type f -delete


#RUN EXAMPLE
"""
python3 code/model/train.py 2>&1 | tee ../../../../../mnt/BSP_NAS2_work/auklab_model/yolo_train_$(date +%Y%m%d_%H%M%S).log
"""

# Load a COCO-pretrained YOLO model
model = YOLO("models/yolo26x.pt")

# Specify dataset
prefix = "seabird_fish"
dataset_version = 10000

# Complete path to the dataset YAML file
dataset_yaml = f"dataset/dataset_{prefix}_{dataset_version}.yaml"

# Train the model on the dataset
results = model.train(data=dataset_yaml, batch=16, epochs=100, imgsz=680, device = [0, 1])

# Save the model
model.save(f'models/auklab_yolo26x_{prefix}_{dataset_version}_V2.pt')
