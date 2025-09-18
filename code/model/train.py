

# Importing the required libraries
from ultralytics import YOLO
from clearml import Task
import torch

torch.cuda.empty_cache()

dataset_version = input("Enter dataset version (e.g., 4564): ").strip()

# Initialize ClearML task
task = Task.init(project_name="YOLOv11 Training fish-seabird", task_name="YOLOv11 fish-seabird")

# Load a COCO-pretrained YOLO11m model
model = YOLO("models/yolo11x.pt")

# Train the model on the dataset
results = model.train(data=f"dataset/dataset_combined_{dataset_version}.yaml", batch=16, epochs=200, imgsz=960, device = [0, 1])

# Log the results to ClearML
task.upload_artifact(f'training_results_dataset_combined_{dataset_version}', results)

# Save the model
model.save(f'models/auklab_model_xlarge_combined_{dataset_version}_v1.pt')

# Close the ClearML task
task.close()
