

# Importing the required libraries
from ultralytics import YOLO
from clearml import Task
import torch

torch.cuda.empty_cache()

# Initialize ClearML task
task = Task.init(project_name="YOLOv11 Training Auklab", task_name="YOLOv11 Model Auklab")

# Load a COCO-pretrained YOLO11m model
model = YOLO("models/yolo11m_pretrained.pt")

# Train the model on the dataset for 50 epochs
results = model.train(data="dataset/dataset3851.yaml", batch=16, epochs=200, imgsz=960, device = [0, 1])

# Log the results to ClearML
task.upload_artifact('training_results_dataset_auklab_v3851', results)

# Save the model
model.save('models/auklab_model_medium_v3851.pt')

# Close the ClearML task
task.close()
