

# Importing the required libraries
from ultralytics import YOLO
from clearml import Task
import torch

torch.cuda.empty_cache()

# Initialize ClearML task
task = Task.init(project_name="YOLOv11 Training Fish", task_name="YOLOv11 Model Fish")

# Load a COCO-pretrained YOLO11m model
model = YOLO("models/auklab_model_nano_v4295.pt")

# Train the model on the dataset for 50 epochs
results = model.train(data="dataset/dataset_fishJuly2025.yaml", batch=32, epochs=200, imgsz=960, device = [0, 1])

# Log the results to ClearML
task.upload_artifact('training_results_dataset_fishJuly2025', results)

# Save the model
model.save('models/auklab_model_nano_fishJuly2025.pt')

# Close the ClearML task
task.close()
