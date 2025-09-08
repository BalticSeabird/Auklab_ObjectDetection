

# Importing the required libraries
from ultralytics import YOLO
from clearml import Task
import torch

torch.cuda.empty_cache()

# Initialize ClearML task
task = Task.init(project_name="YOLOv11 Training fish-seabird", task_name="YOLOv11 fish-seabird")

# Load a COCO-pretrained YOLO11m model
model = YOLO("models/yolo11x.pt")

# Train the model on the dataset
results = model.train(data="dataset/dataset_combined_3495.yaml", batch=16, epochs=200, imgsz=960, device = [0, 1])

# Log the results to ClearML
task.upload_artifact('training_results_dataset_combined3495', results)

# Save the model
model.save('models/auklab_model_xlarge_combined_3495_v1.pt')

# Close the ClearML task
task.close()
