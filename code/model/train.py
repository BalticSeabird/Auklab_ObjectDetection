

# Importing the required libraries
from roboflow import Roboflow
from ultralytics import YOLO
from clearml import Task
import torch

torch.cuda.empty_cache()

# Creating an instance of the Roboflow class
rf = Roboflow(api_key="X2yHJUrUKxkMNDPlzaAd")
project = rf.workspace("research-x1kcu").project("ejder3")
version = project.version(2)
dataset = version.download("yolov11")

#PYTORCH_CUDA_ALLOC_CONF=expandable_segments:Tr          

# Initialize ClearML task
task = Task.init(project_name="YOLOv11 Training Eiders", task_name="YOLOv11 Model Eiders")

# Load a COCO-pretrained YOLO11n model
model = YOLO("models/yolo11n.pt")

# Train the model on the dataset for 50 epochs
results = model.train(data="Ejder3-2/data.yaml", batch=16, epochs=200, imgsz=960, device = [0, 1])

# Log the results to ClearML
task.upload_artifact('training_results_dataset_v5852', results)

# Save the model
model.save('models/eider_model_nano_v5852.pt')

# Close the ClearML task
task.close()
