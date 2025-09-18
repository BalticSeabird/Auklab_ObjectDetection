

# Importing the required libraries
from ultralytics import YOLO

# Load a COCO-pretrained YOLO11m model
model = YOLO("models/yolo11x.pt")

# Train the model on the dataset
results = model.train(data="dataset/dataset_combined_4211.yaml", 
                      batch=16, epochs=200, imgsz=960, device = [0, 1])

# Save the model
model.save('models/auklab_model_xlarge_combined_4211_v1.pt')
