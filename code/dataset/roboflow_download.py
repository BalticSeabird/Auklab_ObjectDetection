from roboflow import Roboflow

# Authenticate with your API key
rf = Roboflow(api_key="2Z8LedwxqBlKAbVYyz8T")

# Connect to your workspace and project
project = rf.workspace("ai-course-2024").project("fish_seabirds_combined-625bd")

# Choose dataset version
dataset = project.version(7).download("yolov11")  # or "coco", "voc", "yolov8", etc.

print("Dataset downloaded to:", dataset.location)