from ultralytics import YOLO
import matplotlib.pyplot as plt
from pathlib import Path
import cv2

# Paths to models
model1_path = "models/auklab_model_nano_fishJuly2025.pt"
model2_path = "models/auklab_model_nano_v4295.pt"

# Load models
model1 = YOLO(model1_path)
model2 = YOLO(model2_path)

# Test image path
test_image = "images/Auklab1_FAR3_2023-07-01_725_960_5382.png"  # change this

# Run inference
results1 = model1.predict(source=test_image, conf=0.05, verbose=False)[0]
results2 = model2.predict(source=test_image, conf=0.05, verbose=False)[0]

# Plot results (Ultralytics returns BGR np.array)
img1 = results1.plot()  # predictions drawn
img2 = results2.plot()

# Convert to RGB for matplotlib
img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

# Display side by side
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.imshow(img1)
plt.title("Model 1 Predictions")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(img2)
plt.title("Model 2 Predictions")
plt.axis("off")

plt.tight_layout()
plt.show()