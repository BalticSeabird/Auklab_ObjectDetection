from pathlib import Path 
from ultralytics import YOLO
import cv2

# Load a pretrained YOLO model
modelpath = Path("models/RingRead20230722_yolov8n_640.pt")
model = YOLO(modelpath)

# Get all images in the folder
image_folder = Path("images")
image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']
images = []
for ext in image_extensions:
    images.extend(image_folder.glob(ext))

if not images:
    print(f"No images found in {image_folder}/")
    exit()

print(f"Found {len(images)} images")
print("Press any key to move to next image, 'q' to quit\n")

# Loop through images
for im in images:
    print(f"Processing: {im.name}")
    
    # Read image
    frame = cv2.imread(str(im))
    if frame is None:
        print(f"Failed to read image: {im}")
        continue
    
    # Run inference
    results = model.predict(frame, verbose=False)
    
    # Draw bounding boxes on the frame
    annotated_frame = results[0].plot()
    
    # Display the image with detections
    cv2.imshow('Object Detection - Press any key for next, q to quit', annotated_frame)
    
    # Wait for key press
    key = cv2.waitKey(0)
    
    # Quit if 'q' is pressed
    if key == ord('q'):
        print("Quitting...")
        break

# Clean up
cv2.destroyAllWindows()




