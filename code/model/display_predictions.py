from ultralytics import YOLO
import cv2
from pathlib import Path

def browse_predictions(model_path, image_dir, conf=0.25):
    # Load model
    model = YOLO(model_path)

    # Collect images
    image_paths = []
    for ext in ["*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"]:
        image_paths.extend(Path(image_dir).rglob(ext))

    image_paths = sorted(image_paths)
    if not image_paths:
        print("No images found!")
        return

    i = 0
    while 0 <= i < len(image_paths):
        img_path = image_paths[i]
        results = model.predict(source=img_path, conf=conf, verbose=False)[0]

        # Draw predictions on image
        pred_img = results.plot()

        # Show in OpenCV window
        cv2.imshow("YOLOv11 Predictions (q=quit, ← back, → next)", pred_img)

        # Wait for key press
        key = cv2.waitKey(0)

        if key == ord("q"):   # quit
            break
        elif key == 81 or key == ord("a"):  # left arrow or 'a'
            i = max(i - 1, 0)
        elif key == 83 or key == ord("d"):  # right arrow or 'd'
            i = min(i + 1, len(image_paths) - 1)
        else:
            # Default → move forward
            i = min(i + 1, len(image_paths) - 1)

    cv2.destroyAllWindows()


# Example usage
browse_predictions(
    model_path="models/auklab_combined_v1394_v2.pt",
    image_dir="../../../../Downloads/ims",
    conf=0.25
)
