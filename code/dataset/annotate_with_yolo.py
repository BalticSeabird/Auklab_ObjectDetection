from ultralytics import YOLO
from pathlib import Path
import os

def annotate_with_yolo(model_path, image_dir, output_dir, conf=0.25):
    """
    Runs inference on all images in a folder and saves YOLO-format labels.
    """
    model = YOLO(model_path)

    # Find all images
    exts = ["*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"]
    image_paths = []
    for ext in exts:
        image_paths.extend(Path(image_dir).rglob(ext))

    if not image_paths:
        print("No images found!")
        return

    os.makedirs(output_dir, exist_ok=True)

    for img_path in image_paths:
        results = model.predict(source=img_path, conf=conf, verbose=False)[0]

        boxes = results.boxes.xywhn.cpu().numpy()  # normalized xywh
        classes = results.boxes.cls.cpu().numpy().astype(int)

        # One .txt file per image
        label_path = Path(output_dir) / f"{img_path.stem}.txt"
        with open(label_path, "w") as f:
            for cls, (x, y, w, h) in zip(classes, boxes):
                f.write(f"{cls} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")

        print(f"Saved {label_path.name} with {len(classes)} objects")

    print(f"\n✅ Done! Annotations saved to {output_dir}")

# Example usage
annotate_with_yolo(
    model_path="models/auklab_combined_v1978_v1.pt",
    image_dir="../../../../Downloads/ims/",     # folder with unlabeled images
    output_dir="../../../../Downloads/labels",    # where .txt files will be written
    conf=0.25
)
