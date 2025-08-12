from ultralytics import YOLO
import os
from pathlib import Path


from ultralytics import YOLO
from pathlib import Path
import os
import numpy as np

def merge_predictions(model1_path, model2_path, image_dir, save_label_dir,
                      num_classes_model1, conf_threshold=0.05):

    model1 = YOLO(model1_path)
    model2 = YOLO(model2_path)

    image_paths = []
    for ext in ["*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"]:
        image_paths.extend(Path(image_dir).rglob(ext))

    os.makedirs(save_label_dir, exist_ok=True)

    for img_path in image_paths:
        merged_annotations = []

        # Model 1 predictions
        res1 = model1.predict(source=img_path, conf=conf_threshold, verbose=False)[0]
        boxes1 = res1.boxes.xywhn.cpu().numpy()
        cls1 = res1.boxes.cls.cpu().numpy().astype(int)
        for c, (x, y, w, h) in zip(cls1, boxes1):
            merged_annotations.append(f"{c} {x:.6f} {y:.6f} {w:.6f} {h:.6f}")

        # Model 2 predictions (class IDs shifted)
        res2 = model2.predict(source=img_path, conf=conf_threshold, verbose=False)[0]
        boxes2 = res2.boxes.xywhn.cpu().numpy()
        cls2 = res2.boxes.cls.cpu().numpy().astype(int)
        for c, (x, y, w, h) in zip(cls2, boxes2):
            merged_annotations.append(f"{c + num_classes_model1} {x:.6f} {y:.6f} {w:.6f} {h:.6f}")

        # Save merged annotations
        label_path = Path(save_label_dir) / (img_path.stem + ".txt")
        with open(label_path, "w") as f:
            for ann in merged_annotations:
                f.write(ann + "\n")

        print(f"Merged annotations for {img_path.name} → {label_path.name}")

# Example usage
merge_predictions(
    model1_path="models/auklab_model_nano_v4295.pt",
    model2_path="models/auklab_model_nano_fishJuly2025.pt",
    image_dir="../../../../../../../mnt/BSP_NAS2/Software_Models/fish_model/annotations/train/images",
    save_label_dir="../../../../../../../mnt/BSP_NAS2_work/seabird_fish_model/annotations/combined_labels_fish_images",
    num_classes_model1=3,  # fish = 1 class in model1
    conf_threshold=0.25
)

#merge_predictions(
#    model1_path="models/auklab_model_nano_v4295.pt",
#    model2_path="models/auklab_model_nano_fishJuly2025.pt",
#    image_dir="../../../../../../../mnt/BSP_NAS2/Software_Models/auklab_model/Annotations/Seabird_detection/all/images",
#    save_label_dir="../../../../../../../mnt/BSP_NAS2_work/seabird_fish_model/annotations/combined_labels_bird_images",
#    num_classes_model1=3,  # fish = 1 class in model1
#    conf_threshold=0.25
#)


