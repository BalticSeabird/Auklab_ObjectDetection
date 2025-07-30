from ultralytics import YOLO
import os
from pathlib import Path


def annotate_dataset_with_model(model_path, image_dir, save_label_dir, class_offset=0, conf_threshold=0.25):
    """
    Runs inference with a given YOLO model on images, and saves predictions as YOLO-format labels.
    Adjusts class indices using `class_offset` to avoid conflicts when merging datasets.
    """
    model = YOLO(model_path)
    image_paths = list(Path(image_dir).rglob("*.jpg")) + list(Path(image_dir).rglob("*.png"))

    os.makedirs(save_label_dir, exist_ok=True)

    for img_path in image_paths:
        results = model.predict(source=img_path, conf=conf_threshold, save=False, verbose=False)[0]

        # YOLOv8/11 returns results in .boxes.xywhn and .boxes.cls
        boxes = results.boxes.xywhn.cpu().numpy()
        classes = results.boxes.cls.cpu().numpy().astype(int)

        label_path = Path(save_label_dir) / (img_path.stem + ".txt")

        with open(label_path, "w") as f:
            for cls, (x, y, w, h) in zip(classes, boxes):
                cls_adjusted = cls + class_offset
                f.write(f"{cls_adjusted} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")

        print(f"Annotated {img_path.name} → {label_path.name}")


#model1 = auklab
#model2 = fish

basic_path = "dataset2/images"

# 1. Annotate fish images with auklab model (class_offset = 0)
annotate_dataset_with_model(
    model_path="models/auklab_model_nano_v4295.pt",
    image_dir="../../../../../../../mnt/BSP_NAS2/Software_Models/fish_model/annotations/all/images",
    save_label_dir="../../../../../../../mnt/BSP_NAS2_work/seabird_fish_model/annotations/pseudo_labels_auklab_model_fish_images",
    class_offset=0  # assumes model1 trained on class IDs 0–n1
)

# 2. Annotate auklab images with fish (class_offset = N1)
NUM_CLASSES_MODEL1 = 3  # Adjust to match actual number of classes in model1
annotate_dataset_with_model(
    model_path="models/auklab_model_nano_fishJuly2025.pt",
    image_dir="../../../../../../../mnt/BSP_NAS2/Software_Models/auklab_model/Annotations/Seabird_detection/all/images",
    save_label_dir="../../../../../../../mnt/BSP_NAS2_work/seabird_fish_model/annotations/pseudo_labels_fish_model_auklab_images",
    class_offset=NUM_CLASSES_MODEL1
)