import random
import shutil
from pathlib import Path

def sample_dataset(image_dir, label_dir, output_dir, sample_size=2000, seed=42):
    random.seed(seed)

    image_paths = []
    for ext in ["*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"]:
        image_paths.extend(Path(image_dir).rglob(ext))

    # Shuffle and pick sample
    sample_paths = random.sample(image_paths, min(sample_size, len(image_paths)))

    # Output directories
    out_img_dir = Path(output_dir) / "images"
    out_lbl_dir = Path(output_dir) / "labels"
    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_lbl_dir.mkdir(parents=True, exist_ok=True)

    for img_path in sample_paths:
        # Copy image
        shutil.copy(img_path, out_img_dir / img_path.name)

        # Copy matching label file
        label_path = Path(label_dir) / (img_path.stem + ".txt")
        if label_path.exists():
            shutil.copy(label_path, out_lbl_dir / label_path.name)
        else:
            # If no label exists, create an empty file
            open(out_lbl_dir / (img_path.stem + ".txt"), "w").close()

    print(f"Sampled {len(sample_paths)} images → saved to {output_dir}")

# Example usage
sample_dataset(
    image_dir="dataset/images",      # your merged images folder
    label_dir="merged_labels",       # your merged labels folder
    output_dir="sample_2000",        # new folder for roboflow
    sample_size=2000
)
