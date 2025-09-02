import os
import shutil
import random
from pathlib import Path

# Source base folder
source_base = Path("fish_seabirds_combined-6")
source_splits = ['train', 'valid', 'test']

# Destination base folder
dest_base = Path("dataset/seabird_fish2206")
dest_splits = ['train', 'val', 'test']
split_ratios = [0.8, 0.1, 0.1]

# Make destination directories
for split in dest_splits:
    (dest_base / "images" / split).mkdir(parents=True, exist_ok=True)
    (dest_base / "labels" / split).mkdir(parents=True, exist_ok=True)

# Collect image-label pairs
image_label_pairs = []

for src_split in source_splits:
    image_dir = source_base / src_split / "images"
    label_dir = source_base / src_split / "labels"

    print(f"Looking in {image_dir} ...")

    if not image_dir.exists() or not label_dir.exists():
        print(f"⚠️ Missing expected folders in {src_split}, skipping.")
        continue

    for img_path in image_dir.glob("*.*"):
        if img_path.suffix.lower() not in [".jpg", ".jpeg", ".png"]:
            continue

        label_path = label_dir / (img_path.stem + ".txt")

        if label_path.exists():
            image_label_pairs.append((img_path, label_path))
        else:
            print(f"⚠️ No label for {img_path.name}")

print(f"Found {len(image_label_pairs)} total pairs.")

# Only continue if we found images
if len(image_label_pairs) == 0:
    raise SystemExit("No image-label pairs found. Check folder structure!")

# Shuffle + split
random.shuffle(image_label_pairs)
n = len(image_label_pairs)
train_split = int(n * split_ratios[0])
val_split = int(n * (split_ratios[0] + split_ratios[1]))

splits_data = {
    'train': image_label_pairs[:train_split],
    'val': image_label_pairs[train_split:val_split],
    'test': image_label_pairs[val_split:]
}

# Copy files
for split, pairs in splits_data.items():
    for img_src, lbl_src in pairs:
        img_dest = dest_base / "images" / split / img_src.name
        lbl_dest = dest_base / "labels" / split / lbl_src.name
        shutil.copyfile(img_src, img_dest)
        shutil.copyfile(lbl_src, lbl_dest)

print(f"✅ Dataset prepared under '{dest_base}' with {len(image_label_pairs)} samples.")
