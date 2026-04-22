import yaml
import shutil
import random
from pathlib import Path

# Example usage:
# Use multiple source directories? (y/n, default: n): y
# Enter source directories (one per line, empty line to finish):
# > Ring_detection-5
# > Tag_detection-3
# >
# Enter output dataset version (e.g., 4564): 1001
# Enter dataset prefix (e.g., tag_detection, ring_detection): combined_detection
#
# Or single directory mode:
# Use multiple source directories? (y/n, default: n): n
# Enter roboflow dataset version number (e.g., 1): 5
# Enter dataset version (e.g., 4564): 1001
# Enter dataset prefix (e.g., tag_detection, ring_detection): tag_detection

# Choose between single and multiple directories
use_multiple = input("Use multiple source directories? (y/n, default: n): ").strip().lower() == 'y'

if use_multiple:
    # Collect source directories from user
    source_bases = []
    print("Enter source directories (one per line, empty line to finish):")
    while True:
        source_dir = input("> ").strip()
        if not source_dir:
            break
        source_bases.append(Path(source_dir))

    if not source_bases:
        raise SystemExit("No source directories provided!")

    ds_version = input("Enter output dataset version (e.g., 4564): ").strip()
    ds_prefix = input("Enter dataset prefix (e.g., tag_detection, ring_detection): ").strip()
else:
    # Single directory mode (original behavior)
    base = input("Enter roboflow dataset version number (e.g., 1): ").strip()
    ds_version = input("Enter dataset version (e.g., 4564): ").strip()
    ds_prefix = input("Enter dataset prefix (e.g., tag_detection, ring_detection): ").strip()
    source_bases = [Path(f"Tag_detection-{base}")]

source_splits = ['train', 'valid', 'test']

# Destination base folder
dest_base = Path(f"dataset/{ds_prefix}_{ds_version}")
dest_splits = ['train', 'val', 'test']
split_ratios = [0.85, 0.1, 0.05]

# Make destination directories
for split in dest_splits:
    (dest_base / "images" / split).mkdir(parents=True, exist_ok=True)
    (dest_base / "labels" / split).mkdir(parents=True, exist_ok=True)

# Collect image-label pairs from all sources
image_label_pairs = []

for source_base in source_bases:
    print(f"\nProcessing source: {source_base}")

    for src_split in source_splits:
        image_dir = source_base / src_split / "images"
        label_dir = source_base / src_split / "labels"

        print(f"  Looking in {image_dir} ...")

        if not image_dir.exists() or not label_dir.exists():
            print(f"  ⚠️ Missing expected folders in {src_split}, skipping.")
            continue

        for img_path in image_dir.glob("*.*"):
            if img_path.suffix.lower() not in [".jpg", ".jpeg", ".png"]:
                continue

            label_path = label_dir / (img_path.stem + ".txt")

            if label_path.exists():
                image_label_pairs.append((img_path, label_path))
            else:
                print(f"  ⚠️ No label for {img_path.name}")

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



# Create yaml file for YOLO training based on template
data = {
    "train": f"{ds_prefix}_{ds_version}/images/train",
    "val": f"{ds_prefix}_{ds_version}/images/val",
    "test": f"{ds_prefix}_{ds_version}/images/test",
    "nc": 1,
    "names": ["text"]
}

# Save to YAML file
with open(f"dataset/dataset_{ds_prefix}_{ds_version}.yaml", "w") as f:
    yaml.dump(data, f, default_flow_style=False, sort_keys=False)

