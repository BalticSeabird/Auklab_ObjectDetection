import os
import shutil
import random
from glob import glob
from pathlib import Path

# Source folders
source_folder = Path("../../../../../mnt/BSP_NAS2/Software_Models/Seabirds_AI/Annotations/Seabird_detection/")
source_dirs = ['seabird1', 'seabird2', 'seabird3', 'seabird4']


# Destination base folder
dest_base = 'dataset'
splits = ['train', 'val', 'test']
split_ratios = [0.8, 0.1, 0.1]

# Make destination directories
for split in splits:
    os.makedirs(f'{dest_base}/images/{split}', exist_ok=True)
    os.makedirs(f'{dest_base}/labels/{split}', exist_ok=True)

# Collect all image-label pairs
image_label_pairs = []
for source in source_dirs:
    longpath = source_folder.joinpath(source)
    image_files = longpath.rglob('*.jpg')
    for img_path in image_files:
        img_name = Path(img_path).stem
        label_path = os.path.join(longpath, 'labels', img_name + '.txt')
        if os.path.exists(label_path):
            image_label_pairs.append((img_path, label_path))

# Shuffle and split
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
        img_dest = os.path.join(dest_base, 'images', split, os.path.basename(img_src))
        lbl_dest = os.path.join(dest_base, 'labels', split, os.path.basename(lbl_src))
        shutil.copyfile(img_src, img_dest)
        shutil.copyfile(lbl_src, lbl_dest)

print(f"Dataset prepared under '{dest_base}' with {len(image_label_pairs)} samples.")

