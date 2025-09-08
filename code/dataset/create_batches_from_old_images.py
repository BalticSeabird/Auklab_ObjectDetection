import os
import random
import shutil
from math import ceil

# Paths
main_dir = "../../../../../mnt/BSP_NAS2/Software_Models/fish_model/annotations/"
source = os.path.join(main_dir, "train/images")
destination = "../../../../../mnt/BSP_NAS2_work/seabird_fish_model/un_annotated/"

# Parameters
batch_size = 500

# Ensure destination exists
os.makedirs(destination, exist_ok=True)

# List all files in source (only images)
all_files = [f for f in os.listdir(source) if os.path.isfile(os.path.join(source, f))]

# Random shuffle instead of sorting
random.shuffle(all_files)

# Total number of batches
num_batches = ceil(len(all_files) / batch_size)
print(f"Found {len(all_files)} images. Creating {num_batches} batches of {batch_size} images each.")

# Process files into batches
for i in range(num_batches):
    batch_folder = os.path.join(destination, f"batch_{i+1:04d}")  # e.g. batch_0001, batch_0002...
    os.makedirs(batch_folder, exist_ok=True)
    
    batch_files = all_files[i*batch_size:(i+1)*batch_size]
    
    for f in batch_files:
        src_path = os.path.join(source, f)
        dst_path = os.path.join(batch_folder, f)
        shutil.copy2(src_path, dst_path)  # copy2 keeps metadata
    
    print(f"Created {batch_folder} with {len(batch_files)} images.")

print("✅ Done! All images have been randomly distributed into batches.")


# Parameters
batch_size = 500

# Ensure destination exists
os.makedirs(destination, exist_ok=True)

# List all files in source (only images)
all_files = [f for f in os.listdir(source) if os.path.isfile(os.path.join(source, f))]
all_files.sort()  # Optional: keeps files ordered

# Total number of batches
num_batches = ceil(len(all_files) / batch_size)
print(f"Found {len(all_files)} images. Creating {num_batches} batches of {batch_size} images each.")

# Process files into batches
for i in range(num_batches):
    batch_folder = os.path.join(destination, f"batch_{i+1:04d}")  # e.g. batch_0001, batch_0002...
    os.makedirs(batch_folder, exist_ok=True)
    
    batch_files = all_files[i*batch_size:(i+1)*batch_size]
    
    for f in batch_files:
        src_path = os.path.join(source, f)
        dst_path = os.path.join(batch_folder, f)
        shutil.copy2(src_path, dst_path)  # copy2 keeps metadata
    
    print(f"Created {batch_folder} with {len(batch_files)} images.")

print("✅ Done! All images have been copied into batches.")
