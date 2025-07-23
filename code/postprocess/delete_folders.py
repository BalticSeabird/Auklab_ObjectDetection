



from pathlib import Path
import shutil


# Define input video path
base_path = Path(f"../../../../../../mnt/BSP_NAS2_vol3/Video/Video2024/EJDER13PATH/")
# Get only directories

#directories = [p for p in base_path.iterdir() if p.is_dir()]
directories = [d for d in base_path.rglob("*eaDir") if d.is_dir()]


# Print the directories
for directory in directories:
    shutil.rmtree(directory)
    print(f"Deleted {directory}")


