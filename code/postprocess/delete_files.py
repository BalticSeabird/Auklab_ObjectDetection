
from pathlib import Path
import os
import sys

# Define station in input
station = sys.argv[1]

# Define input video path
base_path = Path(f"../../../../../../mnt/BSP_NAS2_work/eider_model/inference/2024/eider_model_nano_v5852/{station}")

vids = list(base_path.rglob("*grouped.csv"))

# Delete all
for vid in vids:
    os.remove(vid)
    print(f"Deleted {vid}")

