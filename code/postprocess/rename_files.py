
from pathlib import Path
import os
import sys

station = sys.argv[1]

# Define input video path
base_path = Path(f"../../../../../../mnt/BSP_NAS2_work/eider_model/inference/2024/eider_model_nano_v5852/{station}")

raw = list(base_path.rglob("*raw.csv"))
grouped = list(base_path.rglob("*_grouped5s.csv"))
all = list(base_path.rglob("*.csv"))

notchange = raw+grouped
change = [file for file in all if file not in notchange]


# Delete all
for file in change:
    newname = file.parent.joinpath(file.stem + "_raw.csv")
    os.rename(file, newname) 
    print(f"renamed = {file.stem}")

