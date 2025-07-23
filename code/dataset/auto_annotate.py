
import sys

# Append paths to sys.path
sys.path.append("/home/jonas/Documents/vscode/Eider_detection/code/generic_functions/") # Sprattus
#sys.path.append("/home/jonas/Documents/vscode/ultralytics/") # Sprattus
#sys.path.append("/Users/jonas/Documents/Programming/python/ultralytics/code/generic_functions/") # Mac

# Print sys.path for debugging
print("sys.path:", sys.path)


from pathlib import Path
import os
import pandas as pd
from functions import save_frames, cut_vid_simpler, remove_similar_images, annotate_images


video_dir = "../../../../../../mnt/BSP_NAS2_vol3/Video/Video2024/"
vid_outfold = "../../../../../../mnt/BSP_NAS2_work/eider_model/auto_annotate/vids/"
im_outfold = "../../../../../../mnt/BSP_NAS2_work/eider_model/auto_annotate/ims/"
yaml_outfold = "../../../../../../mnt/BSP_NAS2_work/eider_model/auto_annotate/yaml/"
yolo_model = "../../../../../../mnt/BSP_NAS2/Software_Models/Eider_model/models/eider_model_medium_v5852.pt"


# Loop video generation across stations
event_fold_path = Path("data/events")

for station in event_fold_path.glob("*.csv"):

    print(f'processing {station}')
    
    # Read arguments
    video_meta_path = station

    # Read metadata on interesting videos
    video_meta = pd.read_csv(video_meta_path, sep=",")

    # Run video cutting
    for row in video_meta.index:
        results = cut_vid_simpler(video_dir, video_meta.loc[row], vid_outfold, 10)

# Extract frames from all vids
counter_vids = 0
for file in list(Path(vid_outfold).glob("*.mp4")):
    counter_vids += 1
    save_frames(file, im_outfold, 25)

# Remove similar images
#remove = remove_similar_images(im_outfold, 250000)
#[os.remove(file) for file in remove]


# Annotate images
results = annotate_images(yolo_model, im_outfold, yaml_outfold)

# Run example (Sprattus/Larus)
#python3 code/dataset/auto_annotate.py "../../../../../../mnt/BSP_NAS2_vol3/Video/Video2024/" "data/events_EJDER2.csv" "vids/" "images/" "data/annotations_yaml/" "../../../../../../mnt/BSP_NAS2/Software_Models/Eider_model/models/eider_model_medium_v5852.pt"

