import pandas as pd
from pathlib import Path 
from ultralytics import YOLO
import os 
import sys
import cv2

# Load a pretrained YOLO model
modelpath = Path("models/eider_model_nano_v5852.pt")
model = YOLO(modelpath)
modelname = modelpath.stem

vids = [Path("../../../../../../mnt/BSP_NAS2_work/eider_model/eider_testvids/EjderNVR_EJDER1_2024-04-28_04.00.00_001425_001625.mp4"), 
           Path("../../../../../../mnt/BSP_NAS2_work/eider_model/eider_testvids/EjderNVR_EJDER6_2024-05-03_08.00.00_001015_001215.mp4")]  
#vids = list(Path("../../../../../../mnt/BSP_NAS2_work/eider_model/eider_testvids/").rglob("*.mp4"))

# Run inference using the pretrained model and the inout video

for vid in vids: 

    cap = cv2.VideoCapture(vid)

    # Read the first frame to get the frame size
    success, frame = cap.read()
    if not success:
        print(f"Failed to read the video: {vid}")
        continue

    # Get the frame size
    frame_height, frame_width = frame.shape[:2]

    name = "video_out/" + vid.stem + "_" + modelname + ".mp4"
    
    if Path(name).exists():
        print(f"Video {name} already exists")
        continue
    
    else: 
        
        writer = cv2.VideoWriter(name, cv2.VideoWriter_fourcc(*'mp4v'), 10.0, (frame_width, frame_height))

        # Loop through the video frames
        while cap.isOpened():
            # Read a frame from the video
            success, frame = cap.read()

            if success:
                # Run YOLOv8 tracking on the frame, persisting tracks between frames
                results = model.predict(frame, 
                            stream=True, 
                            save = True,
                            show = False, 
                            save_frames = False)

                # Iterate over the results generator
                for result in results:
                    # Visualize the results on the frame
                    annotated_frame = result.plot()

                    # Write the annotated frame to the video writer
                    writer.write(annotated_frame)

            else:
                # Break the loop if the end of the video is reached
                break

        # Release the video capture and writer objects, and close the display window
        cap.release()
        writer.release()
        cv2.destroyAllWindows()




