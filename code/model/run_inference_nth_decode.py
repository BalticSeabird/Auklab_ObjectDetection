import pandas as pd
from pathlib import Path 
from ultralytics import YOLO
import os 
import sys
sys.path.append("/home/jonas/Documents/vscode/Eider_detection/code/generic_functions/") # Sprattus
#from functions import send_email

# Input arguments (run device and station)
device = sys.argv[1]
stat = sys.argv[2]
#password = input("Please type gmail password ... ")
datelimit = pd.to_datetime("2019-05-11 12:00:00") # Start date...

# Send start email
#now = pd.to_datetime("now").strftime("%Y-%m-%d %H:%M:%S")
#filename = "none"
#send_email(password, now, device, stat, filename, start = True)

# Settings for batch processing
frame_skip = 25  # Process every 25th frame
batch_size = 32  # Send 32 frames at a time to the GPU


# Load a pretrained YOLO model
modelpath = Path("models/auklab_model_combined_xlarge_4211_v1.pt")
model = YOLO(modelpath).to(f'cuda:{device}')
modelname = modelpath.stem
output_dir1 = f'../../../../../../mnt/BSP_NAS2_work/auklab_model/inference/2025/{modelname}/'

if os.path.exists(output_dir1) == False:
    os.makedirs(output_dir1)

# Output folder
output_dir2 = output_dir1+stat

if os.path.exists(output_dir2) == False:
    os.makedirs(output_dir2)

# Define input video path
base_path = Path(f"../../../../../../mnt/BSP_NAS2_vol4/Video/Video2025/{stat}")
vids = list(base_path.rglob("*.mp4"))
vids.sort()


# Remove strange files with ""eadir" in the name
vids = [vid for vid in vids if "@eaDir" not in str(vid)]


for vid in vids: 
    filename = vid.name
    datevid = pd.to_datetime(vid.parents[0].name)

    if datevid > pd.to_datetime(datelimit):

        # Pick out relevant information from name
        name = filename.split("_")
        time = name[-2]+" "+name[-1][0:8]
        time = time.replace(".", ":")
        station = stat

        starttime = pd.to_datetime(time)
        starttime_u = starttime.timestamp()
        fps = 25

        outname = output_dir2+"/"+vid.stem+"_raw.csv"

        # Check that file has not been processed already 

        if os.path.exists(outname):
            print(f"File {outname} already exists")
            continue

        else: 
            
            container = av.open(vid)
            stream = container.streams.video[0]
            stream.thread_type = 'AUTO'

            results_list = []
            frame_buffer = []
            frame_indices = []
            frame_count = 0
            output_count = 0

            def process_batch(frames, indices):
                with torch.no_grad():
                    results = model(frames)
                    for result, idx in zip(results, indices):
                        for box in result.boxes:
                            results_list.append({
                                'frame': idx,
                                'class': result.names[int(box.cls)],
                                'confidence': float(box.conf.cpu()),
                                'xmin': float(box.xyxy[0][0].cpu()),
                                'ymin': float(box.xyxy[0][1].cpu()),
                                'xmax': float(box.xyxy[0][2].cpu()),
                                'ymax': float(box.xyxy[0][3].cpu()),
                            })

            # Read and process every nth frame
            for frame in container.decode(stream):
                if frame_count % frame_skip == 0:
                    img = frame.to_ndarray(format='bgr24')  # OpenCV-style numpy array
                    frame_buffer.append(img)
                    frame_indices.append(frame_count)

                    if len(frame_buffer) == batch_size:
                        process_batch(frame_buffer, frame_indices)
                        frame_buffer = []
                        frame_indices = []
                        torch.cuda.empty_cache()

                frame_count += 1


            # Process remaining frames
            if frame_buffer:
                process_batch(frame_buffer, frame_indices)

            # Save results
            df = pd.DataFrame(results_list)
            df.to_csv(outname, index=False)
            print(f"Saved {len(df)} detections to {outname}")


# Send end email
now = pd.to_datetime("now").strftime("%Y-%m-%d %H:%M:%S")
#send_email(password, now, device, stat, filename, start = False)


# Run example 
# python3 code/model/run_inference_nth_decode.py 1 "FAR3"