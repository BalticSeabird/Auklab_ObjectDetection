import pandas as pd
from pathlib import Path 
from ultralytics import YOLO
import os 
import sys
sys.path.append("/home/jonas/Documents/vscode/Eider_detection/code/generic_functions/") # Sprattus
from functions import send_email

# Input arguments (run device and station)
device = sys.argv[1]
stat = sys.argv[2]
password = input("Please type gmail password ... ")
datelimit = pd.to_datetime("2020-05-11 12:00:00")

# Send start email
now = pd.to_datetime("now").strftime("%Y-%m-%d %H:%M:%S")
filename = "none"
send_email(password, now, device, stat, filename, start = True)


# Load a pretrained YOLO model
modelpath = Path("models/eider_model_nano_v5852.pt")
model = YOLO(modelpath)
modelname = modelpath.stem
output_dir1 = f'../../../../../../mnt/BSP_NAS2_work/eider_model/inference/2024/{modelname}/'

if os.path.exists(output_dir1) == False:
    os.makedirs(output_dir1)

# Output folder
output_dir2 = output_dir1+stat

if os.path.exists(output_dir2) == False:
    os.makedirs(output_dir2)

# Define input video path
base_path = Path(f"../../../../../../mnt/BSP_NAS2_vol3/Video/Video2024/{stat}")
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

        try:

            starttime = pd.to_datetime(time)
            starttime_u = starttime.timestamp()
            fps = 25

            outname = output_dir2+"/"+vid.stem+"_raw.csv"
            #outname_grouped = output_dir2+"/"+vid.stem+"_grouped.csv"

            # Check that file has not been processed already 

            if os.path.exists(outname):
                print(f"File {outname} already exists")
                continue

            else: 
                # Run inference using the pretrained model and the inoutvid video
                results = model.predict(vid, 
                                    stream=True, 
                                    save = False,
                                    show = False, 
                                    device = device)

                # Process results list
                time = []
                boxes = []
                #track_ids = []
                confs = []
                classes = []
                framenum = []
                counter = starttime_u
                counterx = 0

                for r in results:
                    if not r.boxes.conf.nelement() == 0 :
                        boxes.append(r.boxes.xyxy.tolist())
                        #if r.boxes.id is not None:
                        #    track_ids.append(r.boxes.id.tolist())
                        #else:
                        #    track_ids.append([-1 for _ in r.boxes])
                        classes.append(r.boxes.cls.tolist())
                        confs.append(r.boxes.conf.tolist())
                        ndetect = len(r.boxes.conf.tolist())
                        time.append([counter] * ndetect)
                        framenum.append([counterx] * ndetect)

                    counter += 1/fps
                    counterx += 1

                # Concatenate outputs
                conf = sum(confs, [])
                classes = sum(classes, [])
                boxesx = sum(boxes, [])
                #track_ids = sum(track_ids, [])
                times = sum(time, [])
                framenums = sum(framenum, [])

                # Save as data frames
                #out1 = pd.DataFrame(boxesx, columns = ["x", "y", "w", "h"])
                out2 = pd.DataFrame(list(zip(classes, conf, times, framenums)), columns = ["class", "conf", "time", "frame"])

                #out = out1.merge(out2, left_index = True, right_index = True)
                out2["station"] = station
                out2["filename"] = filename

                # Actual time
                out2["datetime"] = pd.to_datetime(out2["time"], unit = "s")

                # Group by second
                #grouped_data = out2.groupby([pd.Grouper(key='datetime', freq='2s'), "class"])

                # Aggregate grouped_data (mean confidence score)
                #grouped = grouped_data.agg({"conf": "mean", 
                #                "frame": "count"}).reset_index()        

                out2.to_csv(outname, index = False)
                #grouped.to_csv(outname_grouped, index = False)
        except: 
            print(f"Error with {vid} (probably file name), continue loop")
    else:   
        continue


# Send end email
now = pd.to_datetime("now").strftime("%Y-%m-%d %H:%M:%S")
send_email(password, now, device, stat, filename, start = False)


# Run example 
# python3 code/model/run_inference.py 1 "EJDER4"