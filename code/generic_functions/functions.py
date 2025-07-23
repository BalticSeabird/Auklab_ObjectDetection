

import pandas as pd
from pathlib import Path
import os
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
import cv2
import numpy as np
from ultralytics import YOLO
import yaml


import smtplib
from email.message import EmailMessage
from datetime import datetime


def send_email(password, time, device, station, filename, start = True):

    # Start or end of run
    if start:
        subject = f"Run of {station} on GPU {device} started {time} !"
        body = f"First video = ..."
    else:
        subject = f"Run of {station} on GPU {device} completed {time} !"
        body = f"Last video processed = {filename} !"
    
    # Email details
    sender_email = "jhsemailservice@gmail.com"
    recipient_email = "jonas.sundberg@slu.se"
    
    # Create the email
    msg = EmailMessage()
    msg["From"] = sender_email
    msg["To"] = recipient_email
    msg["Subject"] = subject
    msg.set_content(body)


    # Send the email
    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as smtp:
        smtp.login(sender_email, password)
        smtp.send_message(msg)

    print("Email with run update sent successfully!")


#password = input("Enter the password for the email account: ")
#now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")



# Minutes and seconds to seconds
def minsec2sec(x):
    return int(x.split(":")[0])*60+int(x.split(":")[1])


def cut_vid_simpler(video_dir, row, savepath, addseconds): 

    # Build path to video
    video = row["filepath"]
    print(video)
    video_station = video.split("_")[-3]
    video_date = video.split("_")[-2]

    full_path = f'{video_dir}{video_station}/{video_date}/{video}'
    print(full_path)

    startclip = row["minute_second_start"]
    endclip = row["minute_second_end"]

    if any(pd.isnull([startclip, endclip, video])):
        print("skip")

    else: 
        startsec = minsec2sec(startclip)-addseconds
        endsec = minsec2sec(endclip)+addseconds

        # Cut down long videos to max 2 minutes
        length = endsec - startsec

        if length > 120: 
            endsec = startsec + 120

        if os.path.isfile(full_path):
            filename_out = f"{savepath}{Path(video).stem}_{startsec}_{endsec}.mp4"
            ffmpeg_extract_subclip(
                full_path,
                startsec,
                endsec+startsec,
                filename_out
            )
            return(filename_out)
            print(f'cut {full_path} from {startsec} to {endsec}')
        else: 
            print("file not found")




# Read a video save frames as images
def save_frames(input_video, image_folder, freq):
    vidname = Path(input_video).stem
    cap = cv2.VideoCapture(input_video)
    if not cap.isOpened():
        print("Error: Could not open the input video file")
        pass
    else: 
        count = 0
        while(cap.isOpened()):
            ret, frame = cap.read()
            if not ret:
                break

    # If the current frame is a multiple of n, save it
            if count % freq == 0:
                countnum = str(count).zfill(4)                
                cv2.imwrite(f'{image_folder}/{vidname}_{countnum}.png', frame)
                print(f"Saved: frame {countnum} from {vidname}")

            count += 1
        cap.release()
        cv2.destroyAllWindows()


# Function for looking through images in a folder and remove those that are very similar to the previous one
def remove_similar_images(folder, similarity_thresh):
    files = list(Path(folder).glob("*.png"))
    files.sort()
    remove = []
    for i in range(0, len(files)-1):
        print(f'reading {files[i+1]}')
        img1 = cv2.imread(files[i])
        img2 = cv2.imread(files[i+1])
        dist = euclidean_images(img1, img2)
        print(f'distance to {files[i]} = {dist}')
        if dist < similarity_thresh:
            remove.append(files[i+1])
            print(f'added {files[i+1]} to remove list')
        else: 
            pass
    return remove




def euclidean_images(img1, img2):
    
    # Flatten the images
    image1_flat = img1.flatten()
    image2_flat = img2.flatten()
    
    # Compute the Euclidean distance
    distance = np.linalg.norm(image1_flat - image2_flat)
    return distance



def annotate_images(yolo_model, im_outfold, yaml_outfold):

    # Load a pretrained YOLO model
    model = YOLO(yolo_model)

    # List of videos for inference 
    ims = os.listdir(im_outfold)

    # Run
    for im in ims: 

        if len(im) > 20:

            results = model(f'{im_outfold}/{im}')

            # Width and height
            imread = cv2.imread(f'{im_outfold}/{im}')
            width = imread.shape[1]
            height = imread.shape[0]

            # Process results list
            boxes = []
            boxesxyxy = []
            classes = []
            confs = []

            for r in results:
                boxes.append(r.boxes.xywh.tolist()) 
                boxesxyxy.append(r.boxes.xyxy.tolist())
                classes.append(r.boxes.cls.tolist())
                confs.append(r.boxes.conf.tolist())

            # Concatenate outputs
            boxesx = sum(boxes, [])
            boxesxyxy2 = sum(boxesxyxy, [])

            # Save as data frames
            nobj = len(boxesx)

            filename = im.replace(".jpg", ".txt")
            filename_simpl = im.replace(".png", "")

            d = {"name": ['crow', 'eider_female', 'eider_male', 'gull', 'razorbill'], 
                "class": [0, 1, 2, 3, 4]}
            class_ids = pd.DataFrame(d)    

            # .yaml
            # Always in file
            data_dict = {}
            data_dict["image"] = filename
            data_dict["size"] = {"depth": 3, "height": height, "width": width}
            data_dict["source"] = {"framenumber": 0, "path": "na", "video": "na"}
            data_dict["state"] = {"verified": False, "warnings": 0}


            if nobj > 0:

                data_dict["objects"] = []

                for row in range(0, nobj):
                    print(f'classes = {classes}')
                    print(f'nobj = {nobj}')
                    print(f'row = {row}')
                    tdat = boxesxyxy2[row]
                    #classdat = row #works
                    classdat = classes[0][row]
                    print(f'classdat = {classdat}')
                    classname = class_ids[class_ids["class"] == classdat]["name"].item()
                    print(f'classname = {classname}')
                    #confdat = confs[row]
                    
                    data_dict["objects"].append(
                        {
                            "bndbox": {
                                "xmax": tdat[2],
                                "xmin": tdat[0],
                                "ymax": tdat[3],
                                "ymin": tdat[1],
                            },
                            "name": classname

                        }
                    )
            
            write_yaml_to_file(yaml_outfold, data_dict, filename_simpl)


            # Plain annotation
            #if nobj > 0:

                #y = np.empty([nobj, 5], dtype = float)
                #for row in range(0, nobj):
                #    y[row, 1] = (boxesx[row][0]+(.5*boxesx[row][2]))/width # x 
                #    y[row, 2] = (boxesx[row][1]+(.5*boxesx[row][3]))/height # y 
                #    y[row, 3] = (boxesx[row][2])/width # w 
                #    y[row, 4] = (boxesx[row][3])/height # h 
                
                #np.savetxt(f'../dataset/annotations/{filename}', y, fmt="%i %1.4f %1.4f %1.4f %1.4f")
            
            #else:
             #   open(f'../dataset/annotations/{filename}', 'a').close()

def write_yaml_to_file(yaml_outfold, py_obj,filename_simpl):
    with open(f'{yaml_outfold}{filename_simpl}.yaml', 'w',) as f :
        yaml.dump(py_obj,f,sort_keys=False) 



