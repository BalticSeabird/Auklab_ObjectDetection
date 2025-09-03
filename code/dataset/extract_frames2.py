import tkinter as tk
from tkinter import filedialog
from pathlib import Path
import cv2
import subprocess

# Calculate number of frames in video
def get_frame_count(input_video):
    cap = cv2.VideoCapture(input_video)
    if not cap.isOpened():
        print("Error: Could not open the input video file")
        return 0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    print(f"Length of {input_video.stem} (frames): {frame_count}")
    return frame_count

# Read a video save frames as images
def save_frames(input_video, image_folder, freq):
    vidname = Path(input_video).stem
    cap = cv2.VideoCapture(input_video)
    if not cap.isOpened():
        print("Error: Could not open the input video file")
        exit()
    count = 0
    while(cap.isOpened()):
        ret, frame = cap.read()
        if not ret:
            break

# If the current frame is a multiple of n, save it
        if count % freq == 0:
            countnum = str(count).zfill(4)                
            cv2.imwrite(f'{image_folder}/{vidname}_{countnum}.png', frame)
            print(f"Saved: frame {countnum}")

        count += 1
    cap.release()
    cv2.destroyAllWindows()


def main():
    for file in Path("/Users/jonas/Downloads/vid/").glob("*.mp4"):
        input_video = Path(file)
        vidlength = get_frame_count(input_video)
        print(f"Video length (frames): {vidlength}")
        frame_output_target = 100
        freq = max(1, vidlength // frame_output_target)
        save_frames(input_video, "/Users/jonas/Downloads/ims/", freq)

if __name__ == "__main__":
    main()

