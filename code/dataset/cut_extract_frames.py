import tkinter as tk
from tkinter import filedialog
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
from pathlib import Path
import ffmpeg
import cv2

def select_file():
    # Open a file dialog and get the selected file's path
    file_path = filedialog.askopenfilename(title="Select a video file to cut")
    
    # If a file is selected, display the file path
    if file_path:
        print(f"Selected file: {file_path}")
        return(file_path)
    else:
        print("No file selected")

# Minutes and seconds to seconds
def minsec2sec(x):
    return int(x.split(":")[0])*60+int(x.split(":")[1])


def cut_vid(file_path, savepath, addseconds): 

    video = Path(file_path)
    startclip = "00:"+input("Start time (min:sec): ")
    endclip = "00:"+input("End time (min:sec): ")
    startclip_n = startclip.replace(":", "")
    endclip_n = endclip.replace(":", "")
    freq = int(input("Sampling frequency: "))
    filename_out = f'{savepath}{video.stem}_{startclip_n}_{endclip_n}.mp4'
        
    if 0 < 1: 
            (
            ffmpeg.input(file_path, ss=startclip, to=endclip)
            .output(filename_out)
            .run()
            )
        
    print(f"{filename_out} cut!")
    return(filename_out, freq)


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
    # Create the main window (it won't be shown, we just need it for the dialog)
    root = tk.Tk()
    root.withdraw()  # Hide the main window
    
    # Call the function to open the file dialog
    filename = select_file()
    videoname, freq = cut_vid(filename, "/Users/jonas/Downloads/vid/", 2)
    #save_frames(videoname, "/Users/jonas/Downloads/ims/", freq)


if __name__ == "__main__":
    main()


