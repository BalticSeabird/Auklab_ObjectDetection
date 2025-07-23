import tkinter as tk
from tkinter import filedialog
from pathlib import Path
import cv2
import subprocess

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


def trim_video_ffmpeg(input_path, output_path, start_time, end_time):
    """
    Trim a video using ffmpeg between start_time and end_time.

    Parameters:
    - input_path: str, path to the input video
    - output_path: str, path to save the trimmed video
    - start_time: str, start time in "MM:SS" format
    - end_time: str, end time in "MM:SS" format
    """
    # Build the ffmpeg command
    command = [
        'ffmpeg',
        '-y',  # overwrite output file without asking
        '-i', input_path,
        '-ss', start_time,
        '-to', end_time,
        '-c', 'copy',  # copy without re-encoding (VERY fast)
        output_path
    ]

    # Run the command
    subprocess.run(command, check=True)



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
    videoname = Path(filename).stem
    savepath = "/Users/jonas/Downloads/vid/"
    
    start = "00:"+input("Start time (min:sec): ")
    end = "00:"+input("End time (min:sec): ")
    savename = videoname + "_" + start.replace(":", "") + "_" + end.replace(":", "") + ".mp4"
    savepath = savepath + savename

    trim_video_ffmpeg(filename, savepath, start, end)
    #save_frames(videoname, "/Users/jonas/Downloads/ims/", freq)


if __name__ == "__main__":
    main()

