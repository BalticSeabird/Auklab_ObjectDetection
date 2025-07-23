

from pathlib import Path
import numpy as np
import os

# Define paths
yaml = Path("../../../../../../mnt/BSP_NAS2_work/eider_model/auto_annotate/yaml/")
images = Path("../../../../../../mnt/BSP_NAS2_work/eider_model/auto_annotate/ims/")
jobs = Path("../../../../../../mnt/BSP_NAS2_work/eider_model/auto_annotate/jobs/")

# Number of files in the folder
n_files = len(list(images.glob("*.png")))

# Annotation job size
job_size = 100

# Number of jobs
n_jobs = 1 + n_files // job_size

# List all images
images = list(images.glob("*.png"))
yaml_file = [Path(f'{yaml}/{file.stem}.yaml') for file in images]

# Create vector where each element is a random number between 0 and n_jobs
job_num = np.random.randint(0, n_jobs, n_files)

# Create folders for each job
# Only create if it doesn't exist
for i in range(n_jobs):
    newdir = jobs.joinpath(f'job_{i}')
    if  newdir.is_dir():
        continue
    else:            
        os.mkdir(newdir)


# Assign each image to a job and move file
for i in range(n_files):
    images[i].rename(jobs.joinpath(f'job_{job_num[i]}/{images[i].name}'))
    yaml_file[i].rename(jobs.joinpath(f'job_{job_num[i]}/{yaml_file[i].name}'))
    print(f'{images[i]} moved to job_{job_num[i]}')

 