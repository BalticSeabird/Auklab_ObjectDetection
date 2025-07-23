
import pandas as pd
from pathlib import Path

#station = sys.argv[1]
#eider_model_nano_v5852/{station}"
path = Path(f"../../../../../../mnt/BSP_NAS2_work/eider_model/inference/2024")
files = list(path.rglob("*raw.csv"))

for file in files: 

    # Read in the file
    outname = file.parent.joinpath(f'{file.stem[0:-3]}grouped5s.csv')

    # Check if the file already exists
    if outname.exists():
        print(f"File {outname} already exists")
        continue
    else: 
        out2 = pd.read_csv(file, parse_dates = ["datetime"])
        
        # Add to the if loop to skip if the file does not exist!

        if len(out2) == 0:
            print(f"File {outname} empty ... continues")
            continue

        else: 
            print(f'processing ... {file.stem}')
            grouped_data = out2.groupby([pd.Grouper(key='datetime', freq='5s'), "class"])

            # Aggregate grouped_data (mean confidence score)
            grouped = grouped_data.agg({"conf": "mean", 
                            "frame": "count"}).reset_index()        

            grouped.to_csv(outname, index = False)

# Run example
# python3 code/postprocess/aggregate_postrun.py "EJDER5"