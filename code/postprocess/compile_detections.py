
import pandas as pd
from pathlib import Path
import sys

# Define folder 
year = sys.argv[1]

# Read folders 
data_path = Path(f"../../../../../../mnt/BSP_NAS2_work/eider_model/inference/{year}/eider_model_nano_v5852")
stations = list(data_path.glob("*"))

for station in stations:

    files = list(data_path.joinpath(station).rglob("*grouped5s.csv"))

    # Compile data files
    out = pd.DataFrame()
    for file in files:
        temp = pd.read_csv(file)
        temp["station"] = file.name.split("_")[-4]
        temp.rename(columns = {"datetime": "datetime", "class": "class", "conf": "conf", "frame": "counts"}, inplace = True)
        out = pd.concat([out, temp])
        print(f'compiling ... {file.name}')

    out = out[["datetime", "class", "conf", "counts", "station"]]

    d = {"name": ['crow', 'eider_female', 'eider_male', 'gull', 'razorbill'], 
        "class": [0, 1, 2, 3, 4]}
    class_id = pd.DataFrame(d)    

    out = out.merge(class_id, on = "class")

    out["datetime"] = pd.to_datetime(out["datetime"], format = "mixed")

    out.sort_values(by = ["datetime"], inplace = True) 

    out.to_csv(f"../../../../../../mnt/BSP_NAS2_work/eider_model/inference/eider{year}_nanov5852_{station.name}_v12.csv", index = False)

    print(f'example rows: {out.head()}')