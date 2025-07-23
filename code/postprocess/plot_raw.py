import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Load the data
data_path = Path("/Users/jonas/Documents/research/seabird/ejder/eider_detection/EJDER7/NVR_Hien_EJDER7_2024-05-12_18.00.00.csv")
out = pd.read_csv(data_path)

d = {"name": ['crow', 'eider_female', 'eider_male', 'gull', 'razorbill'], 
    "class": [0, 1, 2, 3, 4]}
class_id = pd.DataFrame(d)    
out = out.merge(class_id, on = "class")
out["datetime"] = pd.to_datetime(out["datetime"])
out.sort_values(by = ["datetime"], inplace = True) 

# Plot time series of one station at the time, with datetime as time series and bars of bars for counts
fig, ax = plt.subplots(5, 1)

# Subset data for one station at the time: 
stat = "EJDER7"
dx = out[out["station"] == stat]
plt.suptitle(stat)

for i in range(0, 5):
    dx2 = dx[dx["class"] == (i)]
    ax[i].scatter(dx2["datetime"], dx2["conf"]) 
    label = dx2["name"].iloc[0]
    ax[i].annotate(label,
        xy=(0, 1), xycoords='axes fraction',
        xytext=(+0.5, -0.5), textcoords='offset fontsize',
        fontsize='medium', verticalalignment='top', fontfamily='serif',
        bbox=dict(facecolor='0.7', edgecolor='none', pad=3.0))
plt.show()