import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Load the data
data_path = Path("results/run1_clean_per_second.csv")
event_path = Path("results/run1_clean_events.csv")
events = pd.read_csv(event_path)
data = pd.read_csv(data_path)

# Remove 0s 
data = data[data["count_adult"] != 0]

# Plot time series of one station at the time, with datetime as time series and bars of bars for counts
fig, ax = plt.subplots(1, 1)

ax.plot(data["second"], data["count_adult"]) 
ax.bar(events["second"], height = 1)

plt.show()