import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import sys 


# Read data for one station at the time: 
stat = sys.argv[1]
out = pd.read_csv(f"data/eider2024_nanov5852_{stat}_v11.csv")
out = out[out["station"] == stat]
out["datetime"] = pd.to_datetime(out["datetime"])

# Full date time sequence
date_rng = pd.date_range(start=out["datetime"].min(), end=out["datetime"].max(), freq='5s')

# Plot time series of one station at the time, with datetime as time series and bars of bars for counts
fig, ax = plt.subplots(5, 1)

plt.suptitle(stat)

for i in range(0, 5):
    dx2 = out[out["class"] == (i)]
    label = dx2["name"].iloc[0] # Which class is it (eider female etc.)?

    dx2 = dx2.drop_duplicates(subset='datetime')

    # Subset based on confidence level
    dx2 = dx2[dx2["conf"] > 0.6]

    # Fill missing values
    dx2 = dx2.set_index('datetime').reindex(date_rng, fill_value=0).reset_index().rename(columns={'index': 'datetime'})

    # Create a new y series which is a 10 point running mean of the original y series
    y = dx2["counts"]/125
    y_rolling = y.rolling(window=10).mean()

    ax[i].plot(dx2["datetime"], y, c = "red", alpha = 0.4)
    ax[i].plot(dx2["datetime"], y_rolling, color = "black", alpha = 0.9) 

    ax[i].annotate(label,
        xy=(0, 1), xycoords='axes fraction',
        xytext=(+0.5, -0.5), textcoords='offset fontsize',
        fontsize='medium', verticalalignment='top', fontfamily='serif',
        bbox=dict(facecolor='0.7', edgecolor='none', pad=3.0))
plt.show()
