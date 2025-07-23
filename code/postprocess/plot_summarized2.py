import pandas as pd
import matplotlib.pyplot as plt
import sys 
import hvplot.pandas

# Read data for one station at the time: 
stat = sys.argv[1]
out = pd.read_csv(f"data/eider2024_nanov5852_{stat}_v12.csv")
out = out[out["station"] == stat]
out["datetime"] = pd.to_datetime(out["datetime"])
out = out[out["conf"] > 0.6]

# Full date time sequence
date_rng = pd.date_range(start=out["datetime"].min(), end=out["datetime"].max(), freq='5s')

# Plot time series of one station at the time, with datetime as time series and bars of bars for counts
out_wide = out.pivot_table(index = "datetime", columns='name', values='counts').reset_index()
date_rng = pd.date_range(start=out_wide["datetime"].min(), end=out_wide["datetime"].max(), freq='5s')
out_wide = out_wide.set_index('datetime').reindex(date_rng, fill_value=0).reset_index()
out_wide.fillna(0, inplace=True)
out_wide.set_index('index', inplace=True)
out_wide = out_wide.rolling(window=10).mean()
out_wide.fillna(0, inplace=True)
out_wide = out_wide/125

#subcoordinate_y=True, 
px1 = out_wide.hvplot(width = 1300, height = 800)
hvplot.show(px1)

