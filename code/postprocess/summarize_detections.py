import pandas as pd
import sys

# Read data
out = pd.read_csv("../../../../../../mnt/BSP_NAS2_work/eider_model/inference/eider2024_nanov5852_v8.csv", parse_dates = ["datetime"])
out["date"] = out["datetime"].dt.date

# Select station
stat = sys.argv[1]
out = out[out["station"] == stat]

# Select confidence level
conf = 0.6
out = out[out["conf"] > conf]

# Summary statistics in a table with date as index and class as columns, and counts as values
summary = out.groupby(["date", "name"]).agg({"counts": "sum"}).unstack().fillna(0)
summary.columns = summary.columns.droplevel(0)  
summary = summary.reset_index()
print("")
print(f"Summary statistics for station {stat}")
print("")
print(summary)
print("")



# Subset data for one station at the time: 





