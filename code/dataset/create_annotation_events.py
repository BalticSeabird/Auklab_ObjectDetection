
from pathlib import Path
import os
import pandas as pd
import matplotlib.pyplot as plt
import sys 
import numpy as np

# Read data
out = pd.read_csv("data/compiled_nanov5852_v4.csv")
out["datetime"] = pd.to_datetime(out["datetime"])

stations = ["EJDER1", "EJDER2", "EJDER4", "EJDER5", "EJDER6", "EJDER7", "EJDER8", "EJDER12",  "EJDER13PATH"]

for stat in stations:

# One station at the time
    cond1 = out["station"] == stat
    cond2  = out["class"] != 1
    dx = out[cond1 & cond2]

    # Summarize frame counts for all classes combined except for class 1
    dx2 = dx.groupby("datetime").sum(["frame"]).reset_index()

    # Subset based on confidence level
    #dx2 = dx2[dx2["conf"] > 0.3]

    # Full date time sequence
    date_rng = pd.date_range(start=dx2["datetime"].min(), end=dx2["datetime"].max(), freq='2s')

    # Fill missing values
    dx2 = dx2.set_index('datetime').reindex(date_rng, fill_value=0).reset_index().rename(columns={'index': 'datetime'})

    # Plot time series of one station at the time, with datetime as time series and bars of bars for counts
    threshold = 0.75
    y = dx2["counts"]/50
    y_rolling = y.rolling(window=30).mean()


    # CREATE State machine
    dx2["y_rolling"] = y_rolling
    state = np.where(y_rolling > threshold, 1, 0)

    # Make a variable which indicate direction of state change
    state_change = np.diff(state)
    state_change2 = np.where(state_change == 1, 1, 0)
    state_change = list(state_change)
    state_change2 = list(state_change2)

    # Add a 0 to the beginning of the list
    state_change.insert(0, 0)
    state_change2.insert(0, 0)

    # Assign a unique id to each event
    event_id = np.cumsum(state_change2)

    dx2["state"] = state
    dx2["state_change"] = state_change
    dx2["event_id"] = event_id
    dx2["event_id"] = np.where(dx2["state" ] == 0, 0, dx2["event_id"])

    # Loop through each event and save start and end time and duration
    events = []
    for i in range(1, event_id.max() + 1):
        dx3 = dx2[dx2["event_id"] == i]
        start = dx3["datetime"].min()
        end = dx3["datetime"].max()
        duration = end - start
        # Duration in seconds
        duration = duration.total_seconds()
        date = format(start, "%Y-%m-%d")
        hour = format(start, "%H")
        minute_second_start = format(start, "%M:%S")
        minute_second_end = format(end, "%M:%S")
        filepath = f'EjderNVR_{stat}_{date}_{hour}.00.00.mp4'
        # New even id with station prefix
        id = f"nanov5852_{stat}_{date}_{i}"

        events.append([id, start, end, duration, minute_second_end, minute_second_start, filepath])
        
    # Create data frame of events
    events = pd.DataFrame(events, columns = ["event_id", "start", "end", "duration", "minute_second_end", "minute_second_start", "filepath"])
    events["station"] = stat

    events.to_csv(f"data/events/events_{stat}.csv", index = False)
