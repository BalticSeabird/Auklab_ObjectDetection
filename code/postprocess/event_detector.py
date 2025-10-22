"""
seabird_metrics.py

Usage:
    python3 code/postprocess/event_detector.py data/od_data/FAR3_20250630T122002_raw.csv --out_prefix results/run1 --fps 1

Expected input CSV format (header):
    frame,class,confidence,xmin,ymin,xmax,ymax

Assumptions:
    - frame is an integer frame index (0-based).
    - bbox coords are in pixel units.
    - classes include 'adult', 'chick', 'fish' (you can change names easily).
"""

import argparse
import numpy as np
import pandas as pd
from scipy.ndimage import uniform_filter1d
from collections import defaultdict
import os
import re
from datetime import datetime, timedelta

# --------------------------
# Utilities
# --------------------------

def extract_timestamp_from_filename(csv_path):
    """
    Extract timestamp from filename pattern like: FAR3_20250630T122002_raw.csv
    Returns datetime object representing the start time of the video
    """
    filename = os.path.basename(csv_path)
    # Pattern to match YYYYMMDDTHHMMSS format
    timestamp_pattern = r'(\d{8}T\d{6})'
    match = re.search(timestamp_pattern, filename)
    
    if match:
        timestamp_str = match.group(1)
        try:
            # Parse YYYYMMDDTHHMMSS format
            return datetime.strptime(timestamp_str, '%Y%m%dT%H%M%S')
        except ValueError:
            print(f"Warning: Could not parse timestamp {timestamp_str} from filename")
            return None
    else:
        print(f"Warning: No timestamp found in filename {filename}")
        return None

def frames_to_timestamps(frames, fps, original_video_fps=25, start_time=None):
    """
    Convert frame numbers to actual timestamps
    
    Args:
        frames: array of frame numbers
        fps: sampling rate of the input data (samples per second)
        original_video_fps: FPS of the original video
        start_time: datetime object representing video start time
    
    Returns:
        pandas DatetimeIndex or array of seconds (if no start_time)
    """
    # Convert frames to seconds first
    if fps == 1:
        seconds = (frames // original_video_fps).astype(int)
    else:
        seconds = (frames // fps).astype(int)
    
    if start_time is not None:
        # Convert to actual timestamps
        timestamps = [start_time + timedelta(seconds=int(s)) for s in seconds]
        return pd.DatetimeIndex(timestamps)
    else:
        return seconds


def load_detections(csv_path, conf_thresh=0.25):
    df = pd.read_csv(csv_path)
    # ensure numeric columns
    for c in ['frame','confidence','xmin','ymin','xmax','ymax']:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')
    df = df.dropna(subset=['frame','xmin','ymin','xmax','ymax','confidence'])
    df = df[df['confidence'] >= conf_thresh].copy()
    df['frame'] = df['frame'].astype(int)
    df.reset_index(drop=True, inplace=True)
    
    # Extract start time from filename
    start_time = extract_timestamp_from_filename(csv_path)
    if start_time:
        print(f"Video start time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    return df, start_time

def bbox_area(row):
    return max(0.0, (row.xmax - row.xmin) * (row.ymax - row.ymin))

def bbox_center_xy(row):
    return ((row.xmin + row.xmax) / 2.0, (row.ymin + row.ymax) / 2.0)

# --------------------------
# Aggregation to time bins
# --------------------------

def frames_to_seconds(frames, fps, original_video_fps=25):
    # Convert frame numbers to seconds based on sampling rate
    # fps: sampling rate of the input data (how many samples per second)
    # original_video_fps: FPS of the original video from which frames were sampled
    
    # If fps=1, it means we have 1 sample per second from the original video
    # Frame numbers represent which frames were sampled from the original video
    # So frame 0, 25, 50, 75... from a 25 FPS video → seconds 0, 1, 2, 3...
    if fps == 1:
        return (frames // original_video_fps).astype(int)
    else:
        # Standard frame to second conversion for continuous frame sequences
        return (frames // fps).astype(int)

def aggregate_per_second(df, fps, classes=('adult','chick','fish'), original_video_fps=25, start_time=None):
    """
    Returns:
      seconds: numpy array [0..max_sec] or timestamps if start_time provided
      per_second_counts: dict class->np.array(length = max_sec+1)
      per_second_positions: dict class->list of list of (x,y) per second
      per_second_bbox_areas: dict class->list of lists of bbox areas per second
      per_second_detections_df: DataFrame of per-second flattened stats (counts, mean_x, mean_y, mean_area, mean_conf)
    """
    df = df.copy()
    
    # Convert frames to seconds
    df['second'] = frames_to_seconds(df['frame'], fps, original_video_fps)
    
    # If we have start_time, also create timestamp column
    if start_time is not None:
        df['timestamp'] = frames_to_timestamps(df['frame'], fps, original_video_fps, start_time)
    
    max_sec = int(df['second'].max()) if len(df) > 0 else 0
    n_sec = max_sec + 1

    # initialize
    counts = {c: np.zeros(n_sec, dtype=int) for c in classes}
    positions = {c: [[] for _ in range(n_sec)] for c in classes}
    areas = {c: [[] for _ in range(n_sec)] for c in classes}
    confs = {c: [[] for _ in range(n_sec)] for c in classes}

    for r in df.itertuples(index=False):
        cls = r._1  # pandas renames 'class' column to '_1' in itertuples
        if cls not in classes:
            continue
        s = int(r.second)
        counts[cls][s] += 1
        cx, cy = bbox_center_xy(r)
        positions[cls][s].append((cx, cy))
        areas[cls][s].append(bbox_area(r))
        confs[cls][s].append(r.confidence)

    # build per-second summary DataFrame for adults (and optionally others)
    data = {
        'second': np.arange(n_sec)
    }
    # total counts across classes
    data['count_total'] = sum(counts[c] for c in classes)
    for c in classes:
        data[f'count_{c}'] = counts[c]

        # mean positions (NaN if none)
        mean_x = np.array([np.mean([p[0] for p in plist]) if len(plist)>0 else np.nan for plist in positions[c]])
        mean_y = np.array([np.mean([p[1] for p in plist]) if len(plist)>0 else np.nan for plist in positions[c]])
        mean_area = np.array([np.mean(alist) if len(alist)>0 else np.nan for alist in areas[c]])
        max_area = np.array([np.max(alist) if len(alist)>0 else np.nan for alist in areas[c]])
        mean_conf = np.array([np.mean(clist) if len(clist)>0 else np.nan for clist in confs[c]])

        data[f'{c}_mean_x'] = mean_x
        data[f'{c}_mean_y'] = mean_y
        data[f'{c}_mean_area'] = mean_area
        data[f'{c}_max_area'] = max_area
        data[f'{c}_mean_conf'] = mean_conf

    per_sec_df = pd.DataFrame(data)
    
    # Add timestamp column if start_time was provided
    if start_time is not None and n_sec > 0:
        timestamps = [start_time + timedelta(seconds=int(s)) for s in np.arange(n_sec)]
        per_sec_df['timestamp'] = timestamps
    
    #per_sec_df.fillna(0, inplace=False)  # we may want NaNs preserved for some metrics; keep as-is for now
    print(per_sec_df.head())
    print(per_sec_df.iloc[0])
    return per_sec_df, positions, areas, counts

# --------------------------
# Arrival / Departure detection
# --------------------------

def detect_arrivals_departures(per_sec_df,
                               target_class='adult',
                               smooth_window_s=5,
                               error_window_s=10,
                               hold_seconds=8,
                               min_count_step=1):
    """
    Detect arrivals and departures based on per-second counts of `target_class`.

    Algorithm:
      - smooth counts with moving average over smooth_window_s
      - evaluate step changes by comparing average in [t-error_window_s, t-1] vs [t, t+hold_seconds-1]
      - an arrival is when mean_after - mean_before >= min_count_step and mean_after >= 1
      - departure is when mean_before - mean_after >= min_count_step and mean_after == 0 (or less by step)
      - we advance t past the hold interval when a change is detected to avoid double counting.

    Returns:
      events: list of dicts with keys:
        - type: 'arrival' or 'departure'
        - second: int (timestamp second for the event, chosen as start of hold window)
        - before_mean, after_mean
    """
    counts = per_sec_df[f'count_{target_class}'].to_numpy()
    n = len(counts)
    if smooth_window_s > 1:
        # use uniform_filter1d for fast moving average; pad mode='nearest' is fine
        smooth = uniform_filter1d(counts.astype(float), size=smooth_window_s, mode='nearest')
    else:
        smooth = counts.astype(float)

    events = []
    t = 0
    while t < n:
        start_before = max(0, t - error_window_s)
        end_before = t  # exclusive
        if end_before - start_before <= 0:
            mean_before = 0.0
        else:
            mean_before = smooth[start_before:end_before].mean()

        # define after window as [t, t+hold_seconds)
        end_after = min(n, t + hold_seconds)
        mean_after = smooth[t:end_after].mean() if end_after - t > 0 else 0.0

        delta = mean_after - mean_before

        # arrival
        # Only allow arrival detection if the before window is not at the very start
        if (delta >= min_count_step) and (mean_after >= 1.0) and (start_before > 0):
            events.append({'type':'arrival', 'second': t, 'before_mean': float(mean_before), 'after_mean': float(mean_after)})
            t = end_after  # skip to end of hold to avoid repeated detection
            continue

        # departure (count drop)
        if (mean_before - mean_after >= min_count_step) and (mean_after <= max(0.5, mean_before - min_count_step)):
            events.append({'type':'departure', 'second': t, 'before_mean': float(mean_before), 'after_mean': float(mean_after)})
            t = end_after
            continue

        t += 1

    return events, smooth

# --------------------------
# Associate fish with arrivals
# --------------------------

def associate_fish_with_arrivals(events, per_sec_df, fish_window_s=5, areas_by_class=None):
    """
    For each arrival event, look for fish detections within [second - fish_window_s, second + fish_window_s].
    Returns event list with added keys: arrival_with_fish (bool), fish_mean_area, fish_max_area, fish_count
    """
    if areas_by_class is None:
        # fallback to using per_sec_df columns
        # build arrays
        fish_mean_area = per_sec_df['fish_mean_area'].to_numpy() if 'fish_mean_area' in per_sec_df else None
        fish_max_area = per_sec_df['fish_max_area'].to_numpy() if 'fish_max_area' in per_sec_df else None
        fish_count = per_sec_df['count_fish'].to_numpy() if 'count_fish' in per_sec_df else None

    for ev in events:
        if ev['type'] != 'arrival':
            ev.update({'arrival_with_fish': False, 'fish_mean_area': None, 'fish_max_area': None, 'fish_count': 0})
            continue
        s = ev['second']
        start = max(0, s - fish_window_s)
        end = min(len(per_sec_df), s + fish_window_s + 1)
        # if areas_by_class provided as lists-of-lists, we can compute mean over the intervals
        if areas_by_class is not None and 'fish' in areas_by_class:
            # flatten list of lists between start and end
            fish_areas_flat = []
            for i in range(start, end):
                fish_areas_flat.extend(areas_by_class['fish'][i])
            fish_count_local = len(fish_areas_flat)
            if fish_count_local > 0:
                ev['arrival_with_fish'] = True
                ev['fish_count'] = fish_count_local
                ev['fish_mean_area'] = float(np.mean(fish_areas_flat))
                ev['fish_max_area'] = float(np.max(fish_areas_flat))
            else:
                ev['arrival_with_fish'] = False
                ev['fish_count'] = 0
                ev['fish_mean_area'] = None
                ev['fish_max_area'] = None
        else:
            # use per_sec_df columns
            fish_count_local = per_sec_df['count_fish'].iloc[start:end].sum()
            if fish_count_local > 0:
                # compute approximate mean and max using available per-second aggregates
                # weighted mean by count per second
                counts = per_sec_df['count_fish'].iloc[start:end].to_numpy()
                means = per_sec_df['fish_mean_area'].iloc[start:end].to_numpy()
                maxs = per_sec_df['fish_max_area'].iloc[start:end].to_numpy()
                # weighted mean:
                total_count = counts.sum()
                if total_count > 0:
                    # treat NaNs
                    means = np.nan_to_num(means)
                    weighted_mean = (means * counts).sum() / total_count
                    weighted_max = np.nanmax(maxs) if len(maxs)>0 else np.nan
                else:
                    weighted_mean = None
                    weighted_max = None
                ev['arrival_with_fish'] = True
                ev['fish_count'] = int(total_count)
                ev['fish_mean_area'] = float(weighted_mean) if weighted_mean is not None else None
                ev['fish_max_area'] = float(weighted_max) if not np.isnan(weighted_max) else None
            else:
                ev['arrival_with_fish'] = False
                ev['fish_count'] = 0
                ev['fish_mean_area'] = None
                ev['fish_max_area'] = None

    return events

# --------------------------
# Movement metric
# --------------------------

def compute_movement_metric(per_sec_df, classes=('adult',), grid=(10,10), smoothing_s=5, fps=30):
    """
    Compute a movement metric for target class(es):
      - For each second, compute weighted average X and Y position (weights = counts)
      - Optionally aggregate into grid cells and compute centroid movement across time

    Returns:
      movement_df: DataFrame with columns second, x_mean, y_mean, x_smoothed, y_smoothed, movement_delta (frame-to-frame distance)
    """
    sec = per_sec_df['second'].to_numpy()
    # compute mean positions across provided classes; if multiple classes, weight equally by counts
    x_total = np.zeros(len(sec))
    y_total = np.zeros(len(sec))
    total_counts = np.zeros(len(sec))

    for c in classes:
        xcol = f'{c}_mean_x'
        ycol = f'{c}_mean_y'
        if xcol in per_sec_df:
            x_vals = per_sec_df[xcol].to_numpy()
            y_vals = per_sec_df[ycol].to_numpy()
            cnts = per_sec_df[f'count_{c}'].to_numpy()
            # treat NaNs in means -> set to 0 but reduce counts accordingly
            nan_mask = np.isnan(x_vals) | np.isnan(y_vals)
            x_vals = np.nan_to_num(x_vals)
            y_vals = np.nan_to_num(y_vals)
            # accumulate weight
            x_total += x_vals * cnts
            y_total += y_vals * cnts
            total_counts += cnts

    # avoid division by zero
    with np.errstate(invalid='ignore', divide='ignore'):
        x_mean = np.where(total_counts > 0, x_total / total_counts, np.nan)
        y_mean = np.where(total_counts > 0, y_total / total_counts, np.nan)

    # smoothing in seconds: convert to window length (smoothing_s) for uniform_filter1d (works on float arrays)
    if smoothing_s > 1:
        # uniform_filter1d expects an integer window size -> convert to samples per second (we're already in seconds)
        w = int(max(1, smoothing_s))
        x_smooth = uniform_filter1d(np.nan_to_num(x_mean, nan=0.0), size=w, mode='nearest')
        y_smooth = uniform_filter1d(np.nan_to_num(y_mean, nan=0.0), size=w, mode='nearest')
        # re-mask positions where counts were zero to nan
        x_smooth = np.where(total_counts > 0, x_smooth, np.nan)
        y_smooth = np.where(total_counts > 0, y_smooth, np.nan)
    else:
        x_smooth = x_mean.copy()
        y_smooth = y_mean.copy()

    # movement delta: euclidean distance between smoothed positions frame-to-frame (in pixels)
    deltas = np.full_like(x_smooth, np.nan)
    for i in range(1, len(x_smooth)):
        if np.isnan(x_smooth[i]) or np.isnan(x_smooth[i-1]):
            deltas[i] = np.nan
        else:
            deltas[i] = float(np.hypot(x_smooth[i] - x_smooth[i-1], y_smooth[i] - y_smooth[i-1]))

    movement_df = pd.DataFrame({
        'second': sec,
        'x_mean': x_mean,
        'y_mean': y_mean,
        'x_smooth': x_smooth,
        'y_smooth': y_smooth,
        'movement_delta': deltas
    })
    
    # Add timestamp column if it exists in the input DataFrame
    if 'timestamp' in per_sec_df.columns:
        movement_df['timestamp'] = per_sec_df['timestamp'].values
    
    # Filter out rows where there's no position data (all NaN)
    # Keep only rows where we have actual position data
    has_position_data = ~(pd.isna(movement_df['x_mean']) & pd.isna(movement_df['y_mean']))
    movement_df = movement_df[has_position_data].reset_index(drop=True)
    
    return movement_df

# --------------------------
# Wing flapping detection
# --------------------------

def detect_flapping(df, fps=30, target_class='adult', area_multiplier=3.0, baseline_window_s=30, min_consecutive_frames=1, original_video_fps=25, start_time=None):
    """
    Detect wing-flapping events by identifying bounding boxes whose area exceeds a baseline by a multiplier.

    Approach:
      - For each second, gather list of bbox areas for `target_class`.
      - Maintain a rolling baseline area = median area across previous baseline_window_s seconds.
      - If any bbox area in the current second > area_multiplier * baseline (and baseline > 0), it's a candidate flap.
      - Optionally require it to persist for min_consecutive_frames (in seconds) to be counted as one event.

    Returns:
      flaps: list of dict { 'second': t, 'area': area_value, 'multiplier': area / baseline, 'center_x', 'center_y' }
    """
    df = df.copy()
    df['second'] = frames_to_seconds(df['frame'], fps, original_video_fps)
    max_sec = int(df['second'].max()) if len(df) > 0 else 0
    flaps = []

    # build per-second list of areas + centers
    per_second_entries = [[] for _ in range(max_sec+1)]
    for r in df.itertuples(index=False):
        if r._1 != target_class:  # pandas renames 'class' column to '_1' in itertuples
            continue
        s = int(r.second)
        a = bbox_area(r)
        cx, cy = bbox_center_xy(r)
        per_second_entries[s].append({'area': a, 'cx': cx, 'cy': cy, 'confidence': r.confidence})

    # rolling baseline: median area over previous baseline_window_s seconds (exclude current second)
    for t in range(max_sec+1):
        start = max(0, t - baseline_window_s)
        end = t  # exclusive
        # collect baseline areas
        baseline_areas = []
        for i in range(start, end):
            baseline_areas.extend([e['area'] for e in per_second_entries[i]])
        baseline_median = np.median(baseline_areas) if len(baseline_areas) > 0 else 0.0

        # evaluate current second entries
        for e in per_second_entries[t]:
            a = e['area']
            if baseline_median > 0 and a >= area_multiplier * baseline_median:
                flaps.append({'second': t, 'area': float(a), 'baseline_median': float(baseline_median),
                              'multiplier': float(a / baseline_median), 'center_x': float(e['cx']), 'center_y': float(e['cy']),
                              'confidence': float(e['confidence'])})
    # Optionally, collapse consecutive-second flaps into single events (not implemented here; each second is recorded)
    return flaps

# --------------------------
# Aggregation per minute
# --------------------------

def aggregate_per_minute(per_sec_df, events, movement_df, flaps, minute_bin_align=60):
    """
    Produce a per-minute aggregated DataFrame with:
      - minute_start (seconds)
      - total_arrivals, total_departures, arrivals_with_fish, avg_movement_delta, total_flaps, total_adult_count (sum over minute)
      - mean_adult_count (mean/sec within minute)
      - avg_fish_size_for_minutes (mean of fish_mean_area)
    """
    # create minute index
    per_sec_df = per_sec_df.copy()
    per_sec_df['minute'] = (per_sec_df['second'] // 60).astype(int)
    
    # Get all minutes that have data (from per_sec_df) or events
    data_minutes = set(per_sec_df['minute'].unique())
    event_minutes = set()
    for e in events:
        event_minutes.add(e['second'] // 60)
    for f in flaps:
        event_minutes.add(f['second'] // 60)
    
    # Combine and sort all minutes with any activity
    minutes = sorted(data_minutes.union(event_minutes))

    rows = []
    # make events easier to query
    arr_seconds = [e['second'] for e in events if e['type'] == 'arrival']
    arr_with_fish = [e for e in events if e['type']=='arrival' and e.get('arrival_with_fish', False)]
    dep_seconds = [e['second'] for e in events if e['type'] == 'departure']

    flap_seconds = [f['second'] for f in flaps]

    for m in minutes:
        start_s = m * 60
        end_s = start_s + 60
        slice_df = per_sec_df[(per_sec_df['second'] >= start_s) & (per_sec_df['second'] < end_s)]
        total_adult = int(slice_df['count_adult'].sum()) if 'count_adult' in slice_df else 0
        mean_adult = float(slice_df['count_adult'].mean()) if 'count_adult' in slice_df and len(slice_df)>0 else 0.0
        arrivals_in_min = sum(1 for s in arr_seconds if start_s <= s < end_s)
        departures_in_min = sum(1 for s in dep_seconds if start_s <= s < end_s)
        arrivals_with_fish_in_min = sum(1 for e in arr_with_fish if start_s <= e['second'] < end_s)
        flaps_in_min = sum(1 for s in flap_seconds if start_s <= s < end_s)

        # average movement delta for seconds with valid delta
        movement_slice = movement_df[(movement_df['second'] >= start_s) & (movement_df['second'] < end_s)]
        avg_movement = float(movement_slice['movement_delta'].dropna().mean()) if len(movement_slice.dropna())>0 else 0.0

        # fish size averages (mean of per-second fish_mean_area)
        fish_mean_areas = slice_df['fish_mean_area'].dropna() if 'fish_mean_area' in slice_df else pd.Series(dtype=float)
        avg_fish_area = float(fish_mean_areas.mean()) if len(fish_mean_areas)>0 else np.nan

        row_data = {
            'minute': m,
            'start_second': start_s,
            'end_second': end_s-1,
            'total_adult_count': total_adult,
            'mean_adult_count_per_s': mean_adult,
            'arrivals': arrivals_in_min,
            'departures': departures_in_min,
            'arrivals_with_fish': arrivals_with_fish_in_min,
            'flapping_events': flaps_in_min,
            'avg_movement_delta_px': avg_movement,
            'avg_fish_area_px': avg_fish_area
        }
        
        # Only include minutes with some activity (adults, arrivals, departures, or flapping)
        if (total_adult > 0 or arrivals_in_min > 0 or departures_in_min > 0 or flaps_in_min > 0):
            rows.append(row_data)

    return pd.DataFrame(rows)

# --------------------------
# Main pipeline wrapper
# --------------------------

def run_pipeline(csv_path,
                 out_prefix,
                 fps=30,
                 conf_thresh=0.25,
                 smooth_window_s=3,
                 error_window_s=10,
                 hold_seconds=8,
                 fish_window_s=5,
                 movement_smoothing_s=5,
                 flap_area_multiplier=3.0,
                 flap_baseline_s=30,
                 original_video_fps=25):
    os.makedirs(os.path.dirname(out_prefix) or ".", exist_ok=True)

    df, start_time = load_detections(csv_path, conf_thresh=conf_thresh)
    print(f"Loaded {len(df)} detections")

    per_sec_df, positions, areas, counts = aggregate_per_second(df, fps, classes=('adult','chick','fish'), original_video_fps=original_video_fps, start_time=start_time)

    # detect arrivals / departures
    events, smoothed_counts = detect_arrivals_departures(
        per_sec_df,
        target_class='adult',
        smooth_window_s=smooth_window_s,
        error_window_s=error_window_s,
        hold_seconds=hold_seconds
    )

    # enrich events with fish association
    events = associate_fish_with_arrivals(events, per_sec_df, fish_window_s=fish_window_s, areas_by_class=areas)

    # compute movement metric
    movement_df = compute_movement_metric(per_sec_df, classes=('adult',), smoothing_s=movement_smoothing_s, fps=fps)

    # detect flapping
    flaps = detect_flapping(df, fps=fps, target_class='adult', area_multiplier=flap_area_multiplier, baseline_window_s=flap_baseline_s, original_video_fps=original_video_fps, start_time=start_time)

    # save events
    events_df = pd.DataFrame(events)
    if start_time is not None and len(events_df) > 0:
        events_df['timestamp'] = [start_time + timedelta(seconds=int(s)) for s in events_df['second']]
    events_out = out_prefix + "_events.csv"
    events_df.to_csv(events_out, index=False)
    print("Saved events to", events_out)

    # flaps
    flaps_df = pd.DataFrame(flaps)
    if start_time is not None and len(flaps_df) > 0:
        flaps_df['timestamp'] = [start_time + timedelta(seconds=int(s)) for s in flaps_df['second']]
    flaps_out = out_prefix + "_flaps.csv"
    flaps_df.to_csv(flaps_out, index=False)
    print("Saved flap events to", flaps_out)

    # per-second summary (optional)
    per_sec_out = out_prefix + "_per_second.csv"
    per_sec_df.to_csv(per_sec_out, index=False)
    print("Saved per-second summary to", per_sec_out)

    # movement
    movement_out = out_prefix + "_movement.csv"
    movement_df.to_csv(movement_out, index=False)
    print("Saved movement summary to", movement_out)

    # aggregated per-minute
    per_min_df = aggregate_per_minute(per_sec_df, events, movement_df, flaps)
    per_min_out = out_prefix + "_per_minute.csv"
    per_min_df.to_csv(per_min_out, index=False)
    print("Saved per-minute aggregates to", per_min_out)

    return {
        'events_df': events_df,
        'flaps_df': flaps_df,
        'per_second_df': per_sec_df,
        'movement_df': movement_df,
        'per_minute_df': per_min_df
    }

# --------------------------
# CLI
# --------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("csv", help="detections CSV")
    parser.add_argument("--out_prefix", default="results/run", help="output prefix (files will be out_prefix_events.csv etc.)")
    parser.add_argument("--fps", type=int, default=30, help="sampling rate of input data (samples per second)")
    parser.add_argument("--original_video_fps", type=int, default=25, help="FPS of original video from which frames were sampled")
    parser.add_argument("--conf_thresh", type=float, default=0.25)
    parser.add_argument("--smooth_window_s", type=int, default=3)
    parser.add_argument("--error_window_s", type=int, default=10)
    parser.add_argument("--hold_seconds", type=int, default=8)
    parser.add_argument("--fish_window_s", type=int, default=5)
    parser.add_argument("--movement_smoothing_s", type=int, default=5)
    parser.add_argument("--flap_area_multiplier", type=float, default=3.0)
    parser.add_argument("--flap_baseline_s", type=int, default=30)

    args = parser.parse_args()

    run_pipeline(
        args.csv,
        args.out_prefix,
        fps=args.fps,
        conf_thresh=args.conf_thresh,
        smooth_window_s=args.smooth_window_s,
        error_window_s=args.error_window_s,
        hold_seconds=args.hold_seconds,
        fish_window_s=args.fish_window_s,
        movement_smoothing_s=args.movement_smoothing_s,
        flap_area_multiplier=args.flap_area_multiplier,
        flap_baseline_s=args.flap_baseline_s,
        original_video_fps=args.original_video_fps
    )


