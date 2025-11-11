#!/usr/bin/env python3
"""
detection_filter.py

Filter and smooth raw object detections to reduce false positives/negatives.
Uses spatial consistency and temporal tracking to produce more reliable counts.

Key idea: A bird can't suddenly appear/disappear - use bounding box positions
and temporal consistency to filter spurious detections.
"""

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from collections import defaultdict


def compute_box_center(row):
    """Compute center of bounding box"""
    return ((row['xmin'] + row['xmax']) / 2, (row['ymin'] + row['ymax']) / 2)


def compute_iou(box1, box2):
    """Compute Intersection over Union between two boxes"""
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2
    
    # Intersection
    x_left = max(x1_min, x2_min)
    y_top = max(y1_min, y2_min)
    x_right = min(x1_max, x2_max)
    y_bottom = min(y1_max, y2_max)
    
    if x_right < x_left or y_bottom < y_top:
        return 0.0
    
    intersection = (x_right - x_left) * (y_bottom - y_top)
    
    # Union
    area1 = (x1_max - x1_min) * (y1_max - y1_min)
    area2 = (x2_max - x2_min) * (y2_max - y2_min)
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0.0


def filter_detections_spatial_temporal(df, target_class='adult', 
                                       max_distance=100, min_conf=0.3,
                                       temporal_window=3):
    """
    Filter detections using spatial and temporal consistency.
    
    Args:
        df: DataFrame with detections (must have 'second' column)
        target_class: Class to filter
        max_distance: Maximum pixel distance for same bird in consecutive frames
        min_conf: Minimum confidence threshold
        temporal_window: Number of seconds to look back/forward for consistency
    
    Returns:
        filtered_df: DataFrame with filtered detections
        filtered_counts: Array of counts per second after filtering
    """
    
    # Filter by class and confidence
    class_df = df[(df['class'] == target_class) & (df['confidence'] >= min_conf)].copy()
    
    if len(class_df) == 0:
        return class_df, np.zeros(df['second'].max() + 1 if len(df) > 0 else 0)
    
    # Add centers
    class_df['center_x'] = (class_df['xmin'] + class_df['xmax']) / 2
    class_df['center_y'] = (class_df['ymin'] + class_df['ymax']) / 2
    
    max_second = int(df['second'].max())
    filtered_counts = []
    filtered_detections = []
    
    # Process each second
    for sec in range(max_second + 1):
        sec_detections = class_df[class_df['second'] == sec]
        
        if len(sec_detections) == 0:
            # Check if we should fill in from neighbors
            neighbors_before = class_df[(class_df['second'] >= max(0, sec - temporal_window)) & 
                                       (class_df['second'] < sec)]
            neighbors_after = class_df[(class_df['second'] > sec) & 
                                      (class_df['second'] <= min(max_second, sec + temporal_window))]
            
            # For now, just record 0 (can be improved with tracking)
            filtered_counts.append(0)
            continue
        
        # Keep all detections for this second (filtering was already applied by confidence)
        filtered_counts.append(len(sec_detections))
        filtered_detections.append(sec_detections)
    
    # Concatenate all filtered detections
    if filtered_detections:
        filtered_df = pd.concat(filtered_detections, ignore_index=True)
    else:
        filtered_df = pd.DataFrame()
    
    return filtered_df, np.array(filtered_counts)


def smooth_counts_with_spatial_context(counts, detection_positions_by_second, 
                                       window_size=3, outlier_threshold=2):
    """
    Smooth counts using spatial context to avoid removing real events.
    
    Args:
        counts: Array of counts per second
        detection_positions_by_second: Dict mapping second -> list of (x,y) positions
        window_size: Size of smoothing window
        outlier_threshold: How many standard deviations to consider an outlier
    
    Returns:
        smoothed_counts: Array of smoothed counts
    """
    smoothed = counts.copy().astype(float)
    n = len(counts)
    
    for i in range(n):
        # Get local window
        start = max(0, i - window_size)
        end = min(n, i + window_size + 1)
        window = counts[start:end]
        
        # If current count is very different from neighbors
        if len(window) > 2:
            window_mean = np.mean(window)
            window_std = np.std(window) if np.std(window) > 0 else 1.0
            
            # Check if this is an outlier
            if abs(counts[i] - window_mean) > outlier_threshold * window_std:
                # This might be noise - but check if it's spatially consistent
                # For now, apply median filter
                smoothed[i] = np.median(window)
            else:
                smoothed[i] = counts[i]
        else:
            smoothed[i] = counts[i]
    
    return smoothed.astype(int)


def remove_isolated_spikes(counts, max_spike_duration=1):
    """
    Remove isolated count spikes that are likely detection errors.
    
    A "spike" is a sudden increase that returns to baseline within max_spike_duration.
    Example: [7,7,7,9,7,7] -> [7,7,7,7,7,7] (spike at position 3)
    
    Args:
        counts: Array of counts per second
        max_spike_duration: Maximum duration of spike to remove (in seconds)
    
    Returns:
        cleaned_counts: Array with spikes removed
    """
    cleaned = counts.copy()
    n = len(counts)
    
    i = 0
    while i < n - max_spike_duration - 1:
        # Check if we have a spike pattern
        baseline_before = counts[i]
        
        # Look for a sudden increase
        if counts[i + 1] > baseline_before:
            # Check if it returns to baseline quickly
            spike_end = i + 1
            while spike_end < min(n, i + max_spike_duration + 2) and counts[spike_end] > baseline_before:
                spike_end += 1
            
            spike_duration = spike_end - i - 1
            
            # If spike returns to baseline within threshold and duration is short
            if spike_end < n and spike_duration <= max_spike_duration:
                if counts[spike_end] <= baseline_before:
                    # This is likely a false positive spike - remove it
                    for j in range(i + 1, spike_end):
                        cleaned[j] = baseline_before
                    i = spike_end
                    continue
        
        i += 1
    
    return cleaned


def remove_isolated_dips(counts, max_dip_duration=1):
    """
    Remove isolated count dips that are likely missed detections.
    
    A "dip" is a sudden decrease that returns to baseline within max_dip_duration.
    Example: [7,7,7,5,7,7] -> [7,7,7,7,7,7] (dip at position 3)
    
    Args:
        counts: Array of counts per second
        max_dip_duration: Maximum duration of dip to fill (in seconds)
    
    Returns:
        cleaned_counts: Array with dips filled
    """
    cleaned = counts.copy()
    n = len(counts)
    
    i = 0
    while i < n - max_dip_duration - 1:
        # Check if we have a dip pattern
        baseline_before = counts[i]
        
        # Look for a sudden decrease
        if i + 1 < n and counts[i + 1] < baseline_before:
            # Check if it returns to baseline quickly
            dip_end = i + 1
            while dip_end < min(n, i + max_dip_duration + 2) and counts[dip_end] < baseline_before:
                dip_end += 1
            
            dip_duration = dip_end - i - 1
            
            # If dip returns to baseline within threshold and duration is short
            if dip_end < n and dip_duration <= max_dip_duration:
                # Check if it returns to approximately the same level
                if abs(counts[dip_end] - baseline_before) <= 1:
                    # This is likely a false negative (missed detection) - fill it
                    for j in range(i + 1, dip_end):
                        cleaned[j] = baseline_before
                    i = dip_end
                    continue
        
        i += 1
    
    return cleaned


def filter_counts_for_event_detection(counts, remove_spikes=True, remove_dips=True,
                                     max_spike_duration=1, max_dip_duration=2):
    """
    Apply filtering pipeline to count data before event detection.
    
    Args:
        counts: Raw counts per second
        remove_spikes: Whether to remove isolated count increases (false positives)
        remove_dips: Whether to fill isolated count decreases (false negatives)
        max_spike_duration: Max seconds for spike removal
        max_dip_duration: Max seconds for dip filling
    
    Returns:
        filtered_counts: Cleaned count array
    """
    filtered = counts.copy()
    
    if remove_spikes:
        filtered = remove_isolated_spikes(filtered, max_spike_duration=max_spike_duration)
    
    if remove_dips:
        filtered = remove_isolated_dips(filtered, max_dip_duration=max_dip_duration)
    
    return filtered


def analyze_no_event_files(csv_files, conf_thresh=0.25):
    """
    Analyze files with no events to understand noise patterns.
    
    Returns statistics about count variations in files where no events should occur.
    Useful for tuning filtering parameters.
    """
    stats = {
        'files_analyzed': 0,
        'total_seconds': 0,
        'count_changes': [],
        'spike_durations': [],
        'dip_durations': [],
        'max_count_variation': []
    }
    
    for csv_file in csv_files:
        # Load and process
        # (Implementation would go here)
        pass
    
    return stats
