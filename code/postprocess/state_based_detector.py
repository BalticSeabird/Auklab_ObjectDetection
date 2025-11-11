#!/usr/bin/env python3
"""
state_based_detector.py

State-based event detection algorithm for arrival/departure detection.

Core idea: Detect stable states (periods of consistent count) and transitions between them.
A transition from one stable state to another represents an arrival (increase) or departure (decrease).

This approach is more robust than threshold-based detection because:
1. It works for single bird events (change of 1)
2. Handles gradual departures (compares stable states, not instant changes)
3. Robust to noise (requires sustained change to new stable state)
4. Automatically finds exact timestamp of change
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class StableState:
    """Represents a stable period with consistent count"""
    start_second: int
    end_second: int
    mean_count: float
    median_count: int
    min_count: int
    max_count: int
    duration: int
    
    def __repr__(self):
        return f"StableState({self.start_second}-{self.end_second}, count={self.median_count}, duration={self.duration}s)"


def find_stable_periods(counts, min_duration=5, max_variation=1):
    """
    Find periods where the count is stable (low variation).
    
    A stable period is defined as consecutive seconds where:
    - Duration >= min_duration seconds
    - Count variation <= max_variation (max - min)
    
    Uses a sliding window approach to find all non-overlapping stable periods.
    
    Args:
        counts: Array of counts per second
        min_duration: Minimum duration for a stable period (seconds)
        max_variation: Maximum allowed variation (count range) within stable period
    
    Returns:
        List of StableState objects
    """
    stable_periods = []
    n = len(counts)
    
    if n < min_duration:
        return stable_periods
    
    i = 0
    while i <= n - min_duration:
        # Find the longest stable window starting at i
        best_end = -1
        
        for end in range(i + min_duration, n + 1):
            window = counts[i:end]
            variation = window.max() - window.min()
            
            if variation <= max_variation:
                # This window is stable, keep expanding
                best_end = end
            else:
                # Variation exceeded, stop expanding
                break
        
        if best_end > 0:
            # Found a stable period
            window = counts[i:best_end]
            stable_periods.append(StableState(
                start_second=i,
                end_second=best_end - 1,
                mean_count=window.mean(),
                median_count=int(np.median(window)),
                min_count=int(window.min()),
                max_count=int(window.max()),
                duration=best_end - i
            ))
            # Move past this stable period
            i = best_end
        else:
            # No stable period found at this position, move forward
            i += 1
    
    return stable_periods


def find_change_point(counts, start, end, baseline_count, direction='increase'):
    """
    Find the exact second where the count change occurred.
    
    Args:
        counts: Array of counts
        start: Start of transition window
        end: End of transition window
        baseline_count: The count before the transition
        direction: 'increase' for arrival, 'decrease' for departure
    
    Returns:
        Second where change was first detected
    """
    if start >= end or start >= len(counts):
        return start
    
    # Look for first significant deviation from baseline
    for sec in range(start, min(end, len(counts))):
        if direction == 'increase':
            if counts[sec] > baseline_count:
                return sec
        else:  # decrease
            if counts[sec] < baseline_count:
                return sec
    
    return start


def calculate_confidence(state_before, state_after, count_change):
    """
    Calculate confidence score for an event.
    
    Higher confidence when:
    - Stable states are longer (more reliable)
    - Count change is larger (more obvious)
    - States are more stable (less internal variation)
    
    Returns:
        Confidence score between 0 and 1
    """
    # Duration score (longer stable periods = more confident)
    min_duration = min(state_before.duration, state_after.duration)
    duration_score = min(min_duration / 10.0, 1.0)  # Max out at 10 seconds
    
    # Change magnitude score (larger changes = more confident)
    magnitude_score = min(abs(count_change) / 3.0, 1.0)  # Max out at 3 birds
    
    # Stability score (less variation in stable states = more confident)
    before_variation = state_before.max_count - state_before.min_count
    after_variation = state_after.max_count - state_after.min_count
    avg_variation = (before_variation + after_variation) / 2.0
    stability_score = max(0, 1.0 - avg_variation)
    
    # Combined confidence (weighted average)
    confidence = (duration_score * 0.3 + magnitude_score * 0.4 + stability_score * 0.3)
    
    return confidence


def detect_events_state_based(counts, min_stable_duration=5, max_stable_variation=1, 
                              min_count_change=1, max_transition_duration=15):
    """
    Detect arrival and departure events using state-based detection.
    
    Algorithm:
    1. Find all stable periods (consistent count for min_duration)
    2. Compare consecutive stable states
    3. If count changed significantly, it's an event
    4. Refine timestamp to exact change point
    
    Args:
        counts: Array of counts per second
        min_stable_duration: Minimum duration for stable state (seconds)
        max_stable_variation: Maximum count variation within stable state
        min_count_change: Minimum count change to detect event
        max_transition_duration: Maximum seconds allowed for transition
    
    Returns:
        List of event dictionaries with keys:
        - second: when event occurred
        - type: 'arrival' or 'departure'
        - count_change: magnitude of change
        - confidence: 0-1 confidence score
        - from_count: count before event
        - to_count: count after event
    """
    events = []
    
    # Step 1: Find stable periods
    stable_states = find_stable_periods(counts, 
                                       min_duration=min_stable_duration,
                                       max_variation=max_stable_variation)
    
    if len(stable_states) < 2:
        # Need at least two stable states to detect a transition
        return events
    
    # Step 2: Analyze transitions between consecutive stable states
    for i in range(len(stable_states) - 1):
        state_before = stable_states[i]
        state_after = stable_states[i + 1]
        
        # Calculate count change
        count_change = state_after.median_count - state_before.median_count
        
        # Check if change is significant
        if abs(count_change) < min_count_change:
            continue
        
        # Check transition duration
        transition_start = state_before.end_second
        transition_end = state_after.start_second + 1  # Include the first second of new state
        transition_duration = transition_end - transition_start
        
        if transition_duration > max_transition_duration:
            # Transition too long, might not be a single event
            continue
        
        # Determine event type
        event_type = 'arrival' if count_change > 0 else 'departure'
        
        # Find exact change point (use the start of the new stable state as event time)
        event_second = state_after.start_second
        
        # Calculate confidence
        confidence = calculate_confidence(state_before, state_after, count_change)
        
        events.append({
            'second': event_second,
            'type': event_type,
            'count_change': int(count_change),
            'confidence': confidence,
            'from_count': state_before.median_count,
            'to_count': state_after.median_count,
            'before_state': f"{state_before.start_second}-{state_before.end_second}",
            'after_state': f"{state_after.start_second}-{state_after.end_second}"
        })
    
    return events


def detect_arrivals_departures_with_filtering(per_second_df, target_class='adult',
                                              edge_margin=100, max_spike_duration=3, 
                                              max_dip_duration=4,
                                              min_stable_duration=5, max_stable_variation=1,
                                              min_count_change=1, max_transition_duration=15):
    """
    Complete pipeline: Apply filtering then detect events using state-based method.
    
    This is the main function to use. It combines:
    1. Edge filtering (done before this function)
    2. Count aggregation (done before this function)
    3. Spike/dip filtering
    4. State-based event detection
    
    Args:
        per_second_df: DataFrame with per-second aggregated data (from aggregate_per_second)
        target_class: Class to analyze (default 'adult')
        edge_margin: Margin for edge detection filtering (applied upstream)
        max_spike_duration: Max duration for spike removal (seconds)
        max_dip_duration: Max duration for dip filling (seconds)
        min_stable_duration: Minimum stable state duration (seconds)
        max_stable_variation: Max variation within stable state
        min_count_change: Minimum count change to detect
        max_transition_duration: Max seconds for state transition
    
    Returns:
        Tuple of (events, filtered_counts)
        - events: List of detected events
        - filtered_counts: The filtered count array used for detection
    """
    from detection_filter import remove_isolated_spikes, remove_isolated_dips
    
    # Extract counts for target class
    count_col = f'count_{target_class}'
    if count_col not in per_second_df.columns:
        return [], np.array([])
    
    counts = per_second_df[count_col].values
    
    # Apply filtering
    counts_filtered = remove_isolated_spikes(counts, max_spike_duration=max_spike_duration)
    counts_filtered = remove_isolated_dips(counts_filtered, max_dip_duration=max_dip_duration)
    
    # Detect events using state-based method
    events = detect_events_state_based(
        counts_filtered,
        min_stable_duration=min_stable_duration,
        max_stable_variation=max_stable_variation,
        min_count_change=min_count_change,
        max_transition_duration=max_transition_duration
    )
    
    return events, counts_filtered


if __name__ == "__main__":
    # Test with synthetic data
    print("Testing state-based event detector with synthetic data...\n")
    
    # Create test case: stable at 5, arrival (5->8), stable at 8, departure (8->6), stable at 6
    test_counts = np.array([5,5,5,5,5,5,5,5,  # Stable at 5 (8 seconds)
                           6,7,8,              # Arrival transition (3 seconds)
                           8,8,8,8,8,8,8,8,    # Stable at 8 (8 seconds)
                           7,6,                # Departure transition (2 seconds)
                           6,6,6,6,6,6,6])     # Stable at 6 (7 seconds)
    
    print("Test counts:")
    print(test_counts)
    print()
    
    # Find stable periods
    stable_states = find_stable_periods(test_counts, min_duration=5, max_variation=0)
    print(f"Found {len(stable_states)} stable states (max_variation=0):")
    for state in stable_states:
        print(f"  {state}")
    print()
    
    # Detect events
    events = detect_events_state_based(test_counts, min_stable_duration=5, max_stable_variation=0)
    print(f"Detected {len(events)} events:")
    for event in events:
        print(f"  {event['type'].upper()} at second {event['second']}: "
              f"{event['from_count']} -> {event['to_count']} "
              f"(change: {event['count_change']:+d}, confidence: {event['confidence']:.2f})")
    print()
    
    # Now test with max_variation=1 (more lenient)
    print("\n" + "="*60)
    print("Testing with max_variation=1 (allows ±1 within stable state):")
    print("="*60)
    
    stable_states = find_stable_periods(test_counts, min_duration=5, max_variation=1)
    print(f"Found {len(stable_states)} stable states:")
    for state in stable_states:
        print(f"  {state}")
    print()
    
    # Detect events
    events = detect_events_state_based(test_counts, min_stable_duration=5, max_stable_variation=1)
    print(f"Detected {len(events)} events:")
    for event in events:
        print(f"  {event['type'].upper()} at second {event['second']}: "
              f"{event['from_count']} -> {event['to_count']} "
              f"(change: {event['count_change']:+d}, confidence: {event['confidence']:.2f})")
