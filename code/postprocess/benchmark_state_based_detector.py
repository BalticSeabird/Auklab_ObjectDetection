#!/usr/bin/env python3
"""
benchmark_state_based_detector.py

Benchmark the new state-based event detection algorithm against annotated ground truth.
Compares with the old algorithm to show improvements.

Usage:
    python3 code/postprocess/benchmark_state_based_detector.py
"""

import pandas as pd
import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from benchmark_event_detection import load_annotations, match_events, calculate_metrics, print_results
from event_detector import load_detections, aggregate_per_second
from state_based_detector import detect_arrivals_departures_with_filtering
from detection_filter import remove_isolated_spikes, remove_isolated_dips


def run_state_based_detection(csv_base, 
                              edge_margin=100, max_spike_duration=2, max_dip_duration=3,
                              min_stable_duration=15, max_stable_variation=0,
                              min_count_change=1, max_transition_duration=15):
    """Run state-based detection on a single file"""
    csv_path = f'csv_detection_1fps/{csv_base}_raw.csv'
    
    if not os.path.exists(csv_path):
        return None
    
    try:
        # Load with edge filtering
        df, start_time = load_detections(csv_path, conf_thresh=0.25, filter_edges=True,
                                        frame_width=2688, frame_height=1520, edge_margin=edge_margin)
        
        if len(df) == 0:
            return {'events': [], 'start_time': start_time}
        
        # Aggregate to per-second
        per_sec_df, positions, areas, counts = aggregate_per_second(
            df, fps=1, classes=('adult','chick','fish'),
            original_video_fps=25, start_time=start_time
        )
        
        # Apply filtering and detect events
        events, filtered_counts = detect_arrivals_departures_with_filtering(
            per_sec_df,
            target_class='adult',
            edge_margin=edge_margin,
            max_spike_duration=max_spike_duration,
            max_dip_duration=max_dip_duration,
            min_stable_duration=min_stable_duration,
            max_stable_variation=max_stable_variation,
            min_count_change=min_count_change,
            max_transition_duration=max_transition_duration
        )
        
        return {'events': events, 'start_time': start_time}
        
    except Exception as e:
        print(f"Error processing {csv_base}: {e}")
        return None


def benchmark_state_based(annotations, tolerance_seconds=3, **detection_params):
    """Run full benchmark using state-based detector"""
    from collections import defaultdict
    
    results = {
        'files_processed': 0,
        'files_with_events': 0,
        'total_matches': 0,
        'total_false_positives': 0,
        'total_false_negatives': 0,
        'by_type': {
            'arrival': {'matches': 0, 'fp': 0, 'fn': 0},
            'departure': {'matches': 0, 'fp': 0, 'fn': 0}
        },
        'details': []
    }
    
    # Group annotations by file
    files = defaultdict(list)
    files_no_events = set()
    
    for ann in annotations:
        if ann['has_events']:
            files[ann['csv_base']].append(ann)
        else:
            files_no_events.add(ann['csv_base'])
    
    # Process each file
    all_files = set(files.keys()) | files_no_events
    
    for csv_base in sorted(all_files):
        results['files_processed'] += 1
        
        # Get annotated events for this file
        true_events = files.get(csv_base, [])
        
        if true_events:
            results['files_with_events'] += 1
        
        # Run detection
        detection_result = run_state_based_detection(csv_base, **detection_params)
        
        if detection_result is None:
            continue
        
        detected_events = detection_result['events']
        
        # Debug: print detected events count
        if true_events:
            print(f"  {csv_base}: detected {len(detected_events)} events, expected {len(true_events)}")
        
        # Match events
        matched_pairs, false_positives, false_negatives = match_events(true_events, detected_events, tolerance_seconds)
        
        # Count matches by type
        matches_by_type = {'arrival': 0, 'departure': 0}
        for pair in matched_pairs:
            event_type = pair['annotated']['event_type']
            matches_by_type[event_type] += 1
        
        # Count FP by type
        fp_by_type = {'arrival': 0, 'departure': 0}
        for fp_event in false_positives:
            fp_by_type[fp_event['type']] += 1
        
        # Count FN by type
        fn_by_type = {'arrival': 0, 'departure': 0}
        for fn_event in false_negatives:
            fn_by_type[fn_event['event_type']] += 1
        
        results['total_matches'] += len(matched_pairs)
        results['total_false_positives'] += len(false_positives)
        results['total_false_negatives'] += len(false_negatives)
        
        for event_type in ['arrival', 'departure']:
            results['by_type'][event_type]['matches'] += matches_by_type[event_type]
            results['by_type'][event_type]['fp'] += fp_by_type[event_type]
            results['by_type'][event_type]['fn'] += fn_by_type[event_type]
        
        # Store details if there were issues
        if len(false_positives) > 0 or len(false_negatives) > 0:
            results['details'].append({
                'file': csv_base,
                'matched': len(matched_pairs),
                'false_positives': false_positives,
                'false_negatives': false_negatives
            })
    
    return results


def main():
    # Load annotations
    print("Loading annotations...")
    annotations = load_annotations('data/Annotated_arrivals_departures.csv')
    print(f"Loaded {len(annotations)} annotation entries")
    
    # Count annotated events
    event_count = sum(1 for a in annotations if a['has_events'])
    arrival_count = sum(1 for a in annotations if a['has_events'] and a['event_type'] == 'arrival')
    departure_count = sum(1 for a in annotations if a['has_events'] and a['event_type'] == 'departure')
    
    print(f"  - {event_count} events total")
    print(f"  - {arrival_count} arrivals")
    print(f"  - {departure_count} departures")
    
    # Run benchmark with state-based detector
    print("\n" + "="*80)
    print("STATE-BASED EVENT DETECTOR with OPTIMIZED PARAMETERS")
    print("="*80)
    print("Filtering: edge=100px, spike=2s, dip=3s")
    print("Detection: min_stable=15s, max_variation=0, min_change=1, max_transition=15s")
    print()
    
    results = benchmark_state_based(
        annotations,
        tolerance_seconds=3,
        edge_margin=100,
        max_spike_duration=2,
        max_dip_duration=3,
        min_stable_duration=15,
        max_stable_variation=0,
        min_count_change=1,
        max_transition_duration=15
    )
    
    # Print results
    print_results(results, tolerance_seconds=3)
    
    # Calculate overall metrics
    tp = results['total_matches']
    fp = results['total_false_positives']
    fn = results['total_false_negatives']
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    print("\n" + "="*80)
    print("COMPARISON WITH OLD ALGORITHM")
    print("="*80)
    print("\nOLD (threshold-based with smoothing):")
    print("  Overall F1: 55.9% (Arrivals: 79.1%, Departures: 16.0%)")
    print("  Major issue: Very poor departure detection")
    print()
    print("\nNEW (state-based with strong filtering):")
    
    # Calculate metrics for each type
    arrival_prec, arrival_rec, arrival_f1 = calculate_metrics(
        results['by_type']['arrival']['matches'],
        results['by_type']['arrival']['fp'],
        results['by_type']['arrival']['fn']
    )
    departure_prec, departure_rec, departure_f1 = calculate_metrics(
        results['by_type']['departure']['matches'],
        results['by_type']['departure']['fp'],
        results['by_type']['departure']['fn']
    )
    
    print(f"  Overall F1: {f1*100:.1f}% (Arrivals: {arrival_f1*100:.1f}%, "
          f"Departures: {departure_f1*100:.1f}%)")
    
    # Show improvement
    improvement_overall = f1 * 100 - 55.9
    improvement_departures = departure_f1 * 100 - 16.0
    
    print(f"\n✨ IMPROVEMENT:")
    print(f"  Overall F1: {improvement_overall:+.1f} percentage points")
    print(f"  Departure F1: {improvement_departures:+.1f} percentage points")
    
    if f1 > 0.559:
        print(f"\n🎉 STATE-BASED DETECTOR IS BETTER!")
    else:
        print(f"\n⚠️  Results suggest further tuning needed")


if __name__ == "__main__":
    main()
