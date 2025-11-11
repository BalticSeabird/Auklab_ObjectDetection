#!/usr/bin/env python3
"""
benchmark_with_filtering.py

Benchmark event detection with pre-filtering applied to clean detection noise.
"""

import pandas as pd
import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from event_detector import load_detections, aggregate_per_second, detect_arrivals_departures
from benchmark_event_detection import load_annotations, match_events, calculate_metrics, print_results
from detection_filter import filter_counts_for_event_detection


def run_detection_with_filtering(csv_base, conf_thresh=0.25, smooth_window_s=3,
                                 error_window_s=10, hold_seconds=8,
                                 apply_spike_filter=True, apply_dip_filter=True,
                                 max_spike_duration=1, max_dip_duration=2):
    """Run detection with optional pre-filtering of counts"""
    csv_path = f'csv_detection_1fps/{csv_base}_raw.csv'
    
    if not os.path.exists(csv_path):
        return None
    
    try:
        df, start_time = load_detections(csv_path, conf_thresh=conf_thresh)
        
        if len(df) == 0:
            return {'events': [], 'start_time': start_time}
        
        per_sec_df, positions, areas, counts = aggregate_per_second(
            df, fps=1, classes=('adult','chick','fish'),
            original_video_fps=25, start_time=start_time
        )
        
        # Apply filtering to counts
        raw_counts = per_sec_df['count_adult'].to_numpy()
        filtered_counts = filter_counts_for_event_detection(
            raw_counts,
            remove_spikes=apply_spike_filter,
            remove_dips=apply_dip_filter,
            max_spike_duration=max_spike_duration,
            max_dip_duration=max_dip_duration
        )
        
        # Replace counts in dataframe
        per_sec_df_filtered = per_sec_df.copy()
        per_sec_df_filtered['count_adult'] = filtered_counts
        
        # Run event detection on filtered counts
        events, smoothed = detect_arrivals_departures(
            per_sec_df_filtered, target_class='adult',
            smooth_window_s=smooth_window_s,
            error_window_s=error_window_s,
            hold_seconds=hold_seconds
        )
        
        return {'events': events, 'start_time': start_time, 
                'raw_counts': raw_counts, 'filtered_counts': filtered_counts}
    
    except Exception as e:
        print(f"Error processing {csv_base}: {e}")
        return None


def benchmark_with_filtering(annotations, tolerance_seconds=3, **params):
    """Run benchmark with filtering enabled"""
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
        
        # Run detection with filtering
        detection_result = run_detection_with_filtering(csv_base, **params)
        
        if detection_result is None:
            continue
        
        detected_events = detection_result['events']
        annotated_events = files.get(csv_base, [])
        
        if len(annotated_events) > 0:
            results['files_with_events'] += 1
        
        # Match events
        matches, fps, fns = match_events(annotated_events, detected_events, tolerance_seconds)
        
        results['total_matches'] += len(matches)
        results['total_false_positives'] += len(fps)
        results['total_false_negatives'] += len(fns)
        
        # Count by type
        for match in matches:
            event_type = match['annotated']['event_type']
            results['by_type'][event_type]['matches'] += 1
        
        for fp in fps:
            event_type = fp['type']
            results['by_type'][event_type]['fp'] += 1
        
        for fn in fns:
            event_type = fn['event_type']
            results['by_type'][event_type]['fn'] += 1
        
        # Store details for files with issues
        if len(fps) > 0 or len(fns) > 0:
            results['details'].append({
                'file': csv_base,
                'matches': len(matches),
                'false_positives': fps,
                'false_negatives': fns
            })
    
    return results


def main():
    # Load annotations
    print("Loading annotations...")
    annotations = load_annotations('data/Annotated_arrivals_departures.csv')
    
    event_count = sum(1 for a in annotations if a['has_events'])
    arrival_count = sum(1 for a in annotations if a['has_events'] and a['event_type'] == 'arrival')
    departure_count = sum(1 for a in annotations if a['has_events'] and a['event_type'] == 'departure')
    
    print(f"Loaded {len(annotations)} annotation entries")
    print(f"  - {event_count} events total ({arrival_count} arrivals, {departure_count} departures)")
    
    # Test with filtering enabled
    print("\n" + "="*80)
    print("TESTING WITH PRE-FILTERING ENABLED")
    print("="*80)
    print("Parameters:")
    print("  Detection: smooth_window=3s, error_window=12s, hold=10s")
    print("  Filtering: spike_removal=ON (max 1s), dip_filling=ON (max 2s)")
    
    results_filtered = benchmark_with_filtering(
        annotations,
        tolerance_seconds=3,
        conf_thresh=0.25,
        smooth_window_s=3,
        error_window_s=12,
        hold_seconds=10,
        apply_spike_filter=True,
        apply_dip_filter=True,
        max_spike_duration=1,
        max_dip_duration=2
    )
    
    print_results(results_filtered, tolerance_seconds=3)
    
    # Compare with baseline
    print("\n" + "="*80)
    print("COMPARISON: WITHOUT FILTERING")
    print("="*80)
    print("Parameters:")
    print("  Detection: smooth_window=3s, error_window=12s, hold=10s")
    print("  Filtering: OFF")
    
    results_no_filter = benchmark_with_filtering(
        annotations,
        tolerance_seconds=3,
        conf_thresh=0.25,
        smooth_window_s=3,
        error_window_s=12,
        hold_seconds=10,
        apply_spike_filter=False,
        apply_dip_filter=False
    )
    
    print_results(results_no_filter, tolerance_seconds=3)
    
    # Summary comparison
    print("\n" + "="*80)
    print("FILTERING IMPACT SUMMARY")
    print("="*80)
    
    tp_filtered = results_filtered['total_matches']
    fp_filtered = results_filtered['total_false_positives']
    fn_filtered = results_filtered['total_false_negatives']
    prec_filt, rec_filt, f1_filt = calculate_metrics(tp_filtered, fp_filtered, fn_filtered)
    
    tp_baseline = results_no_filter['total_matches']
    fp_baseline = results_no_filter['total_false_positives']
    fn_baseline = results_no_filter['total_false_negatives']
    prec_base, rec_base, f1_base = calculate_metrics(tp_baseline, fp_baseline, fn_baseline)
    
    print(f"\nOverall Performance:")
    print(f"                    Baseline    With Filtering    Change")
    print(f"  Precision:        {prec_base:.1%}         {prec_filt:.1%}          {(prec_filt-prec_base)*100:+.1f}pp")
    print(f"  Recall:           {rec_base:.1%}         {rec_filt:.1%}          {(rec_filt-rec_base)*100:+.1f}pp")
    print(f"  F1-Score:         {f1_base:.1%}         {f1_filt:.1%}          {(f1_filt-f1_base)*100:+.1f}pp")
    
    print(f"\nFalse Positives: {fp_baseline} → {fp_filtered} ({fp_filtered-fp_baseline:+d})")
    print(f"False Negatives: {fn_baseline} → {fn_filtered} ({fn_filtered-fn_baseline:+d})")
    
    if f1_filt > f1_base:
        print("\n✅ Filtering IMPROVED performance")
    elif f1_filt < f1_base:
        print("\n⚠️  Filtering REDUCED performance")
    else:
        print("\n➖ Filtering had NO IMPACT")


if __name__ == "__main__":
    main()
