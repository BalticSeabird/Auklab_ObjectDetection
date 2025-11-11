#!/usr/bin/env python3
"""
benchmark_event_detection.py

Benchmark the event detection algorithm against manually annotated ground truth.
Calculates precision, recall, F1-score and provides detailed analysis of matches/misses.

Usage:
    python3 code/postprocess/benchmark_event_detection.py
"""

import pandas as pd
import numpy as np
import re
import os
import sys
from datetime import datetime, timedelta
from collections import defaultdict

# Import detection functions
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from event_detector import (
    load_detections,
    aggregate_per_second,
    detect_arrivals_departures
)


def load_annotations(annotation_file):
    """Load and parse the annotation file"""
    df = pd.read_csv(annotation_file, sep=';')
    
    annotations = []
    for idx, row in df.iterrows():
        filename = row['File_name']
        
        # Extract base filename for CSV matching
        match = re.search(r'(TRI3_\d{8}T\d{6})\.mkv', filename)
        if not match:
            continue
        
        csv_base = match.group(1)
        
        # If no event, record as file with no events
        if pd.isna(row['Event']) or row['Event'] == '':
            annotations.append({
                'csv_base': csv_base,
                'has_events': False,
                'event_type': None,
                'time': None,
                'second': None
            })
            continue
        
        # Parse event time
        time_str = row['Time']
        event_type = row['Event'].lower()
        
        # Extract video start time from filename
        match_time = re.search(r'(\d{8})T(\d{6})', filename)
        if match_time:
            date_str = match_time.group(1)
            time_str_file = match_time.group(2)
            video_start = datetime.strptime(f'{date_str}{time_str_file}', '%Y%m%d%H%M%S')
            
            # Parse event time
            event_time = datetime.strptime(f'2025-06-28 {time_str}', '%Y-%m-%d %H:%M:%S')
            
            # Calculate elapsed seconds
            elapsed_seconds = int((event_time - video_start).total_seconds())
            
            annotations.append({
                'csv_base': csv_base,
                'has_events': True,
                'event_type': event_type,
                'time': time_str,
                'second': elapsed_seconds,
                'with_fish': row.get('With_fish', None) == 'yes'
            })
    
    return annotations


def run_detection(csv_base, conf_thresh=0.25, smooth_window_s=3, 
                 error_window_s=10, hold_seconds=8, filter_edges=True, 
                 frame_width=2688, frame_height=1520, edge_margin=50):
    """Run detection algorithm on a single file"""
    csv_path = f'csv_detection_1fps/{csv_base}_raw.csv'
    
    if not os.path.exists(csv_path):
        return None
    
    try:
        df, start_time = load_detections(csv_path, conf_thresh=conf_thresh, 
                                        filter_edges=filter_edges,
                                        frame_width=frame_width, 
                                        frame_height=frame_height, 
                                        edge_margin=edge_margin)
        
        if len(df) == 0:
            return {'events': [], 'start_time': start_time}
        
        per_sec_df, positions, areas, counts = aggregate_per_second(
            df, fps=1, classes=('adult','chick','fish'),
            original_video_fps=25, start_time=start_time
        )
        
        events, smoothed = detect_arrivals_departures(
            per_sec_df, target_class='adult', 
            smooth_window_s=smooth_window_s,
            error_window_s=error_window_s, 
            hold_seconds=hold_seconds
        )
        
        return {'events': events, 'start_time': start_time}
    
    except Exception as e:
        print(f"Error processing {csv_base}: {e}")
        return None


def match_events(annotated, detected, tolerance_seconds=3):
    """
    Match detected events to annotated events within tolerance window.
    Returns: matched_pairs, false_positives, false_negatives
    """
    matched_pairs = []
    unmatched_detected = list(range(len(detected)))
    unmatched_annotated = list(range(len(annotated)))
    
    # Try to match each annotated event
    for ann_idx, ann_event in enumerate(annotated):
        best_match = None
        best_distance = float('inf')
        
        for det_idx in unmatched_detected:
            det_event = detected[det_idx]
            
            # Must be same type
            if ann_event['event_type'] != det_event['type']:
                continue
            
            # Check time distance
            time_diff = abs(ann_event['second'] - det_event['second'])
            
            if time_diff <= tolerance_seconds and time_diff < best_distance:
                best_match = det_idx
                best_distance = time_diff
        
        if best_match is not None:
            matched_pairs.append({
                'annotated': ann_event,
                'detected': detected[best_match],
                'time_diff': best_distance
            })
            unmatched_detected.remove(best_match)
            unmatched_annotated.remove(ann_idx)
    
    false_positives = [detected[i] for i in unmatched_detected]
    false_negatives = [annotated[i] for i in unmatched_annotated]
    
    return matched_pairs, false_positives, false_negatives


def benchmark_algorithm(annotations, tolerance_seconds=3, **detection_params):
    """Run full benchmark across all annotated files"""
    
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
        
        # Run detection
        detection_result = run_detection(csv_base, **detection_params)
        
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


def calculate_metrics(tp, fp, fn):
    """Calculate precision, recall, F1-score"""
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return precision, recall, f1


def print_results(results, tolerance_seconds):
    """Print comprehensive benchmark results"""
    print("\n" + "="*80)
    print("EVENT DETECTION ALGORITHM BENCHMARK RESULTS")
    print("="*80)
    
    print(f"\n📁 FILES PROCESSED:")
    print(f"   Total files: {results['files_processed']}")
    print(f"   Files with annotated events: {results['files_with_events']}")
    
    print(f"\n⏱️  MATCHING TOLERANCE: ±{tolerance_seconds} seconds")
    
    # Overall metrics
    total_tp = results['total_matches']
    total_fp = results['total_false_positives']
    total_fn = results['total_false_negatives']
    
    precision, recall, f1 = calculate_metrics(total_tp, total_fp, total_fn)
    
    print(f"\n📊 OVERALL PERFORMANCE:")
    print(f"   True Positives:  {total_tp}")
    print(f"   False Positives: {total_fp}")
    print(f"   False Negatives: {total_fn}")
    print(f"   ───────────────────────────")
    print(f"   Precision: {precision:.1%} ({total_tp}/{total_tp + total_fp})")
    print(f"   Recall:    {recall:.1%} ({total_tp}/{total_tp + total_fn})")
    print(f"   F1-Score:  {f1:.1%}")
    
    # Per-type metrics
    print(f"\n📈 PERFORMANCE BY EVENT TYPE:")
    
    for event_type in ['arrival', 'departure']:
        tp = results['by_type'][event_type]['matches']
        fp = results['by_type'][event_type]['fp']
        fn = results['by_type'][event_type]['fn']
        
        prec, rec, f1_score = calculate_metrics(tp, fp, fn)
        
        print(f"\n   {event_type.upper()}:")
        print(f"     TP: {tp}  FP: {fp}  FN: {fn}")
        print(f"     Precision: {prec:.1%}  Recall: {rec:.1%}  F1: {f1_score:.1%}")
    
    # Detailed issues
    if results['details']:
        print(f"\n⚠️  FILES WITH DETECTION ISSUES ({len(results['details'])} files):")
        
        for detail in results['details'][:10]:  # Show first 10
            print(f"\n   📄 {detail['file']}:")
            print(f"      Matches: {detail.get('matches', detail.get('matched', 0))}")
            
            if detail['false_negatives']:
                print(f"      ❌ MISSED {len(detail['false_negatives'])} events:")
                for fn in detail['false_negatives']:
                    print(f"         • {fn['event_type'].title()} at {fn['time']} (second {fn['second']})")
            
            if detail['false_positives']:
                print(f"      ⚠️  FALSE POSITIVES {len(detail['false_positives'])} events:")
                for fp in detail['false_positives']:
                    print(f"         • {fp['type'].title()} at second {fp['second']}")
        
        if len(results['details']) > 10:
            print(f"\n   ... and {len(results['details']) - 10} more files with issues")
    
    print("\n" + "="*80)


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
    
    # Run benchmark with current parameters + edge filtering
    print("\nRunning detection algorithm on all files...")
    print("Parameters: smooth_window_s=3, error_window_s=10, hold_seconds=8")
    print("Edge filtering: ENABLED (margin=50px from 2688x1520 frame)")
    
    results = benchmark_algorithm(
        annotations,
        tolerance_seconds=3,
        conf_thresh=0.25,
        smooth_window_s=3,
        error_window_s=10,
        hold_seconds=8,
        filter_edges=True,
        frame_width=2688,
        frame_height=1520,
        edge_margin=50
    )
    
    # Print results
    print_results(results, tolerance_seconds=3)


if __name__ == "__main__":
    main()
