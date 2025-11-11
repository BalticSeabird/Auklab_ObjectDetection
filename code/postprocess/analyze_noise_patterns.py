#!/usr/bin/env python3
"""
analyze_noise_patterns.py

Analyze files with no annotated events to understand detection noise patterns.
This helps us tune the filtering parameters optimally.
"""

import pandas as pd
import numpy as np
import sys
import os
from collections import Counter

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from event_detector import load_detections, aggregate_per_second
from benchmark_event_detection import load_annotations


def analyze_count_stability(counts):
    """Analyze stability of count signal"""
    if len(counts) == 0:
        return {}
    
    # Count changes
    changes = np.diff(counts)
    
    # Find spikes (sudden increase then decrease)
    spikes = []
    i = 0
    while i < len(counts) - 2:
        if counts[i+1] > counts[i] and counts[i+2] <= counts[i]:
            spike_height = counts[i+1] - counts[i]
            spikes.append({'position': i, 'height': spike_height, 'duration': 1})
        i += 1
    
    # Find dips (sudden decrease then increase)
    dips = []
    i = 0
    while i < len(counts) - 2:
        if counts[i+1] < counts[i] and counts[i+2] >= counts[i]:
            dip_depth = counts[i] - counts[i+1]
            dips.append({'position': i, 'depth': dip_depth, 'duration': 1})
        i += 1
    
    # Count variation
    count_std = np.std(counts)
    count_range = counts.max() - counts.min()
    
    return {
        'mean_count': np.mean(counts),
        'std_count': count_std,
        'min_count': counts.min(),
        'max_count': counts.max(),
        'count_range': count_range,
        'num_changes': np.sum(np.abs(changes) > 0),
        'num_spikes': len(spikes),
        'num_dips': len(dips),
        'spikes': spikes,
        'dips': dips,
        'mean_change': np.mean(np.abs(changes))
    }


def main():
    print("ANALYZING DETECTION NOISE IN NO-EVENT FILES")
    print("="*70)
    
    # Load annotations
    annotations = load_annotations('data/Annotated_arrivals_departures.csv')
    
    # Get files with no events
    no_event_files = set()
    for ann in annotations:
        if not ann['has_events']:
            no_event_files.add(ann['csv_base'])
    
    print(f"\nFound {len(no_event_files)} files with no annotated events")
    print("These files should have stable counts (no arrivals/departures)\n")
    
    all_stats = []
    
    for i, csv_base in enumerate(sorted(no_event_files), 1):
        csv_path = f'csv_detection_1fps/{csv_base}_raw.csv'
        
        if not os.path.exists(csv_path):
            continue
        
        try:
            df, start_time = load_detections(csv_path, conf_thresh=0.25)
            
            if len(df) == 0:
                continue
            
            per_sec_df, _, _, _ = aggregate_per_second(
                df, fps=1, classes=('adult',), 
                original_video_fps=25, start_time=start_time
            )
            
            counts = per_sec_df['count_adult'].to_numpy()
            stats = analyze_count_stability(counts)
            stats['file'] = csv_base
            stats['duration'] = len(counts)
            
            all_stats.append(stats)
            
            if i % 5 == 0:
                print(f"Progress: {i}/{len(no_event_files)}", end='\r')
        
        except Exception as e:
            print(f"Error processing {csv_base}: {e}")
            continue
    
    print(f"Progress: {len(all_stats)}/{len(no_event_files)} - Complete!")
    
    # Aggregate statistics
    print("\n" + "="*70)
    print("AGGREGATED NOISE STATISTICS")
    print("="*70)
    
    total_seconds = sum(s['duration'] for s in all_stats)
    total_spikes = sum(s['num_spikes'] for s in all_stats)
    total_dips = sum(s['num_dips'] for s in all_stats)
    total_changes = sum(s['num_changes'] for s in all_stats)
    
    print(f"\n📊 Overall Statistics:")
    print(f"   Files analyzed: {len(all_stats)}")
    print(f"   Total seconds: {total_seconds}")
    print(f"   Total count changes: {total_changes} ({total_changes/total_seconds*100:.1f}% of seconds)")
    print(f"   Total spikes (false positive patterns): {total_spikes}")
    print(f"   Total dips (false negative patterns): {total_dips}")
    print(f"   Spikes per minute: {total_spikes / (total_seconds/60):.2f}")
    print(f"   Dips per minute: {total_dips / (total_seconds/60):.2f}")
    
    # Count stability metrics
    mean_stds = [s['std_count'] for s in all_stats]
    mean_ranges = [s['count_range'] for s in all_stats]
    mean_changes = [s['mean_change'] for s in all_stats]
    
    print(f"\n📈 Count Stability:")
    print(f"   Mean std deviation: {np.mean(mean_stds):.2f}")
    print(f"   Mean count range: {np.mean(mean_ranges):.1f}")
    print(f"   Mean absolute change per transition: {np.mean(mean_changes):.2f}")
    
    # Analyze spike patterns
    all_spikes = [spike for s in all_stats for spike in s['spikes']]
    if all_spikes:
        spike_heights = [s['height'] for s in all_spikes]
        print(f"\n🔺 Spike Analysis (n={len(all_spikes)}):")
        print(f"   Mean height: {np.mean(spike_heights):.2f}")
        print(f"   Height distribution: {Counter(spike_heights)}")
    
    # Analyze dip patterns  
    all_dips = [dip for s in all_stats for dip in s['dips']]
    if all_dips:
        dip_depths = [d['depth'] for d in all_dips]
        print(f"\n🔻 Dip Analysis (n={len(all_dips)}):")
        print(f"   Mean depth: {np.mean(dip_depths):.2f}")
        print(f"   Depth distribution: {Counter(dip_depths)}")
    
    # Show worst offenders
    print(f"\n⚠️  Files with most noise (top 5):")
    sorted_stats = sorted(all_stats, key=lambda x: x['num_spikes'] + x['num_dips'], reverse=True)
    for i, s in enumerate(sorted_stats[:5], 1):
        print(f"   {i}. {s['file']}: {s['num_spikes']} spikes, {s['num_dips']} dips, "
              f"range={s['count_range']}, std={s['std_count']:.2f}")
    
    print("\n" + "="*70)
    print("FILTERING RECOMMENDATIONS")
    print("="*70)
    
    # Calculate optimal filtering parameters
    spike_rate = total_spikes / (total_seconds/60)
    dip_rate = total_dips / (total_seconds/60)
    
    print(f"\nBased on noise analysis:")
    print(f"   • Spike removal: {'RECOMMENDED' if spike_rate > 0.5 else 'OPTIONAL'} ({spike_rate:.2f} spikes/min)")
    print(f"   • Dip filling: {'RECOMMENDED' if dip_rate > 0.5 else 'OPTIONAL'} ({dip_rate:.2f} dips/min)")
    
    if all_spikes:
        max_spike_height = max(spike_heights)
        print(f"   • Max spike height observed: {max_spike_height}")
        print(f"   • Suggested max_spike_duration: 1-2 seconds")
    
    if all_dips:
        max_dip_depth = max(dip_depths)
        print(f"   • Max dip depth observed: {max_dip_depth}")
        print(f"   • Suggested max_dip_duration: 2-3 seconds")
    
    print("\n✅ Analysis complete!")


if __name__ == "__main__":
    main()
