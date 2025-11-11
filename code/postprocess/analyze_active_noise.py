#!/usr/bin/env python3
"""
analyze_active_noise.py

Analyze detection noise in active periods (times between annotated events).
This helps us understand if filtering works well during high-activity periods.
"""

import pandas as pd
import numpy as np
import sys
import os
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import timedelta

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from event_detector import load_detections, aggregate_per_second
from benchmark_event_detection import load_annotations
from detection_filter import filter_counts_for_event_detection, remove_isolated_spikes, remove_isolated_dips


def find_quiet_periods(annotated_events, total_seconds, buffer_seconds=30):
    """
    Find periods with no events (adding buffer around events).
    
    Args:
        annotated_events: List of event dicts with 'second' keys
        total_seconds: Total duration of the file
        buffer_seconds: Seconds to exclude before/after each event
    
    Returns:
        List of (start, end) tuples for quiet periods
    """
    if not annotated_events:
        return [(0, total_seconds)]
    
    # Get event times
    event_times = sorted([e['second'] for e in annotated_events])
    
    quiet_periods = []
    current_start = 0
    
    for event_time in event_times:
        # End quiet period before this event (with buffer)
        quiet_end = max(0, event_time - buffer_seconds)
        
        if current_start < quiet_end:
            quiet_periods.append((current_start, quiet_end))
        
        # Start next quiet period after this event (with buffer)
        current_start = min(total_seconds, event_time + buffer_seconds)
    
    # Add final period
    if current_start < total_seconds:
        quiet_periods.append((current_start, total_seconds))
    
    return quiet_periods


def analyze_noise_in_quiet_periods(csv_base, annotated_events):
    """Analyze detection noise in periods with no events"""
    csv_path = f'csv_detection_1fps/{csv_base}_raw.csv'
    
    if not os.path.exists(csv_path):
        return None
    
    try:
        df, start_time = load_detections(csv_path, conf_thresh=0.25)
        
        if len(df) == 0:
            return None
        
        per_sec_df, _, _, _ = aggregate_per_second(
            df, fps=1, classes=('adult',), 
            original_video_fps=25, start_time=start_time
        )
        
        raw_counts = per_sec_df['count_adult'].to_numpy()
        total_seconds = len(raw_counts)
        
        # Find quiet periods
        quiet_periods = find_quiet_periods(annotated_events, total_seconds, buffer_seconds=30)
        
        # Extract counts from quiet periods
        quiet_counts = []
        for start, end in quiet_periods:
            quiet_counts.extend(raw_counts[start:end])
        
        if len(quiet_counts) == 0:
            return None
        
        quiet_counts = np.array(quiet_counts)
        
        # Analyze noise
        changes = np.diff(quiet_counts)
        
        # Count spikes and dips
        spikes = 0
        dips = 0
        for i in range(len(quiet_counts) - 2):
            if quiet_counts[i+1] > quiet_counts[i] and quiet_counts[i+2] <= quiet_counts[i]:
                spikes += 1
            if quiet_counts[i+1] < quiet_counts[i] and quiet_counts[i+2] >= quiet_counts[i]:
                dips += 1
        
        return {
            'file': csv_base,
            'total_seconds': total_seconds,
            'quiet_seconds': len(quiet_counts),
            'mean_count': np.mean(quiet_counts),
            'std_count': np.std(quiet_counts),
            'count_range': quiet_counts.max() - quiet_counts.min(),
            'num_changes': np.sum(np.abs(changes) > 0),
            'num_spikes': spikes,
            'num_dips': dips,
            'has_events': len(annotated_events) > 0,
            'num_events': len(annotated_events)
        }
    
    except Exception as e:
        print(f"Error processing {csv_base}: {e}")
        return None


def visualize_filtering(csv_base, annotated_events, output_dir='plots/filtering'):
    """Create visualization comparing raw vs filtered counts"""
    csv_path = f'csv_detection_1fps/{csv_base}_raw.csv'
    
    if not os.path.exists(csv_path):
        return False
    
    try:
        df, start_time = load_detections(csv_path, conf_thresh=0.25)
        
        if len(df) == 0:
            return False
        
        per_sec_df, _, _, _ = aggregate_per_second(
            df, fps=1, classes=('adult',), 
            original_video_fps=25, start_time=start_time
        )
        
        raw_counts = per_sec_df['count_adult'].to_numpy()
        
        # Apply different filtering strategies
        spike_only = remove_isolated_spikes(raw_counts, max_spike_duration=1)
        dip_only = remove_isolated_dips(raw_counts, max_dip_duration=2)
        both_filtered = filter_counts_for_event_detection(raw_counts, 
                                                          remove_spikes=True, 
                                                          remove_dips=True,
                                                          max_spike_duration=1,
                                                          max_dip_duration=2)
        
        # Create timestamps
        if start_time:
            timestamps = [start_time + timedelta(seconds=i) for i in range(len(raw_counts))]
        else:
            timestamps = list(range(len(raw_counts)))
        
        # Create plot
        fig, axes = plt.subplots(4, 1, figsize=(16, 12), sharex=True)
        
        # Raw counts
        axes[0].plot(timestamps, raw_counts, 'b-', linewidth=1, label='Raw counts')
        axes[0].set_ylabel('Adult Count')
        axes[0].set_title(f'{csv_base} - Raw Detection Counts')
        axes[0].grid(True, alpha=0.3)
        axes[0].legend()
        
        # Spike removal only
        axes[1].plot(timestamps, raw_counts, 'b-', linewidth=0.5, alpha=0.3, label='Raw')
        axes[1].plot(timestamps, spike_only, 'r-', linewidth=1, label='Spike removal')
        axes[1].set_ylabel('Adult Count')
        axes[1].set_title('Spike Removal Only (removes false positive blips)')
        axes[1].grid(True, alpha=0.3)
        axes[1].legend()
        
        # Dip filling only
        axes[2].plot(timestamps, raw_counts, 'b-', linewidth=0.5, alpha=0.3, label='Raw')
        axes[2].plot(timestamps, dip_only, 'g-', linewidth=1, label='Dip filling')
        axes[2].set_ylabel('Adult Count')
        axes[2].set_title('Dip Filling Only (fills false negative gaps)')
        axes[2].grid(True, alpha=0.3)
        axes[2].legend()
        
        # Both filters
        axes[3].plot(timestamps, raw_counts, 'b-', linewidth=0.5, alpha=0.3, label='Raw')
        axes[3].plot(timestamps, both_filtered, 'm-', linewidth=1, label='Both filters')
        axes[3].set_ylabel('Adult Count')
        axes[3].set_title('Both Filters Applied')
        axes[3].grid(True, alpha=0.3)
        axes[3].legend()
        
        # Mark annotated events with vertical lines
        for ax in axes:
            for event in annotated_events:
                event_time = timestamps[event['second']] if event['second'] < len(timestamps) else None
                if event_time:
                    color = 'green' if event['event_type'] == 'arrival' else 'red'
                    ax.axvline(event_time, color=color, linestyle='--', alpha=0.5, linewidth=1)
        
        # Format x-axis
        if start_time:
            for ax in axes:
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
                ax.xaxis.set_major_locator(mdates.MinuteLocator(interval=1))
        
        axes[-1].set_xlabel('Time' if start_time else 'Second')
        
        plt.tight_layout()
        
        # Save plot
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f'{csv_base}_filtering_comparison.png')
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"  Saved: {output_path}")
        return True
        
    except Exception as e:
        print(f"Error visualizing {csv_base}: {e}")
        return False


def main():
    print("ANALYZING DETECTION NOISE IN ACTIVE PERIODS")
    print("="*70)
    
    # Load annotations
    annotations = load_annotations('data/Annotated_arrivals_departures.csv')
    
    # Group by file
    files_with_events = {}
    files_no_events = set()
    
    for ann in annotations:
        if ann['has_events']:
            if ann['csv_base'] not in files_with_events:
                files_with_events[ann['csv_base']] = []
            files_with_events[ann['csv_base']].append(ann)
        else:
            files_no_events.add(ann['csv_base'])
    
    print(f"\nFiles with events: {len(files_with_events)}")
    print(f"Files with no events: {len(files_no_events)}")
    
    # Analyze noise in quiet periods of active files
    print("\n" + "="*70)
    print("NOISE ANALYSIS IN ACTIVE FILES (between events)")
    print("="*70)
    
    active_stats = []
    for csv_base, events in sorted(files_with_events.items()):
        result = analyze_noise_in_quiet_periods(csv_base, events)
        if result:
            active_stats.append(result)
            print(f"  {csv_base}: "
                  f"{result['num_events']} events, "
                  f"{result['quiet_seconds']}s quiet, "
                  f"mean_count={result['mean_count']:.1f}, "
                  f"spikes={result['num_spikes']}, "
                  f"dips={result['num_dips']}")
    
    # Aggregate statistics
    if active_stats:
        print("\n" + "="*70)
        print("AGGREGATED STATISTICS FOR ACTIVE PERIODS")
        print("="*70)
        
        total_quiet_seconds = sum(s['quiet_seconds'] for s in active_stats)
        total_spikes = sum(s['num_spikes'] for s in active_stats)
        total_dips = sum(s['num_dips'] for s in active_stats)
        
        mean_counts = [s['mean_count'] for s in active_stats]
        
        print(f"\n📊 Overall:")
        print(f"   Files analyzed: {len(active_stats)}")
        print(f"   Total quiet time: {total_quiet_seconds}s ({total_quiet_seconds/60:.1f} min)")
        print(f"   Mean adult count: {np.mean(mean_counts):.1f} (range: {min(mean_counts):.1f}-{max(mean_counts):.1f})")
        print(f"   Total spikes in quiet periods: {total_spikes} ({total_spikes/(total_quiet_seconds/60):.2f}/min)")
        print(f"   Total dips in quiet periods: {total_dips} ({total_dips/(total_quiet_seconds/60):.2f}/min)")
    
    # Create visualizations for a few interesting files
    print("\n" + "="*70)
    print("GENERATING FILTERING VISUALIZATIONS")
    print("="*70)
    
    # Select diverse files to visualize
    files_to_plot = []
    
    # 1. File with no events (low activity)
    if files_no_events:
        files_to_plot.append((list(files_no_events)[0], []))
    
    # 2-3. Files with high activity (most events)
    sorted_active = sorted(files_with_events.items(), key=lambda x: len(x[1]), reverse=True)
    for csv_base, events in sorted_active[:2]:
        files_to_plot.append((csv_base, events))
    
    # 4. File with moderate activity
    if len(sorted_active) >= len(sorted_active)//2:
        mid_file = sorted_active[len(sorted_active)//2]
        files_to_plot.append(mid_file)
    
    print(f"\nCreating visualizations for {len(files_to_plot)} files...")
    
    success_count = 0
    for csv_base, events in files_to_plot:
        print(f"\n  Processing {csv_base} ({len(events)} events)...")
        if visualize_filtering(csv_base, events):
            success_count += 1
    
    print(f"\n✅ Created {success_count} visualization plots in plots/filtering/")
    print("\nPlease review the plots to validate filtering effectiveness!")


if __name__ == "__main__":
    main()
