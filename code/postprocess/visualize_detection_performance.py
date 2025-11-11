"""
Visualize detection performance: show cleaned time series with annotated and detected events.
This helps identify where the algorithm succeeds and where it fails.
"""
import sys
import os
sys.path.append(os.path.dirname(__file__))

from event_detector import load_detections, aggregate_per_second
from detection_filter import remove_isolated_spikes, remove_isolated_dips
from state_based_detector import detect_arrivals_departures_with_filtering, find_stable_periods
from benchmark_event_detection import load_annotations
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
import numpy as np

def visualize_detection_performance(csv_base, annotations, 
                                   edge_margin=100, max_spike_duration=2, max_dip_duration=3,
                                   min_stable_duration=15, max_stable_variation=0):
    """
    Create detailed visualization for a single file showing:
    - Raw counts
    - Filtered counts (after spike/dip removal)
    - Stable periods detected
    - Annotated events (ground truth)
    - Detected events
    """
    
    csv_path = f'csv_detection_1fps/{csv_base}_raw.csv'
    
    if not os.path.exists(csv_path):
        print(f"File not found: {csv_path}")
        return None
    
    # Load and process data
    df, start_time = load_detections(csv_path, conf_thresh=0.25, filter_edges=True,
                                     frame_width=2688, frame_height=1520, edge_margin=edge_margin)
    
    if len(df) == 0:
        print(f"No detections in {csv_base}")
        return None
    
    per_sec_df, _, _, _ = aggregate_per_second(
        df, fps=1, classes=('adult','chick','fish'),
        original_video_fps=25, start_time=start_time
    )
    
    # Get counts
    counts_raw = per_sec_df['count_adult'].values
    
    # Apply filtering step by step
    counts_no_spikes = remove_isolated_spikes(counts_raw, max_spike_duration=max_spike_duration)
    counts_filtered = remove_isolated_dips(counts_no_spikes, max_dip_duration=max_dip_duration)
    
    # Find stable periods
    stable_periods = find_stable_periods(counts_filtered, min_stable_duration, max_stable_variation)
    
    # Run detection
    events, _ = detect_arrivals_departures_with_filtering(
        per_sec_df,
        target_class='adult',
        edge_margin=edge_margin,
        max_spike_duration=max_spike_duration,
        max_dip_duration=max_dip_duration,
        min_stable_duration=min_stable_duration,
        max_stable_variation=max_stable_variation
    )
    
    # Get annotated events for this file
    file_annotations = [a for a in annotations if a['csv_base'] == csv_base and a['has_events']]
    
    # Create time axis
    timestamps = [start_time + timedelta(seconds=i) for i in range(len(counts_raw))]
    
    # Create visualization
    fig, axes = plt.subplots(3, 1, figsize=(16, 12), sharex=True)
    fig.suptitle(f'{csv_base} - Detection Performance Analysis', fontsize=14, fontweight='bold')
    
    # Panel 1: Raw vs Filtered counts
    ax1 = axes[0]
    ax1.plot(timestamps, counts_raw, 'o-', color='gray', alpha=0.3, linewidth=1, markersize=2, label='Raw counts')
    ax1.plot(timestamps, counts_filtered, 'b-', linewidth=2, label='Filtered counts (after spike/dip removal)')
    ax1.set_ylabel('Adult Count', fontsize=11)
    ax1.set_title('Count Time Series: Raw vs Filtered', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper right')
    
    # Panel 2: Filtered counts with stable periods highlighted
    ax2 = axes[1]
    ax2.plot(timestamps, counts_filtered, 'b-', linewidth=2, label='Filtered counts')
    
    # Highlight stable periods with different colors
    colors = plt.cm.Set3(np.linspace(0, 1, len(stable_periods)))
    for i, period in enumerate(stable_periods):
        start_t = timestamps[period.start_second]
        end_t = timestamps[period.end_second]
        ax2.axvspan(start_t, end_t, alpha=0.2, color=colors[i])
        # Add count label in the middle of the period
        mid_idx = (period.start_second + period.end_second) // 2
        if mid_idx < len(timestamps):
            ax2.text(timestamps[mid_idx], period.median_count + 0.3, 
                    f'n={period.median_count}', 
                    ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    ax2.set_ylabel('Adult Count', fontsize=11)
    ax2.set_title(f'Stable Periods Detection (min_duration={min_stable_duration}s, max_variation={max_stable_variation})', 
                 fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='upper right')
    
    # Panel 3: Events comparison
    ax3 = axes[2]
    ax3.plot(timestamps, counts_filtered, 'b-', linewidth=2, alpha=0.5, label='Filtered counts')
    
    # Mark annotated events (ground truth)
    for ann in file_annotations:
        second = ann['second']
        if second < len(counts_filtered):
            event_type = ann['event_type']
            color = 'green' if event_type == 'arrival' else 'red'
            marker = '^' if event_type == 'arrival' else 'v'
            
            time_point = timestamps[second]
            count_at_event = counts_filtered[second]
            
            # Draw vertical line and marker
            ax3.axvline(time_point, color=color, linestyle='--', alpha=0.7, linewidth=2)
            ax3.plot(time_point, count_at_event, marker, color=color, markersize=15, 
                    markeredgecolor='black', markeredgewidth=1.5, 
                    label=f'Annotated {event_type}' if ann == file_annotations[0] or \
                          (ann != file_annotations[0] and file_annotations[0]['event_type'] != event_type) else '')
            
            # Add time label
            time_str = time_point.strftime('%H:%M:%S')
            ax3.text(time_point, count_at_event + 0.5, time_str, 
                    rotation=45, ha='left', va='bottom', fontsize=8, color=color, fontweight='bold')
    
    # Mark detected events
    for event in events:
        second = event['second']
        if second < len(counts_filtered):
            event_type = event['type']
            # Use different markers for detected events
            color = 'lime' if event_type == 'arrival' else 'orange'
            marker = 's' if event_type == 'arrival' else 'D'
            
            time_point = timestamps[second]
            count_at_event = counts_filtered[second]
            
            # Draw square/diamond marker
            ax3.plot(time_point, count_at_event, marker, color=color, markersize=12, 
                    markeredgecolor='black', markeredgewidth=1.5, alpha=0.8,
                    label=f'Detected {event_type}' if event == events[0] or \
                          (event != events[0] and events[0]['type'] != event_type) else '')
    
    ax3.set_ylabel('Adult Count', fontsize=11)
    ax3.set_xlabel('Time', fontsize=11)
    ax3.set_title('Event Detection: Ground Truth vs Algorithm', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.legend(loc='upper right', fontsize=9)
    
    # Format x-axis
    for ax in axes:
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
        ax.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    
    # Save
    os.makedirs('plots/detection_analysis', exist_ok=True)
    output_path = f'plots/detection_analysis/{csv_base}_analysis.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    # Print summary
    print(f"\n{'='*70}")
    print(f"File: {csv_base}")
    print(f"{'='*70}")
    print(f"Duration: {len(counts_raw)} seconds")
    print(f"Count range: {counts_filtered.min()}-{counts_filtered.max()}")
    print(f"\nStable periods: {len(stable_periods)}")
    for i, period in enumerate(stable_periods[:5], 1):  # Show first 5
        print(f"  {i}. Seconds {period.start_second:3d}-{period.end_second:3d} ({period.duration:3d}s): count={period.median_count}")
    if len(stable_periods) > 5:
        print(f"  ... and {len(stable_periods)-5} more")
    
    print(f"\nAnnotated events: {len(file_annotations)}")
    for ann in file_annotations:
        print(f"  - {ann['event_type']:9s} at {ann['time']} (second {ann['second']})")
    
    print(f"\nDetected events: {len(events)}")
    for event in events:
        print(f"  - {event['type']:9s} at second {event['second']:3d}: "
              f"{event['from_count']}→{event['to_count']} (confidence={event['confidence']:.2f})")
    
    # Analyze matches
    matched = 0
    false_positives = []
    false_negatives = list(file_annotations)
    
    for event in events:
        matched_ann = False
        for ann in file_annotations:
            if abs(event['second'] - ann['second']) <= 3:
                matched += 1
                matched_ann = True
                if ann in false_negatives:
                    false_negatives.remove(ann)
                break
        if not matched_ann:
            false_positives.append(event)
    
    print(f"\nMatching (tolerance ±3s):")
    print(f"  Matched: {matched}/{len(file_annotations)}")
    print(f"  False Positives: {len(false_positives)}")
    print(f"  False Negatives: {len(false_negatives)}")
    
    if false_positives:
        print(f"\n  False positive events:")
        for fp in false_positives:
            print(f"    - {fp['type']:9s} at second {fp['second']:3d}: {fp['from_count']}→{fp['to_count']}")
    
    if false_negatives:
        print(f"\n  Missed events:")
        for fn in false_negatives:
            print(f"    - {fn['event_type']:9s} at second {fn['second']:3d} ({fn['time']})")
    
    print(f"\n✅ Saved: {output_path}")
    print("="*70)
    
    return {
        'file': csv_base,
        'matched': matched,
        'fp': len(false_positives),
        'fn': len(false_negatives),
        'total_events': len(events),
        'annotated_events': len(file_annotations)
    }


def main():
    # Load annotations
    print("Loading annotations...")
    annotations = load_annotations('data/Annotated_arrivals_departures.csv')
    
    # Get files with annotated events
    files_with_events = {}
    for ann in annotations:
        if ann['has_events']:
            if ann['csv_base'] not in files_with_events:
                files_with_events[ann['csv_base']] = []
            files_with_events[ann['csv_base']].append(ann)
    
    print(f"Found {len(files_with_events)} files with annotated events")
    
    # Use optimized parameters from tuning
    params = {
        'edge_margin': 100,
        'max_spike_duration': 2,
        'max_dip_duration': 3,
        'min_stable_duration': 15,
        'max_stable_variation': 0
    }
    
    print(f"\nUsing optimized parameters:")
    for k, v in params.items():
        print(f"  {k}: {v}")
    
    print(f"\n{'='*70}")
    print("Generating visualizations for all files with events...")
    print('='*70)
    
    # Visualize all files
    results = []
    for csv_base in sorted(files_with_events.keys()):
        result = visualize_detection_performance(
            csv_base, 
            annotations,
            **params
        )
        if result:
            results.append(result)
    
    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print('='*70)
    
    total_matched = sum(r['matched'] for r in results)
    total_fp = sum(r['fp'] for r in results)
    total_fn = sum(r['fn'] for r in results)
    total_detected = sum(r['total_events'] for r in results)
    total_annotated = sum(r['annotated_events'] for r in results)
    
    print(f"\nOverall performance:")
    print(f"  Total annotated events: {total_annotated}")
    print(f"  Total detected events: {total_detected}")
    print(f"  Matched: {total_matched}")
    print(f"  False Positives: {total_fp}")
    print(f"  False Negatives: {total_fn}")
    
    if total_annotated > 0:
        recall = total_matched / total_annotated
        print(f"  Recall: {recall*100:.1f}%")
    
    if total_detected > 0:
        precision = total_matched / total_detected
        print(f"  Precision: {precision*100:.1f}%")
    
    print(f"\n✅ All visualizations saved to: plots/detection_analysis/")
    print(f"   Generated {len(results)} plots")
    
    # Identify problematic files
    print(f"\n{'='*70}")
    print("FILES WITH ISSUES:")
    print('='*70)
    
    for r in sorted(results, key=lambda x: x['fn'] + x['fp'], reverse=True):
        if r['fn'] > 0 or r['fp'] > 1:
            print(f"\n{r['file']}:")
            print(f"  Detected {r['total_events']}, Expected {r['annotated_events']}")
            print(f"  Matched: {r['matched']}, FP: {r['fp']}, FN: {r['fn']}")


if __name__ == '__main__':
    main()
