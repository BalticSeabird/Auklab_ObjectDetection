#!/usr/bin/env python3
"""
visualize_edge_filtering.py

Visualize the effect of edge filtering on detection counts.
Shows how removing edge detections affects the count signal and reduces flickering.

Usage:
    python3 code/postprocess/visualize_edge_filtering.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from pathlib import Path

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from event_detector import load_detections, aggregate_per_second


def visualize_edge_filtering_effect(csv_path, frame_width=2688, frame_height=1520, edge_margin=50):
    """
    Create visualization showing the effect of edge filtering.
    
    Shows 4 panels:
    1. Raw counts (no edge filtering)
    2. Filtered counts (edge filtering enabled)
    3. Difference (raw - filtered)
    4. Scatter plot of edge vs non-edge detections
    """
    print(f"\n{'='*80}")
    print(f"Analyzing: {os.path.basename(csv_path)}")
    print(f"{'='*80}")
    
    # Load WITHOUT edge filtering
    df_raw, start_time = load_detections(csv_path, conf_thresh=0.25, filter_edges=False)
    print(f"Raw detections: {len(df_raw)}")
    
    # Load WITH edge filtering
    df_filtered, _ = load_detections(csv_path, conf_thresh=0.25, filter_edges=True,
                                     frame_width=frame_width, frame_height=frame_height,
                                     edge_margin=edge_margin)
    print(f"Filtered detections: {len(df_filtered)}")
    print(f"Edge detections removed: {len(df_raw) - len(df_filtered)} ({(len(df_raw) - len(df_filtered)) / len(df_raw) * 100:.1f}%)")
    
    # Identify edge detections
    if len(df_raw) > 0:
        edge_mask = (
            (df_raw['xmin'] < edge_margin) |
            (df_raw['xmax'] > frame_width - edge_margin) |
            (df_raw['ymin'] < edge_margin) |
            (df_raw['ymax'] > frame_height - edge_margin)
        )
        df_edge = df_raw[edge_mask].copy()
        print(f"Edge detections (within {edge_margin}px of borders): {len(df_edge)}")
    else:
        df_edge = df_raw.copy()
    
    # Aggregate to per-second counts
    per_sec_raw, _, _, counts_raw = aggregate_per_second(
        df_raw, fps=1, classes=('adult', 'chick', 'fish'),
        original_video_fps=25, start_time=start_time
    )
    
    per_sec_filtered, _, _, counts_filtered = aggregate_per_second(
        df_filtered, fps=1, classes=('adult', 'chick', 'fish'),
        original_video_fps=25, start_time=start_time
    )
    
    # Get adult counts
    adult_counts_raw = counts_raw['adult']
    adult_counts_filtered = counts_filtered['adult']
    seconds = np.arange(len(adult_counts_raw))
    
    # Calculate difference
    count_difference = adult_counts_raw - adult_counts_filtered
    
    print(f"\nCount statistics:")
    print(f"  Raw: mean={adult_counts_raw.mean():.2f}, max={adult_counts_raw.max()}")
    print(f"  Filtered: mean={adult_counts_filtered.mean():.2f}, max={adult_counts_filtered.max()}")
    print(f"  Difference: mean={count_difference.mean():.2f}, max={count_difference.max()}")
    
    # Count flickering (sudden changes)
    def count_flickering(counts):
        """Count number of times count changes by more than 1"""
        if len(counts) < 2:
            return 0
        changes = np.abs(np.diff(counts))
        return np.sum(changes > 1)
    
    flicker_raw = count_flickering(adult_counts_raw)
    flicker_filtered = count_flickering(adult_counts_filtered)
    
    print(f"\nFlickering events (changes > 1):")
    print(f"  Raw: {flicker_raw}")
    print(f"  Filtered: {flicker_filtered}")
    print(f"  Reduction: {flicker_raw - flicker_filtered} ({(flicker_raw - flicker_filtered) / flicker_raw * 100 if flicker_raw > 0 else 0:.1f}%)")
    
    # Create visualization
    fig, axes = plt.subplots(4, 1, figsize=(16, 12))
    fig.suptitle(f'Edge Filtering Effect: {os.path.basename(csv_path)}', fontsize=14, fontweight='bold')
    
    # Panel 1: Raw counts
    ax = axes[0]
    ax.plot(seconds, adult_counts_raw, 'b-', linewidth=1, alpha=0.7, label='Raw counts')
    ax.set_ylabel('Adult Count', fontsize=11)
    ax.set_title(f'Raw Counts (no filtering) - Mean: {adult_counts_raw.mean():.2f}, Flicker events: {flicker_raw}', fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Panel 2: Filtered counts
    ax = axes[1]
    ax.plot(seconds, adult_counts_filtered, 'g-', linewidth=1, alpha=0.7, label='Filtered counts')
    ax.set_ylabel('Adult Count', fontsize=11)
    ax.set_title(f'Filtered Counts (edge margin={edge_margin}px) - Mean: {adult_counts_filtered.mean():.2f}, Flicker events: {flicker_filtered}', fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Panel 3: Difference
    ax = axes[2]
    ax.plot(seconds, count_difference, 'r-', linewidth=1, alpha=0.7, label='Difference (raw - filtered)')
    ax.fill_between(seconds, 0, count_difference, where=(count_difference > 0), alpha=0.3, color='red', label='Edge detections')
    ax.set_ylabel('Count Difference', fontsize=11)
    ax.set_xlabel('Time (seconds)', fontsize=11)
    ax.set_title(f'Impact of Edge Filtering - Removed {len(df_edge)} edge detections ({(len(df_edge) / len(df_raw) * 100 if len(df_raw) > 0 else 0):.1f}%)', fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Panel 4: Spatial distribution of edge detections
    ax = axes[3]
    if len(df_raw) > 0:
        # Plot all detections
        df_non_edge = df_raw[~edge_mask]
        if len(df_non_edge) > 0:
            x_center_ne = (df_non_edge['xmin'] + df_non_edge['xmax']) / 2
            y_center_ne = (df_non_edge['ymin'] + df_non_edge['ymax']) / 2
            ax.scatter(x_center_ne, y_center_ne, c='blue', s=10, alpha=0.3, label=f'Non-edge ({len(df_non_edge)})')
        
        # Plot edge detections
        if len(df_edge) > 0:
            x_center_e = (df_edge['xmin'] + df_edge['xmax']) / 2
            y_center_e = (df_edge['ymin'] + df_edge['ymax']) / 2
            ax.scatter(x_center_e, y_center_e, c='red', s=20, alpha=0.6, label=f'Edge ({len(df_edge)})')
        
        # Draw frame boundaries
        ax.axvline(edge_margin, color='orange', linestyle='--', linewidth=2, alpha=0.7, label=f'Edge margin ({edge_margin}px)')
        ax.axvline(frame_width - edge_margin, color='orange', linestyle='--', linewidth=2, alpha=0.7)
        ax.axhline(edge_margin, color='orange', linestyle='--', linewidth=2, alpha=0.7)
        ax.axhline(frame_height - edge_margin, color='orange', linestyle='--', linewidth=2, alpha=0.7)
        
        # Draw frame edges
        ax.axvline(0, color='black', linestyle='-', linewidth=1)
        ax.axvline(frame_width, color='black', linestyle='-', linewidth=1)
        ax.axhline(0, color='black', linestyle='-', linewidth=1)
        ax.axhline(frame_height, color='black', linestyle='-', linewidth=1)
    
    ax.set_xlim(-50, frame_width + 50)
    ax.set_ylim(-50, frame_height + 50)
    ax.set_xlabel('X position (pixels)', fontsize=11)
    ax.set_ylabel('Y position (pixels)', fontsize=11)
    ax.set_title(f'Spatial Distribution of Detections (Frame: {frame_width}x{frame_height})', fontsize=11)
    ax.legend(loc='upper right')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.invert_yaxis()  # Match image coordinates (0,0 at top-left)
    
    plt.tight_layout()
    
    return fig, {
        'raw_count': len(df_raw),
        'filtered_count': len(df_filtered),
        'edge_count': len(df_edge),
        'flicker_raw': flicker_raw,
        'flicker_filtered': flicker_filtered,
        'mean_raw': adult_counts_raw.mean(),
        'mean_filtered': adult_counts_filtered.mean()
    }


def main():
    # Create output directory
    os.makedirs('plots/edge_filtering', exist_ok=True)
    
    # Load annotations to select interesting files
    annotations_file = 'data/Annotated_arrivals_departures.csv'
    if os.path.exists(annotations_file):
        df_ann = pd.read_csv(annotations_file, sep=';')
        
        # Get unique files with events
        import re
        files_with_events = []
        for filename in df_ann['File_name'].dropna().unique():
            match = re.search(r'(TRI3_\d{8}T\d{6})\.mkv', filename)
            if match:
                files_with_events.append(match.group(1))
        
        print(f"Found {len(files_with_events)} files with annotated events")
        
        # Select a diverse set for visualization
        # Pick files with different activity levels
        selected_files = [
            'TRI3_20250628T011000',  # Low activity
            'TRI3_20250628T030002',  # High activity with many events
            'TRI3_20250628T033000',  # Very active
            'TRI3_20250628T045002',  # Rapid arrivals
        ]
    else:
        # Default selection
        selected_files = [
            'TRI3_20250628T011000',
            'TRI3_20250628T030002',
            'TRI3_20250628T033000',
            'TRI3_20250628T045002',
        ]
    
    print(f"\n{'='*80}")
    print("EDGE FILTERING VISUALIZATION")
    print(f"{'='*80}")
    print(f"Analyzing {len(selected_files)} files to visualize edge filtering effects")
    print(f"Output directory: plots/edge_filtering/")
    
    summary_stats = []
    
    for csv_base in selected_files:
        csv_path = f'csv_detection_1fps/{csv_base}_raw.csv'
        
        if not os.path.exists(csv_path):
            print(f"\n⚠️  Skipping {csv_base}: file not found")
            continue
        
        # Generate visualization
        fig, stats = visualize_edge_filtering_effect(csv_path)
        
        # Save plot
        output_path = f'plots/edge_filtering/{csv_base}_edge_filtering.png'
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        
        print(f"✅ Saved visualization: {output_path}")
        
        stats['file'] = csv_base
        summary_stats.append(stats)
    
    # Print summary
    print(f"\n{'='*80}")
    print("SUMMARY STATISTICS")
    print(f"{'='*80}")
    
    df_summary = pd.DataFrame(summary_stats)
    if len(df_summary) > 0:
        print("\nDetection counts:")
        print(df_summary[['file', 'raw_count', 'filtered_count', 'edge_count']].to_string(index=False))
        
        print("\nFlickering reduction:")
        print(df_summary[['file', 'flicker_raw', 'flicker_filtered']].to_string(index=False))
        
        print("\nMean adult counts:")
        print(df_summary[['file', 'mean_raw', 'mean_filtered']].to_string(index=False))
        
        # Overall statistics
        total_raw = df_summary['raw_count'].sum()
        total_edge = df_summary['edge_count'].sum()
        total_flicker_reduction = df_summary['flicker_raw'].sum() - df_summary['flicker_filtered'].sum()
        
        print(f"\n📊 OVERALL:")
        print(f"   Total detections: {total_raw}")
        print(f"   Edge detections: {total_edge} ({total_edge / total_raw * 100:.1f}%)")
        print(f"   Flicker events reduced: {total_flicker_reduction}")
        print(f"\n✅ Visualizations saved to plots/edge_filtering/")


if __name__ == "__main__":
    main()
