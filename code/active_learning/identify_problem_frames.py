"""
Active Learning Frame Selection Strategy

This module identifies problematic frames that would benefit model retraining:
1. Frames with spike detections (sudden count increases that disappear)
2. Frames with dip detections (sudden count decreases that recover)
3. Frames with edge detections (detections near frame borders)
4. Frames around event transitions (where count changes occur)

By annotating and retraining on these "hard examples", we can improve the base
object detection model, which will in turn improve event detection.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'postprocess'))

from event_detector import load_detections, aggregate_per_second
from detection_filter import remove_isolated_spikes, remove_isolated_dips
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import json


class ProblemFrameIdentifier:
    """
    Identifies frames that are likely causing detection problems.
    These frames are candidates for annotation and model retraining.
    """
    
    def __init__(self, frame_width=2688, frame_height=1520):
        self.frame_width = frame_width
        self.frame_height = frame_height
        
    def identify_problematic_frames(self, csv_path, edge_margin=100, 
                                   max_spike_duration=2, max_dip_duration=3):
        """
        Analyze a detection CSV and identify problematic frames.
        
        Returns:
            Dictionary with different categories of problematic frames
        """
        
        # Load detections
        df, start_time = load_detections(csv_path, conf_thresh=0.25, filter_edges=False,
                                        frame_width=self.frame_width, 
                                        frame_height=self.frame_height)
        
        if len(df) == 0:
            return None
        
        # Aggregate to per-second
        per_sec_df, _, _, _ = aggregate_per_second(
            df, fps=1, classes=('adult','chick','fish'),
            original_video_fps=25, start_time=start_time
        )
        
        problems = {
            'video_file': Path(csv_path).stem.replace('_raw', ''),
            'start_time': start_time.isoformat(),
            'duration_seconds': len(per_sec_df),
            'edge_detections': [],
            'spike_frames': [],
            'dip_frames': [],
            'high_count_frames': [],
            'count_transition_frames': [],
            'fish_frames': []
        }
        
        # 1. Find edge detections
        edge_detections = self._find_edge_detections(df, edge_margin)
        if len(edge_detections) > 0:
            problems['edge_detections'] = edge_detections
        
        # 2. Find spike frames (sudden increases that disappear)
        spike_frames = self._find_spike_frames(per_sec_df, max_spike_duration)
        if len(spike_frames) > 0:
            problems['spike_frames'] = spike_frames
        
        # 3. Find dip frames (sudden decreases that recover)
        dip_frames = self._find_dip_frames(per_sec_df, max_dip_duration)
        if len(dip_frames) > 0:
            problems['dip_frames'] = dip_frames
        
        # 4. Find frames with high bird counts (more detections = more potential errors)
        high_count_frames = self._find_high_count_frames(per_sec_df, threshold=10)
        if len(high_count_frames) > 0:
            problems['high_count_frames'] = high_count_frames
        
        # 5. Find count transition frames (where count changes - important for events)
        transition_frames = self._find_count_transitions(per_sec_df)
        if len(transition_frames) > 0:
            problems['count_transition_frames'] = transition_frames
        
        # 6. Find fish detections (class = 'fish')
        fish_frames = self._find_fish_detections(df, fish_conf_thresh=0.25)
        if len(fish_frames) > 0:
            problems['fish_frames'] = fish_frames
        
        return problems
    
    def _find_edge_detections(self, df, edge_margin):
        """Find frames with detections near frame borders"""
        edge_mask = (
            (df['xmin'] < edge_margin) | 
            (df['xmax'] > self.frame_width - edge_margin) |
            (df['ymin'] < edge_margin) | 
            (df['ymax'] > self.frame_height - edge_margin)
        )
        
        edge_df = df[edge_mask].copy()
        
        if len(edge_df) == 0:
            return []
        
        # Group by second and count edge detections
        edge_df['second'] = edge_df['frame'] // 25
        edge_by_second = edge_df.groupby('second').size().reset_index(name='edge_count')
        
        results = []
        for _, row in edge_by_second.iterrows():
            if row['edge_count'] > 0:
                results.append({
                    'second': int(row['second']),
                    'edge_detection_count': int(row['edge_count']),
                    'reason': 'edge_detection'
                })
        
        return results
    
    def _find_spike_frames(self, per_sec_df, max_spike_duration):
        """Find frames that are part of isolated spikes"""
        counts = per_sec_df['count_adult'].values
        
        # Compare before and after spike removal
        counts_no_spikes = remove_isolated_spikes(counts, max_spike_duration)
        
        # Find where values changed (these are spike frames)
        spike_mask = counts != counts_no_spikes
        spike_seconds = np.where(spike_mask)[0]
        
        results = []
        for second in spike_seconds:
            results.append({
                'second': int(second),
                'original_count': int(counts[second]),
                'filtered_count': int(counts_no_spikes[second]),
                'reason': 'spike'
            })
        
        return results
    
    def _find_dip_frames(self, per_sec_df, max_dip_duration):
        """Find frames that are part of isolated dips"""
        counts = per_sec_df['count_adult'].values
        counts_no_spikes = remove_isolated_spikes(counts, 2)
        
        # Compare before and after dip filling
        counts_filtered = remove_isolated_dips(counts_no_spikes, max_dip_duration)
        
        # Find where values changed (these are dip frames)
        dip_mask = counts_no_spikes != counts_filtered
        dip_seconds = np.where(dip_mask)[0]
        
        results = []
        for second in dip_seconds:
            results.append({
                'second': int(second),
                'original_count': int(counts_no_spikes[second]),
                'filtered_count': int(counts_filtered[second]),
                'reason': 'dip'
            })
        
        return results
    
    def _find_high_count_frames(self, per_sec_df, threshold=10):
        """Find frames with high bird counts (more birds = more potential errors)"""
        counts = per_sec_df['count_adult'].values
        high_mask = counts >= threshold
        high_seconds = np.where(high_mask)[0]
        
        results = []
        for second in high_seconds:
            results.append({
                'second': int(second),
                'count': int(counts[second]),
                'reason': 'high_count'
            })
        
        return results
    
    def _find_count_transitions(self, per_sec_df):
        """Find frames where bird count changes (important for event detection)"""
        counts = per_sec_df['count_adult'].values
        
        # Find where count changes
        count_changes = np.diff(counts)
        transition_seconds = np.where(count_changes != 0)[0]
        
        results = []
        for second in transition_seconds:
            if second + 1 < len(counts):
                results.append({
                    'second': int(second),
                    'from_count': int(counts[second]),
                    'to_count': int(counts[second + 1]),
                    'change': int(counts[second + 1] - counts[second]),
                    'reason': 'count_transition'
                })
        
        return results
    
    def _find_fish_detections(self, df, fish_conf_thresh=0.25):
        """Find frames with fish detections (class = 'fish')"""
        # Filter for fish detections with confidence threshold
        fish_df = df[(df['class'] == 'fish') & (df['confidence'] >= fish_conf_thresh)].copy()
        
        if len(fish_df) == 0:
            return []
        
        # Group by second and count fish detections
        fish_df['second'] = fish_df['frame'] // 25
        fish_by_second = fish_df.groupby('second').size().reset_index(name='fish_count')
        
        results = []
        for _, row in fish_by_second.iterrows():
            results.append({
                'second': int(row['second']),
                'fish_count': int(row['fish_count']),
                'reason': 'fish'
            })
        
        return results


def analyze_directory(csv_dir='csv_detection_1fps', output_file='data/problematic_frames.json'):
    """
    Analyze all CSV files in a directory and identify problematic frames.
    """
    
    identifier = ProblemFrameIdentifier()
    
    csv_dir = Path(csv_dir)
    csv_files = sorted(csv_dir.glob('*_raw.csv'))
    
    print(f"Analyzing {len(csv_files)} CSV files...")
    print("="*80)
    
    all_problems = []
    
    for i, csv_path in enumerate(csv_files, 1):
        print(f"[{i}/{len(csv_files)}] Processing {csv_path.stem}...", end=' ')
        
        problems = identifier.identify_problematic_frames(str(csv_path))
        
        if problems:
            # Count total problems
            total_problems = (
                len(problems['edge_detections']) +
                len(problems['spike_frames']) +
                len(problems['dip_frames']) +
                len(problems['high_count_frames']) +
                len(problems['count_transition_frames']) +
                len(problems['fish_frames'])
            )
            
            print(f"✓ Found {total_problems} problematic frames")
            all_problems.append(problems)
        else:
            print("✗ No detections")
    
    # Save results
    output_path = Path(output_file)
    output_path.parent.mkdir(exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(all_problems, f, indent=2)
    
    print("\n" + "="*80)
    print(f"✅ Saved analysis to: {output_path}")
    
    # Print summary statistics
    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)
    
    total_edge = sum(len(p['edge_detections']) for p in all_problems)
    total_spikes = sum(len(p['spike_frames']) for p in all_problems)
    total_dips = sum(len(p['dip_frames']) for p in all_problems)
    total_high_count = sum(len(p['high_count_frames']) for p in all_problems)
    total_transitions = sum(len(p['count_transition_frames']) for p in all_problems)
    total_fish = sum(len(p['fish_frames']) for p in all_problems)
    
    print(f"\nFiles analyzed: {len(csv_files)}")
    print(f"Files with problems: {len(all_problems)}")
    print(f"\nProblem types:")
    print(f"  Edge detections: {total_edge} frames")
    print(f"  Spike frames: {total_spikes} frames")
    print(f"  Dip frames: {total_dips} frames")
    print(f"  High count frames: {total_high_count} frames")
    print(f"  Count transitions: {total_transitions} frames")
    print(f"  Fish detections: {total_fish} frames")
    print(f"\nTotal problematic frames: {total_edge + total_spikes + total_dips + total_high_count + total_transitions + total_fish}"))
    
    return all_problems


if __name__ == '__main__':
    analyze_directory()
