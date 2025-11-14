#!/usr/bin/env python3
"""
Scalable Active Learning Pipeline

This script orchestrates the entire active learning workflow with support for:
- Large-scale datasets
- Random sampling of stations/dates
- Configurable paths via YAML config or command-line arguments
- Multi-directory video sources

Usage:
    # Using config file
    python run_active_learning_pipeline.py --config config.yaml
    
    # Or with command-line arguments
    python run_active_learning_pipeline.py \
        --csv-dir /path/to/csvs \
        --video-dirs /path/to/videos \
        --output-dir data/active_learning_batch \
        --sample-size 100

    # Run specific steps only
    python run_active_learning_pipeline.py --config config.yaml --steps identify extract
"""

import sys
import os
import argparse
import yaml
import json
import random
import re
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional
import numpy as np

# Add parent directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'postprocess'))

from event_detector import load_detections, aggregate_per_second
from detection_filter import remove_isolated_spikes, remove_isolated_dips


class ScalableActiveLearningSampler:
    """
    Manages sampling and discovery of CSV/video files across large datasets.
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.csv_dir = Path(config['paths']['csv_dir'])
        self.video_dirs = [Path(d) for d in config['paths'].get('video_dirs', [])]
        self.sampling_config = config.get('sampling', {})
        
    def discover_csv_files(self) -> List[Path]:
        """
        Discover all CSV files in the directory tree.
        """
        print("\n" + "="*80)
        print("DISCOVERING CSV FILES")
        print("="*80)
        print(f"Searching in: {self.csv_dir}")
        
        # Find all CSV files (including in subdirectories)
        csv_files = list(self.csv_dir.rglob("*_raw.csv"))
        
        print(f"Found {len(csv_files)} CSV files")
        
        # Apply filters if specified
        csv_files = self._apply_filters(csv_files)
        
        # Apply sampling if enabled
        if self.sampling_config.get('enabled', False):
            csv_files = self._apply_sampling(csv_files)
        
        print(f"Selected {len(csv_files)} files for processing")
        return sorted(csv_files)
    
    def _apply_filters(self, files: List[Path]) -> List[Path]:
        """Apply include/exclude pattern filters."""
        include_patterns = self.sampling_config.get('include_patterns', [])
        exclude_patterns = self.sampling_config.get('exclude_patterns', [])
        
        if include_patterns:
            print(f"Applying include filters: {include_patterns}")
            filtered = []
            for f in files:
                if any(re.search(pattern, str(f)) for pattern in include_patterns):
                    filtered.append(f)
            files = filtered
            print(f"  → {len(files)} files after include filter")
        
        if exclude_patterns:
            print(f"Applying exclude filters: {exclude_patterns}")
            filtered = []
            for f in files:
                if not any(re.search(pattern, str(f)) for pattern in exclude_patterns):
                    filtered.append(f)
            files = filtered
            print(f"  → {len(files)} files after exclude filter")
        
        return files
    
    def _apply_sampling(self, files: List[Path]) -> List[Path]:
        """Apply random sampling to file list."""
        n_files = self.sampling_config.get('n_files')
        
        if n_files is None or n_files >= len(files):
            print("Sampling disabled or sample size >= total files - using all files")
            return files
        
        seed = self.sampling_config.get('random_seed', 42)
        random.seed(seed)
        
        print(f"Random sampling: {n_files} files (seed={seed})")
        sampled = random.sample(files, n_files)
        
        return sampled
    
    def build_video_index(self) -> Dict[str, Path]:
        """
        Build an index of video files for fast lookup.
        Returns dict mapping basename -> video_path
        """
        print("\n" + "="*80)
        print("BUILDING VIDEO INDEX")
        print("="*80)
        
        video_index = {}
        extensions = self.config.get('extraction', {}).get('video_extensions', 
                                                           ['.mkv', '.mp4', '.avi', '.mov'])
        
        for video_dir in self.video_dirs:
            print(f"Searching: {video_dir}")
            
            if not video_dir.exists():
                print(f"  ⚠ Directory not found, skipping")
                continue
            
            for ext in extensions:
                for video_path in video_dir.rglob(f"*{ext}"):
                    basename = video_path.stem
                    if basename not in video_index:
                        video_index[basename] = video_path
        
        print(f"Indexed {len(video_index)} video files")
        return video_index


class ScalableProblemFrameIdentifier:
    """
    Identifies problematic frames from CSV files with scalable processing.
    """
    
    def __init__(self, config: Dict):
        self.config = config
        detection_config = config.get('detection', {})
        self.frame_width = detection_config.get('frame_width', 2688)
        self.frame_height = detection_config.get('frame_height', 1520)
        self.conf_thresh = detection_config.get('confidence_threshold', 0.25)
        self.fish_conf_thresh = detection_config.get('fish_confidence_threshold', 0.25)
        self.edge_margin = detection_config.get('edge_margin', 100)
        self.max_spike_duration = detection_config.get('max_spike_duration', 2)
        self.max_dip_duration = detection_config.get('max_dip_duration', 3)
        self.high_count_threshold = detection_config.get('high_count_threshold', 10)
        
    def analyze_csv_files(self, csv_files: List[Path], output_file: Path):
        """
        Analyze multiple CSV files and identify problematic frames.
        """
        print("\n" + "="*80)
        print("ANALYZING CSV FILES FOR PROBLEMATIC FRAMES")
        print("="*80)
        print(f"Files to analyze: {len(csv_files)}")
        
        all_problems = []
        failed_files = []
        
        for i, csv_path in enumerate(csv_files, 1):
            print(f"[{i}/{len(csv_files)}] {csv_path.name}...", end=' ')
            
            try:
                problems = self._identify_problematic_frames(csv_path)
                
                if problems:
                    total_problems = sum(len(problems[key]) for key in 
                                       ['edge_detections', 'spike_frames', 'dip_frames',
                                        'high_count_frames', 'count_transition_frames', 'fish_frames'])
                    print(f"✓ {total_problems} problems")
                    all_problems.append(problems)
                else:
                    print("✗ No detections")
                    
            except Exception as e:
                print(f"✗ Error: {e}")
                failed_files.append((csv_path, str(e)))
        
        # Save results
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, 'w') as f:
            json.dump(all_problems, f, indent=2)
        
        # Print summary
        self._print_summary(all_problems, failed_files, output_file)
        
        return all_problems
    
    def _identify_problematic_frames(self, csv_path: Path) -> Optional[Dict]:
        """Analyze a single CSV file."""
        
        # Load detections
        df, start_time = load_detections(
            str(csv_path), 
            conf_thresh=self.conf_thresh, 
            filter_edges=False,
            frame_width=self.frame_width, 
            frame_height=self.frame_height
        )
        
        if len(df) == 0:
            return None
        
        # Aggregate to per-second
        per_sec_df, _, _, _ = aggregate_per_second(
            df, fps=1, classes=('adult','chick','fish'),
            original_video_fps=25, start_time=start_time
        )
        
        problems = {
            'video_file': csv_path.stem.replace('_raw', ''),
            'csv_path': str(csv_path.relative_to(csv_path.parents[len(csv_path.parents)-1])),
            'start_time': start_time.isoformat(),
            'duration_seconds': len(per_sec_df),
            'edge_detections': self._find_edge_detections(df),
            'spike_frames': self._find_spike_frames(per_sec_df),
            'dip_frames': self._find_dip_frames(per_sec_df),
            'high_count_frames': self._find_high_count_frames(per_sec_df),
            'count_transition_frames': self._find_count_transitions(per_sec_df),
            'fish_frames': self._find_fish_detections(df, per_sec_df)
        }
        
        return problems
    
    def _find_edge_detections(self, df):
        """Find frames with detections near frame borders."""
        edge_mask = (
            (df['xmin'] < self.edge_margin) | 
            (df['xmax'] > self.frame_width - self.edge_margin) |
            (df['ymin'] < self.edge_margin) | 
            (df['ymax'] > self.frame_height - self.edge_margin)
        )
        
        edge_df = df[edge_mask].copy()
        
        if len(edge_df) == 0:
            return []
        
        edge_df['second'] = edge_df['frame'] // 25
        edge_by_second = edge_df.groupby('second').size().reset_index(name='edge_count')
        
        return [{'second': int(row['second']), 
                'edge_detection_count': int(row['edge_count']),
                'reason': 'edge_detection'}
                for _, row in edge_by_second.iterrows() if row['edge_count'] > 0]
    
    def _find_spike_frames(self, per_sec_df):
        """Find frames that are part of isolated spikes."""
        counts = per_sec_df['count_adult'].values
        counts_no_spikes = remove_isolated_spikes(counts, self.max_spike_duration)
        
        spike_mask = counts != counts_no_spikes
        spike_seconds = np.where(spike_mask)[0]
        
        return [{'second': int(second),
                'original_count': int(counts[second]),
                'filtered_count': int(counts_no_spikes[second]),
                'reason': 'spike'}
                for second in spike_seconds]
    
    def _find_dip_frames(self, per_sec_df):
        """Find frames that are part of isolated dips."""
        counts = per_sec_df['count_adult'].values
        counts_no_spikes = remove_isolated_spikes(counts, 2)
        counts_filtered = remove_isolated_dips(counts_no_spikes, self.max_dip_duration)
        
        dip_mask = counts_no_spikes != counts_filtered
        dip_seconds = np.where(dip_mask)[0]
        
        return [{'second': int(second),
                'original_count': int(counts_no_spikes[second]),
                'filtered_count': int(counts_filtered[second]),
                'reason': 'dip'}
                for second in dip_seconds]
    
    def _find_high_count_frames(self, per_sec_df):
        """Find frames with high bird counts."""
        counts = per_sec_df['count_adult'].values
        high_mask = counts >= self.high_count_threshold
        high_seconds = np.where(high_mask)[0]
        
        return [{'second': int(second),
                'count': int(counts[second]),
                'reason': 'high_count'}
                for second in high_seconds]
    
    def _find_count_transitions(self, per_sec_df):
        """Find frames where bird count changes."""
        counts = per_sec_df['count_adult'].values
        count_changes = np.diff(counts)
        transition_seconds = np.where(count_changes != 0)[0]
        
        return [{'second': int(second),
                'from_count': int(counts[second]),
                'to_count': int(counts[second + 1]),
                'change': int(counts[second + 1] - counts[second]),
                'reason': 'count_transition'}
                for second in transition_seconds if second + 1 < len(counts)]
    
    def _find_fish_detections(self, df, per_sec_df):
        """Find frames with fish detections (class = 'fish')."""
        # Filter for fish detections with confidence threshold
        fish_df = df[(df['class'] == 'fish') & (df['confidence'] >= self.fish_conf_thresh)].copy()
        
        if len(fish_df) == 0:
            return []
        
        # Group by second and count fish detections
        fish_df['second'] = fish_df['frame'] // 25
        fish_by_second = fish_df.groupby('second').size().reset_index(name='fish_count')
        
        return [{'second': int(row['second']),
                'fish_count': int(row['fish_count']),
                'reason': 'fish'}
                for _, row in fish_by_second.iterrows()]
    
    def _print_summary(self, all_problems, failed_files, output_file):
        """Print analysis summary."""
        print("\n" + "="*80)
        print("ANALYSIS SUMMARY")
        print("="*80)
        
        if failed_files:
            print(f"\n⚠ Failed to process {len(failed_files)} files:")
            for path, error in failed_files[:5]:  # Show first 5
                print(f"  - {path.name}: {error}")
            if len(failed_files) > 5:
                print(f"  ... and {len(failed_files)-5} more")
        
        total_edge = sum(len(p['edge_detections']) for p in all_problems)
        total_spikes = sum(len(p['spike_frames']) for p in all_problems)
        total_dips = sum(len(p['dip_frames']) for p in all_problems)
        total_high_count = sum(len(p['high_count_frames']) for p in all_problems)
        total_transitions = sum(len(p['count_transition_frames']) for p in all_problems)
        total_fish = sum(len(p['fish_frames']) for p in all_problems)
        
        print(f"\nFiles with problems: {len(all_problems)}")
        print(f"\nProblem types:")
        print(f"  Edge detections:    {total_edge:6d} frames")
        print(f"  Spike frames:       {total_spikes:6d} frames")
        print(f"  Dip frames:         {total_dips:6d} frames")
        print(f"  High count frames:  {total_high_count:6d} frames")
        print(f"  Count transitions:  {total_transitions:6d} frames")
        print(f"  Fish detections:    {total_fish:6d} frames")
        print(f"\nTotal problematic:  {total_edge + total_spikes + total_dips + total_high_count + total_transitions + total_fish:6d} frames")
        print(f"\n✅ Saved to: {output_file}")


class ScalableFrameExtractor:
    """
    Extracts frames from videos with support for multiple video directories.
    """
    
    def __init__(self, config: Dict, video_index: Dict[str, Path]):
        self.config = config
        self.video_index = video_index
        self.output_dir = Path(config['paths']['output_dir']) / 'frames'
        self.extraction_config = config.get('extraction', {})
        
    def extract_frames(self, problems_file: Path):
        """Extract frames based on problem analysis."""
        import cv2
        from collections import defaultdict
        
        print("\n" + "="*80)
        print("EXTRACTING FRAMES")
        print("="*80)
        
        # Load problems
        with open(problems_file, 'r') as f:
            all_problems = json.load(f)
        
        # Organize frames by type
        frames_by_type = self._organize_frames_by_type(all_problems)
        
        # Select frames to extract
        max_per_type = self.extraction_config.get('max_per_type', 100)
        priority = self.extraction_config.get('priority', 'diverse')
        
        selected_frames = self._select_frames_to_extract(
            frames_by_type, max_per_type, priority
        )
        
        # Extract frames
        extraction_results = []
        frames_by_video = defaultdict(list)
        
        for frame_type, frames in selected_frames.items():
            for frame_info in frames:
                frame_info['frame_type'] = frame_type
                frames_by_video[frame_info['video_file']].append(frame_info)
        
        print(f"\nExtracting from {len(frames_by_video)} videos...")
        
        for i, (video_file, frames) in enumerate(sorted(frames_by_video.items()), 1):
            print(f"[{i}/{len(frames_by_video)}] {video_file}...", end=' ')
            
            video_path = self.video_index.get(video_file)
            if video_path is None:
                print("✗ Video not found")
                continue
            
            extracted = self._extract_frames_from_video(video_path, frames)
            extraction_results.extend(extracted)
            print(f"✓ {len(extracted)} frames")
        
        # Save manifest
        self._create_manifest(extraction_results)
        
        print(f"\n✅ Extracted {len(extraction_results)} frames total")
        return extraction_results
    
    def _organize_frames_by_type(self, all_problems):
        """Organize frames by problem type."""
        from datetime import datetime, timedelta
        
        frames_by_type = {
            'edge_detection': [],
            'spike': [],
            'dip': [],
            'high_count': [],
            'count_transition': [],
            'fish': []
        }
        
        for video_problems in all_problems:
            video_file = video_problems['video_file']
            start_time = datetime.fromisoformat(video_problems['start_time'])
            
            for frame_info in video_problems['edge_detections']:
                frames_by_type['edge_detection'].append({
                    'video_file': video_file,
                    'second': frame_info['second'],
                    'timestamp': (start_time + timedelta(seconds=frame_info['second'])).isoformat(),
                    'details': frame_info
                })
            
            for frame_info in video_problems['spike_frames']:
                frames_by_type['spike'].append({
                    'video_file': video_file,
                    'second': frame_info['second'],
                    'timestamp': (start_time + timedelta(seconds=frame_info['second'])).isoformat(),
                    'details': frame_info
                })
            
            for frame_info in video_problems['dip_frames']:
                frames_by_type['dip'].append({
                    'video_file': video_file,
                    'second': frame_info['second'],
                    'timestamp': (start_time + timedelta(seconds=frame_info['second'])).isoformat(),
                    'details': frame_info
                })
            
            for frame_info in video_problems['high_count_frames']:
                frames_by_type['high_count'].append({
                    'video_file': video_file,
                    'second': frame_info['second'],
                    'timestamp': (start_time + timedelta(seconds=frame_info['second'])).isoformat(),
                    'details': frame_info
                })
            
            for frame_info in video_problems['count_transition_frames']:
                frames_by_type['count_transition'].append({
                    'video_file': video_file,
                    'second': frame_info['second'],
                    'timestamp': (start_time + timedelta(seconds=frame_info['second'])).isoformat(),
                    'details': frame_info
                })
            
            for frame_info in video_problems['fish_frames']:
                frames_by_type['fish'].append({
                    'video_file': video_file,
                    'second': frame_info['second'],
                    'timestamp': (start_time + timedelta(seconds=frame_info['second'])).isoformat(),
                    'details': frame_info
                })
        
        print("\nAvailable frames by type:")
        for frame_type, frames in frames_by_type.items():
            print(f"  {frame_type:20s}: {len(frames):6d} frames")
        
        return frames_by_type
    
    def _select_frames_to_extract(self, frames_by_type, max_per_type, priority):
        """Select which frames to extract."""
        from collections import defaultdict
        
        # Check which types to extract (from config)
        extract_types = self.extraction_config.get('extract_types', {})
        
        selected = {}
        
        for frame_type, frames in frames_by_type.items():
            # Skip if this type is disabled in config
            if extract_types and not extract_types.get(frame_type, True):
                selected[frame_type] = []
                continue
            
            if len(frames) == 0:
                selected[frame_type] = []
                continue
            
            if priority == 'diverse':
                selected[frame_type] = self._diverse_selection(frames, max_per_type)
            else:
                selected[frame_type] = frames[:max_per_type]
        
        print("\nSelected for extraction:")
        for frame_type, frames in selected.items():
            print(f"  {frame_type:20s}: {len(frames):6d} frames")
        
        return selected
    
    def _diverse_selection(self, frames, max_frames):
        """Select frames diversely across different videos."""
        from collections import defaultdict
        
        by_video = defaultdict(list)
        for frame in frames:
            by_video[frame['video_file']].append(frame)
        
        selected = []
        videos = list(by_video.keys())
        frames_per_video = max(1, max_frames // len(videos))
        
        for video in videos:
            video_frames = by_video[video]
            if len(video_frames) <= frames_per_video:
                selected.extend(video_frames)
            else:
                indices = np.linspace(0, len(video_frames)-1, frames_per_video, dtype=int)
                selected.extend([video_frames[i] for i in indices])
        
        if len(selected) < max_frames:
            remaining = [f for f in frames if f not in selected]
            selected.extend(remaining[:max_frames - len(selected)])
        
        return selected[:max_frames]
    
    def _extract_frames_from_video(self, video_path, frames_to_extract):
        """Extract specific frames from a video."""
        import cv2
        
        extracted = []
        frames_to_extract = sorted(frames_to_extract, key=lambda x: x['second'])
        
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return extracted
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        for frame_info in frames_to_extract:
            second = frame_info['second']
            frame_number = int(second * fps)
            
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            ret, frame = cap.read()
            
            if not ret:
                continue
            
            frame_type = frame_info['frame_type']
            video_basename = video_path.stem
            
            output_filename = f"{video_basename}_s{second:04d}_{frame_type}.jpg"
            output_path = self.output_dir / frame_type / output_filename
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            cv2.imwrite(str(output_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
            
            extracted.append({
                'output_path': str(output_path.relative_to(self.output_dir)),
                'video_file': video_basename,
                'second': second,
                'frame_number': frame_number,
                'frame_type': frame_type,
                'timestamp': frame_info['timestamp'],
                'details': frame_info['details']
            })
        
        cap.release()
        return extracted
    
    def _create_manifest(self, extraction_results):
        """Create manifest file for extracted frames."""
        manifest = {
            'created': datetime.now().isoformat(),
            'total_frames': len(extraction_results),
            'frames_by_type': {},
            'frames': extraction_results
        }
        
        for frame in extraction_results:
            frame_type = frame['frame_type']
            if frame_type not in manifest['frames_by_type']:
                manifest['frames_by_type'][frame_type] = 0
            manifest['frames_by_type'][frame_type] += 1
        
        manifest_path = self.output_dir / 'extraction_manifest.json'
        with open(manifest_path, 'w') as f:
            json.dump(manifest, f, indent=2)
        
        # Create CSV
        csv_path = self.output_dir / 'frames_list.csv'
        with open(csv_path, 'w') as f:
            f.write("filename,video,second,frame_type,timestamp\n")
            for frame in extraction_results:
                f.write(f"{frame['output_path']},{frame['video_file']},{frame['second']},"
                       f"{frame['frame_type']},{frame['timestamp']}\n")


def load_config(config_path: Optional[Path] = None, args: Optional[argparse.Namespace] = None) -> Dict:
    """
    Load configuration from YAML file and/or command-line arguments.
    CLI arguments override config file values.
    """
    config = {}
    
    # Load from file if provided
    if config_path and config_path.exists():
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        print(f"Loaded configuration from: {config_path}")
    
    # Override with CLI arguments
    if args:
        if args.csv_dir:
            config.setdefault('paths', {})['csv_dir'] = args.csv_dir
        if args.video_dirs:
            config.setdefault('paths', {})['video_dirs'] = args.video_dirs
        if args.output_dir:
            config.setdefault('paths', {})['output_dir'] = args.output_dir
        if args.sample_size:
            config.setdefault('sampling', {})['enabled'] = True
            config['sampling']['n_files'] = args.sample_size
        if args.no_sampling:
            config.setdefault('sampling', {})['enabled'] = False
    
    return config


def main():
    parser = argparse.ArgumentParser(
        description='Scalable Active Learning Pipeline for Object Detection',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run full pipeline with config file
  python run_active_learning_pipeline.py --config config.yaml
  
  # Run with command-line arguments
  python run_active_learning_pipeline.py \\
      --csv-dir /path/to/csvs \\
      --video-dirs /path/to/videos \\
      --sample-size 100
  
  # Run specific steps only
  python run_active_learning_pipeline.py --config config.yaml --steps identify extract
        """
    )
    
    parser.add_argument('--config', type=str, help='Path to YAML configuration file')
    parser.add_argument('--csv-dir', type=str, help='Directory containing CSV detection files')
    parser.add_argument('--video-dirs', type=str, nargs='+', help='Directories containing video files')
    parser.add_argument('--output-dir', type=str, help='Output directory')
    parser.add_argument('--sample-size', type=int, help='Number of files to randomly sample')
    parser.add_argument('--no-sampling', action='store_true', help='Disable sampling, process all files')
    parser.add_argument('--steps', nargs='+', choices=['identify', 'extract', 'annotate'],
                       default=['identify', 'extract', 'annotate'],
                       help='Pipeline steps to run')
    
    args = parser.parse_args()
    
    # Load configuration
    config_path = Path(args.config) if args.config else None
    config = load_config(config_path, args)
    
    # Validate configuration
    if not config.get('paths', {}).get('csv_dir'):
        print("Error: csv_dir must be specified in config file or via --csv-dir")
        return 1
    
    if not config.get('paths', {}).get('video_dirs'):
        print("Error: video_dirs must be specified in config file or via --video-dirs")
        return 1
    
    output_dir = Path(config['paths']['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*80)
    print("SCALABLE ACTIVE LEARNING PIPELINE")
    print("="*80)
    print(f"CSV directory: {config['paths']['csv_dir']}")
    print(f"Video directories: {config['paths']['video_dirs']}")
    print(f"Output directory: {output_dir}")
    print(f"Steps to run: {', '.join(args.steps)}")
    
    # Step 1: Discover and sample files
    sampler = ScalableActiveLearningSampler(config)
    csv_files = sampler.discover_csv_files()
    
    if len(csv_files) == 0:
        print("\n⚠ No CSV files found to process!")
        return 1
    
    # Build video index (needed for extraction)
    if 'extract' in args.steps:
        video_index = sampler.build_video_index()
    else:
        video_index = {}
    
    # Step 2: Identify problematic frames
    if 'identify' in args.steps:
        identifier = ScalableProblemFrameIdentifier(config)
        problems_file = output_dir / 'problematic_frames.json'
        all_problems = identifier.analyze_csv_files(csv_files, problems_file)
    else:
        problems_file = output_dir / 'problematic_frames.json'
        if not problems_file.exists():
            print(f"\n⚠ {problems_file} not found. Run 'identify' step first.")
            return 1
    
    # Step 3: Extract frames
    if 'extract' in args.steps:
        extractor = ScalableFrameExtractor(config, video_index)
        extraction_results = extractor.extract_frames(problems_file)
    
    # Step 4: Pre-annotate frames (if enabled)
    if 'annotate' in args.steps and config.get('pre_annotation', {}).get('enabled', True):
        try:
            from pre_annotate_frames import FramePreAnnotator
            
            model_path = config['pre_annotation']['model_path']
            confidence = config['pre_annotation']['confidence']
            frames_dir = output_dir / 'frames'
            
            print("\n" + "="*80)
            print("PRE-ANNOTATING FRAMES")
            print("="*80)
            
            annotator = FramePreAnnotator(model_path, confidence)
            annotator.pre_annotate_frames(frames_dir)
            
        except ImportError as e:
            print(f"\n⚠ Could not run pre-annotation: {e}")
            print("  Skipping pre-annotation step")
    
    # Final summary
    print("\n" + "="*80)
    print("PIPELINE COMPLETE!")
    print("="*80)
    print(f"\nResults saved to: {output_dir}")
    print("\nNext steps:")
    print("  1. Review extracted frames and pre-annotations")
    print("  2. Upload to annotation tool (Roboflow, CVAT, etc.)")
    print("  3. Correct/validate annotations")
    print("  4. Export and add to training dataset")
    print("  5. Retrain object detection model")
    print("="*80)
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
