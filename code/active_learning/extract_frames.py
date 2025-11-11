"""
Frame Extraction for Active Learning

Extracts specific frames from videos based on the problematic frame analysis.
These frames can then be annotated and used to retrain the object detection model.
"""

import sys
import os
import json
import cv2
from pathlib import Path
from datetime import datetime, timedelta
import numpy as np
from collections import defaultdict


class FrameExtractor:
    """
    Extracts frames from videos for annotation and retraining.
    """
    
    def __init__(self, video_dir='video', output_dir='data/frames_for_annotation'):
        self.video_dir = Path(video_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def extract_frames_from_problems(self, problems_file='data/problematic_frames.json',
                                    max_frames_per_type=100, priority='diverse'):
        """
        Extract frames based on problem analysis.
        
        Args:
            problems_file: JSON file with problematic frame analysis
            max_frames_per_type: Maximum frames to extract per problem type
            priority: 'diverse' (spread across videos) or 'concentrated' (worst videos first)
        """
        
        # Load problems
        with open(problems_file, 'r') as f:
            all_problems = json.load(f)
        
        print("="*80)
        print("FRAME EXTRACTION FOR ACTIVE LEARNING")
        print("="*80)
        print(f"\nStrategy: Extract up to {max_frames_per_type} frames per problem type")
        print(f"Priority: {priority}")
        
        # Organize frames by problem type
        frames_by_type = self._organize_frames_by_type(all_problems)
        
        # Select frames to extract based on priority
        selected_frames = self._select_frames_to_extract(
            frames_by_type, max_frames_per_type, priority
        )
        
        # Extract the frames
        extraction_results = self._extract_selected_frames(selected_frames)
        
        # Generate annotation batch manifest
        self._create_annotation_manifest(extraction_results)
        
        return extraction_results
    
    def _organize_frames_by_type(self, all_problems):
        """Organize all problematic frames by type"""
        
        frames_by_type = {
            'edge_detection': [],
            'spike': [],
            'dip': [],
            'high_count': [],
            'count_transition': []
        }
        
        for video_problems in all_problems:
            video_file = video_problems['video_file']
            start_time = datetime.fromisoformat(video_problems['start_time'])
            
            # Edge detections
            for frame_info in video_problems['edge_detections']:
                frames_by_type['edge_detection'].append({
                    'video_file': video_file,
                    'second': frame_info['second'],
                    'timestamp': (start_time + timedelta(seconds=frame_info['second'])).isoformat(),
                    'details': frame_info
                })
            
            # Spikes
            for frame_info in video_problems['spike_frames']:
                frames_by_type['spike'].append({
                    'video_file': video_file,
                    'second': frame_info['second'],
                    'timestamp': (start_time + timedelta(seconds=frame_info['second'])).isoformat(),
                    'details': frame_info
                })
            
            # Dips
            for frame_info in video_problems['dip_frames']:
                frames_by_type['dip'].append({
                    'video_file': video_file,
                    'second': frame_info['second'],
                    'timestamp': (start_time + timedelta(seconds=frame_info['second'])).isoformat(),
                    'details': frame_info
                })
            
            # High count
            for frame_info in video_problems['high_count_frames']:
                frames_by_type['high_count'].append({
                    'video_file': video_file,
                    'second': frame_info['second'],
                    'timestamp': (start_time + timedelta(seconds=frame_info['second'])).isoformat(),
                    'details': frame_info
                })
            
            # Transitions
            for frame_info in video_problems['count_transition_frames']:
                frames_by_type['count_transition'].append({
                    'video_file': video_file,
                    'second': frame_info['second'],
                    'timestamp': (start_time + timedelta(seconds=frame_info['second'])).isoformat(),
                    'details': frame_info
                })
        
        print("\n" + "="*80)
        print("AVAILABLE FRAMES BY TYPE:")
        print("="*80)
        for frame_type, frames in frames_by_type.items():
            print(f"{frame_type:20s}: {len(frames):5d} frames")
        
        return frames_by_type
    
    def _select_frames_to_extract(self, frames_by_type, max_per_type, priority='diverse'):
        """Select which frames to extract based on priority strategy"""
        
        selected = {}
        
        for frame_type, frames in frames_by_type.items():
            if len(frames) == 0:
                selected[frame_type] = []
                continue
            
            if priority == 'diverse':
                # Spread selection across different videos
                selected[frame_type] = self._diverse_selection(frames, max_per_type)
            else:  # concentrated
                # Focus on videos with most problems
                selected[frame_type] = frames[:max_per_type]
            
        print("\n" + "="*80)
        print("SELECTED FRAMES FOR EXTRACTION:")
        print("="*80)
        for frame_type, frames in selected.items():
            print(f"{frame_type:20s}: {len(frames):5d} frames")
        
        return selected
    
    def _diverse_selection(self, frames, max_frames):
        """Select frames diversely across different videos"""
        
        # Group by video
        by_video = defaultdict(list)
        for frame in frames:
            by_video[frame['video_file']].append(frame)
        
        # Sample evenly from each video
        selected = []
        videos = list(by_video.keys())
        
        frames_per_video = max(1, max_frames // len(videos))
        
        for video in videos:
            video_frames = by_video[video]
            # Sample evenly from this video
            if len(video_frames) <= frames_per_video:
                selected.extend(video_frames)
            else:
                # Take evenly spaced samples
                indices = np.linspace(0, len(video_frames)-1, frames_per_video, dtype=int)
                selected.extend([video_frames[i] for i in indices])
        
        # If we haven't reached max_frames, add more
        if len(selected) < max_frames:
            remaining = [f for f in frames if f not in selected]
            selected.extend(remaining[:max_frames - len(selected)])
        
        return selected[:max_frames]
    
    def _extract_selected_frames(self, selected_frames):
        """Extract the selected frames from videos"""
        
        print("\n" + "="*80)
        print("EXTRACTING FRAMES...")
        print("="*80)
        
        extraction_results = []
        
        # Group by video file for efficient extraction
        frames_by_video = defaultdict(list)
        for frame_type, frames in selected_frames.items():
            for frame_info in frames:
                frame_info['frame_type'] = frame_type
                frames_by_video[frame_info['video_file']].append(frame_info)
        
        total_videos = len(frames_by_video)
        
        for i, (video_file, frames) in enumerate(sorted(frames_by_video.items()), 1):
            print(f"\n[{i}/{total_videos}] Processing {video_file}...")
            
            # Find video file
            video_path = self._find_video_file(video_file)
            if video_path is None:
                print(f"  ✗ Video file not found: {video_file}")
                continue
            
            # Extract frames from this video
            extracted = self._extract_frames_from_video(video_path, frames)
            extraction_results.extend(extracted)
            
            print(f"  ✓ Extracted {len(extracted)} frames")
        
        print("\n" + "="*80)
        print(f"✅ Total frames extracted: {len(extraction_results)}")
        print(f"✅ Saved to: {self.output_dir}")
        print("="*80)
        
        return extraction_results
    
    def _find_video_file(self, video_basename):
        """Find video file (could be .mp4, .avi, .mkv, etc.)"""
        
        # Try common video extensions
        for ext in ['.mkv', '.mp4', '.avi', '.mov', '.MP4', '.AVI', '.MKV']:
            video_path = self.video_dir / f"{video_basename}{ext}"
            if video_path.exists():
                return video_path
        
        return None
    
    def _extract_frames_from_video(self, video_path, frames_to_extract):
        """Extract specific frames from a video"""
        
        extracted = []
        
        # Sort frames by second for efficient sequential access
        frames_to_extract = sorted(frames_to_extract, key=lambda x: x['second'])
        
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            print(f"  ✗ Could not open video: {video_path}")
            return extracted
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        for frame_info in frames_to_extract:
            second = frame_info['second']
            frame_number = int(second * fps)
            
            # Seek to frame
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            ret, frame = cap.read()
            
            if not ret:
                continue
            
            # Create output filename
            frame_type = frame_info['frame_type']
            video_basename = video_path.stem
            
            output_filename = f"{video_basename}_s{second:04d}_{frame_type}.jpg"
            output_path = self.output_dir / frame_type / output_filename
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Save frame
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
    
    def _create_annotation_manifest(self, extraction_results):
        """Create a manifest file for the annotation batch"""
        
        manifest = {
            'created': datetime.now().isoformat(),
            'total_frames': len(extraction_results),
            'frames_by_type': {},
            'frames': extraction_results
        }
        
        # Count by type
        for frame in extraction_results:
            frame_type = frame['frame_type']
            if frame_type not in manifest['frames_by_type']:
                manifest['frames_by_type'][frame_type] = 0
            manifest['frames_by_type'][frame_type] += 1
        
        # Save manifest
        manifest_path = self.output_dir / 'annotation_manifest.json'
        with open(manifest_path, 'w') as f:
            json.dump(manifest, f, indent=2)
        
        print(f"\n✅ Created annotation manifest: {manifest_path}")
        
        # Create a simple CSV for easy viewing
        csv_path = self.output_dir / 'frames_list.csv'
        with open(csv_path, 'w') as f:
            f.write("filename,video,second,frame_type,timestamp\n")
            for frame in extraction_results:
                f.write(f"{frame['output_path']},{frame['video_file']},{frame['second']},"
                       f"{frame['frame_type']},{frame['timestamp']}\n")
        
        print(f"✅ Created frame list CSV: {csv_path}")


def main():
    """
    Main workflow for extracting frames for active learning.
    """
    
    import argparse
    
    parser = argparse.ArgumentParser(description='Extract problematic frames for annotation')
    parser.add_argument('--problems', default='data/problematic_frames.json',
                       help='JSON file with problematic frame analysis')
    parser.add_argument('--video-dir', default='video',
                       help='Directory containing video files')
    parser.add_argument('--output-dir', default='data/frames_for_annotation',
                       help='Output directory for extracted frames')
    parser.add_argument('--max-per-type', type=int, default=100,
                       help='Maximum frames to extract per problem type')
    parser.add_argument('--priority', choices=['diverse', 'concentrated'], default='diverse',
                       help='Frame selection strategy')
    
    args = parser.parse_args()
    
    extractor = FrameExtractor(video_dir=args.video_dir, output_dir=args.output_dir)
    
    results = extractor.extract_frames_from_problems(
        problems_file=args.problems,
        max_frames_per_type=args.max_per_type,
        priority=args.priority
    )
    
    print("\n" + "="*80)
    print("NEXT STEPS:")
    print("="*80)
    print("\n1. Annotate the extracted frames using your annotation tool (Roboflow, CVAT, etc.)")
    print(f"   Frames are organized by problem type in: {args.output_dir}")
    print("\n2. Combine with your existing training dataset")
    print("\n3. Retrain your object detection model")
    print("\n4. Run inference again and compare results")
    print("\n5. Iterate: The event detection should improve with better base detections!")
    

if __name__ == '__main__':
    main()
