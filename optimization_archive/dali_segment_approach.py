#!/usr/bin/env python3
"""
DALI Segmented Processing Approach

Since DALI cannot seek to arbitrary positions in HEVC files, we need a different strategy:

1. Split the video into smaller segments using ffmpeg
2. Process each segment with DALI independently  
3. Combine results at the end

This approach leverages DALI's excellent decode performance (900+ img/s) while working 
around its seeking limitations for HEVC files.
"""

import subprocess
import os
import tempfile
from pathlib import Path
import time

class DALISegmentedProcessor:
    def __init__(self, segment_duration=60):
        """
        Args:
            segment_duration: Duration of each segment in seconds
        """
        self.segment_duration = segment_duration
        
    def split_video_into_segments(self, video_path, output_dir):
        """Split video into segments using ffmpeg"""
        print(f"🔪 Splitting video into {self.segment_duration}s segments...")
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Use ffmpeg to split video
        cmd = [
            'ffmpeg', '-i', str(video_path),
            '-c', 'copy',  # Copy without re-encoding
            '-f', 'segment',
            '-segment_time', str(self.segment_duration),
            '-segment_format', 'mkv',
            '-reset_timestamps', '1',
            f'{output_dir}/segment_%03d.mkv'
        ]
        
        print(f"   Command: {' '.join(cmd)}")
        
        start_time = time.time()
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"   ❌ ffmpeg failed: {result.stderr}")
            return []
        
        elapsed = time.time() - start_time
        print(f"   ✅ Video split completed in {elapsed:.1f}s")
        
        # Find created segments
        segments = sorted(Path(output_dir).glob("segment_*.mkv"))
        print(f"   Created {len(segments)} segments")
        
        return segments
    
    def process_segments_with_dali(self, segments, processor, output_csv):
        """Process each segment with DALI and combine results"""
        all_results = []
        total_frames_processed = 0
        
        print(f"\n🚀 Processing {len(segments)} segments with DALI...")
        
        for i, segment_path in enumerate(segments):
            print(f"\n📹 Processing segment {i+1}/{len(segments)}: {segment_path.name}")
            
            try:
                # Process this segment
                segment_results = processor.process_full_video(
                    str(segment_path), 
                    f"{output_csv}.segment_{i:03d}.csv"
                )
                
                # Adjust frame numbers to be continuous across segments
                segment_start_frame = total_frames_processed
                for result in segment_results:
                    result['frame'] += segment_start_frame
                
                all_results.extend(segment_results)
                total_frames_processed += len(segment_results)
                
                print(f"   ✅ Segment {i+1} completed: {len(segment_results)} detections")
                
            except Exception as e:
                print(f"   ❌ Segment {i+1} failed: {e}")
                print(f"   Continuing with next segment...")
                continue
        
        # Save combined results
        if all_results:
            self._save_combined_results(all_results, output_csv)
        
        print(f"\n🎉 Segmented processing complete!")
        print(f"   Total frames processed: {total_frames_processed}")
        print(f"   Total detections: {len(all_results)}")
        
        return all_results
    
    def _save_combined_results(self, results, output_csv):
        """Save combined results to CSV"""
        import pandas as pd
        
        if not results:
            print("   No results to save")
            return
        
        df = pd.DataFrame(results)
        df = df.sort_values('frame')  # Ensure frame order
        df.to_csv(output_csv, index=False)
        print(f"   Combined results saved to: {output_csv}")

def demo_segmented_approach():
    """Demonstrate the segmented processing approach"""
    video_path = "vid/input2.mkv"
    
    # Create temporary directory for segments
    with tempfile.TemporaryDirectory() as temp_dir:
        print(f"Using temporary directory: {temp_dir}")
        
        # Initialize segmented processor
        processor = DALISegmentedProcessor(segment_duration=30)  # 30 second segments
        
        # Split video
        segments = processor.split_video_into_segments(video_path, temp_dir)
        
        if not segments:
            print("No segments created")
            return
        
        # For demo, just show what would happen
        print(f"\n📊 Segmented Processing Plan:")
        print(f"   Original video: {video_path}")
        print(f"   Segments created: {len(segments)}")
        print(f"   Segment duration: {processor.segment_duration}s")
        print(f"   Each segment would be processed by DALI independently")
        print(f"   Results would be combined with continuous frame numbering")
        
        # Show segment details
        for i, segment in enumerate(segments):
            size_mb = segment.stat().st_size / (1024 * 1024)
            print(f"   Segment {i+1}: {segment.name} ({size_mb:.1f} MB)")

if __name__ == "__main__":
    demo_segmented_approach()