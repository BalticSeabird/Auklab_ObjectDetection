#!/usr/bin/env python3
"""
extract_event_clips.py

Extract video clips for events from daily_events CSV files.
Creates short clips around arrival/departure events with text overlays showing event information.

Usage:
    # Extract all events (single line - copy and paste ready)
    python code/postprocess/extract_event_clips.py --events-csv daily_analysis/TRI3/20250630/csv/daily_events_20250630.csv --video-dir /mnt/BSP_NAS2_vol4/Video/Video2025/TRI3/ --output-dir clips/TRI3/20250630/ --station TRI3

    # Extract only arrivals with fish (single line - copy and paste ready)
    python code/postprocess/extract_event_clips.py --events-csv ../../../../../../mnt/BSP_NAS2_work/auklab_model/summarized_inference/2025/6080/BONDEN6/20250623/csv/daily_events_20250623.csv --video-dir ../../../../../../mnt/BSP_NAS2_vol4/Video/Video2025/BONDEN6/ --output-dir clips/BONDEN6/20250623/ --station BONDEN6 --event-types arrival --fish-only

    # Process multiple dates for a station (single line - copy and paste ready)
    python code/postprocess/extract_event_clips.py --daily-analysis-dir daily_analysis/TRI3/ --video-dir /mnt/BSP_NAS2_vol4/Video/Video2025/TRI3/ --output-dir clips/TRI3/ --station TRI3 --event-types arrival

Features:
- Extracts clips around events with configurable before/after padding
- Adds text overlay with event information (type, timestamp, fish presence)
- Filters by event type (arrival, departure, or both)
- Optional fish-only filtering for arrivals
- Handles video files organized by date in subdirectories
- Creates descriptive filenames: {station}_{date}_{time}_{type}_{fish_status}_evt{id}.mp4
- Batch processing of multiple dates

Output:
    clips/
        TRI3/
            20250630/
                TRI3_20250630_122345_arrival_with_fish_evt001.mp4
                TRI3_20250630_122530_arrival_no_fish_evt002.mp4
                TRI3_20250630_123015_departure_evt003.mp4
"""

import argparse
import os
import sys
import subprocess
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd
import re
from typing import Optional, List, Tuple
import glob

def extract_date_from_csv_path(csv_path: str) -> Optional[str]:
    """Extract date string (YYYYMMDD) from CSV path or filename."""
    basename = os.path.basename(csv_path)
    match = re.search(r'(\d{8})', basename)
    if match:
        return match.group(1)
    return None

def find_video_for_timestamp(video_dir: Path, timestamp: datetime, 
                             date_str: str, extensions=['.mkv', '.mp4', '.avi']) -> Optional[Path]:
    """
    Find the video file that contains the given timestamp.
    
    Searches in:
    1. video_dir/YYYY-MM-DD/
    2. video_dir/
    
    Args:
        video_dir: Root directory containing videos
        timestamp: Event timestamp to find
        date_str: Date string in YYYYMMDD format
        extensions: Video file extensions to search for
    
    Returns:
        Path to video file or None if not found
    """
    # Format date for directory search (YYYY-MM-DD)
    date_obj = datetime.strptime(date_str, '%Y%m%d')
    date_formatted = date_obj.strftime('%Y-%m-%d')
    
    # Search in date subdirectory first
    date_subdir = video_dir / date_formatted
    if date_subdir.exists():
        search_dir = date_subdir
    else:
        search_dir = video_dir
    
    # Find all video files in the directory
    video_files = []
    for ext in extensions:
        video_files.extend(list(search_dir.glob(f"*{ext}")))
    
    if not video_files:
        print(f"  Warning: No video files found in {search_dir}")
        return None
    
    # Extract timestamps from video filenames
    # Expected format: STATION_YYYYMMDDTHHMMSS_*.ext
    for video_path in video_files:
        match = re.search(r'(\d{8})T(\d{6})', video_path.name)
        if match:
            video_date = match.group(1)
            video_time = match.group(2)
            
            # Parse video start timestamp
            video_start = datetime.strptime(f"{video_date}T{video_time}", '%Y%m%dT%H%M%S')
            
            # Get video duration using ffprobe
            try:
                duration = get_video_duration(video_path)
                video_end = video_start + timedelta(seconds=duration)
                
                # Check if timestamp falls within video range
                if video_start <= timestamp < video_end:
                    return video_path
            except Exception as e:
                print(f"  Warning: Could not get duration for {video_path.name}: {e}")
                continue
    
    return None

def get_video_duration(video_path: Path) -> float:
    """Get video duration in seconds using ffprobe."""
    cmd = [
        'ffprobe',
        '-v', 'error',
        '-show_entries', 'format=duration',
        '-of', 'default=noprint_wrappers=1:nokey=1',
        str(video_path)
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return float(result.stdout.strip())

def format_seconds_to_time(seconds: float) -> str:
    """Convert seconds to HH:MM:SS.mmm format for ffmpeg."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    return f"{hours:02d}:{minutes:02d}:{secs:06.3f}"

def extract_clip_with_overlay(video_path: Path, start_time: float, end_time: float,
                              output_path: Path, overlay_text: str) -> bool:
    """
    Extract video clip with text overlay using ffmpeg.
    
    Args:
        video_path: Input video file
        start_time: Start time in seconds
        end_time: End time in seconds
        output_path: Output clip path
        overlay_text: Text to overlay on video
    
    Returns:
        True if successful, False otherwise
    """
    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Format times for ffmpeg
    start_str = format_seconds_to_time(start_time)
    duration = end_time - start_time
    
    # Escape text for ffmpeg drawtext filter
    # Replace single quotes with '\'' and escape special chars
    text_escaped = overlay_text.replace("'", "'\\''").replace(":", "\\:")
    
    # Build ffmpeg command with text overlay
    # Using drawtext filter to add text at top of video
    cmd = [
        'ffmpeg',
        '-y',  # Overwrite output
        '-ss', start_str,  # Start time
        '-i', str(video_path),  # Input video
        '-t', str(duration),  # Duration
        '-vf', f"drawtext=text='{text_escaped}':fontcolor=white:fontsize=32:box=1:boxcolor=black@0.7:boxborderw=10:x=10:y=10",
        '-c:v', 'libx264',  # Video codec
        '-preset', 'fast',  # Encoding speed
        '-crf', '23',  # Quality (lower = better)
        '-c:a', 'aac',  # Audio codec
        '-b:a', '128k',  # Audio bitrate
        str(output_path)
    ]
    
    try:
        # Run ffmpeg with suppressed output
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"  Error: ffmpeg failed for {output_path.name}")
        if e.stderr:
            print(f"  {e.stderr[:200]}")  # Show first 200 chars of error
        return False

def format_event_filename(station: str, date_str: str, timestamp: datetime,
                         event_type: str, event_id: int, 
                         arrival_with_fish: Optional[bool] = None) -> str:
    """
    Create descriptive filename for event clip.
    
    Format: {station}_{date}_{time}_{type}_{fish_status}_evt{id}.mp4
    Example: TRI3_20250630_122345_arrival_with_fish_evt001.mp4
    """
    time_str = timestamp.strftime('%H%M%S')
    
    # Add fish status for arrivals
    if event_type == 'arrival' and arrival_with_fish is not None:
        fish_status = 'with_fish' if arrival_with_fish else 'no_fish'
        return f"{station}_{date_str}_{time_str}_{event_type}_{fish_status}_evt{event_id:03d}.mp4"
    else:
        return f"{station}_{date_str}_{time_str}_{event_type}_evt{event_id:03d}.mp4"

def format_overlay_text(event: pd.Series, event_id: int) -> str:
    """Create overlay text for video clip."""
    event_type = event['type'].upper()
    timestamp = event['timestamp'].strftime('%Y-%m-%d %H:%M:%S')
    
    text_parts = [
        f"Event #{event_id}: {event_type}",
        f"Time: {timestamp}"
    ]
    
    # Add fish information for arrivals
    if event['type'] == 'arrival':
        if event.get('arrival_with_fish', False):
            fish_count = event.get('fish_count', 0)
            text_parts.append(f"Fish: YES (count={fish_count})")
        else:
            text_parts.append("Fish: NO")
    
    return " | ".join(text_parts)

def load_events_csv(csv_path: Path, event_types: Optional[List[str]] = None,
                   fish_only: bool = False) -> pd.DataFrame:
    """
    Load and filter events CSV.
    
    Args:
        csv_path: Path to daily_events CSV
        event_types: List of event types to include ('arrival', 'departure')
        fish_only: If True, only include arrivals with fish
    
    Returns:
        Filtered DataFrame
    """
    df = pd.read_csv(csv_path)
    
    # Convert timestamp column to datetime
    # Handle both 'timestamp' and 'absolute_timestamp' column names
    if 'absolute_timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['absolute_timestamp'])
    elif 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
    else:
        print(f"Warning: No 'timestamp' or 'absolute_timestamp' column in {csv_path}")
        return pd.DataFrame()
    
    # Filter by event type
    if event_types:
        df = df[df['type'].isin(event_types)]
    
    # Filter by fish presence (only for arrivals)
    if fish_only:
        df = df[
            (df['type'] == 'arrival') & 
            (df.get('arrival_with_fish', False) == True)
        ]
    
    return df

def extract_events_from_csv(csv_path: Path, video_dir: Path, output_dir: Path,
                            station: str, event_types: Optional[List[str]] = None,
                            fish_only: bool = False, clip_before: int = 5,
                            clip_after: int = 10, video_extensions: List[str] = None) -> dict:
    """
    Extract clips for all events in a CSV file.
    
    Returns:
        Dictionary with statistics
    """
    if video_extensions is None:
        video_extensions = ['.mkv', '.mp4', '.avi']
    
    # Extract date from CSV path
    date_str = extract_date_from_csv_path(str(csv_path))
    if not date_str:
        print(f"Error: Could not extract date from {csv_path}")
        return {'total': 0, 'extracted': 0, 'failed': 0, 'skipped': 0}
    
    print(f"\nProcessing: {csv_path.name}")
    print(f"Date: {date_str}, Station: {station}")
    
    # Load and filter events
    events_df = load_events_csv(csv_path, event_types, fish_only)
    
    if len(events_df) == 0:
        print("  No events match the filter criteria")
        return {'total': 0, 'extracted': 0, 'failed': 0, 'skipped': 0}
    
    print(f"  Found {len(events_df)} events to extract")
    
    # Create output directory for this date
    date_output_dir = output_dir / date_str
    date_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Statistics
    stats = {'total': len(events_df), 'extracted': 0, 'failed': 0, 'skipped': 0}
    
    # Process each event
    for idx, (_, event) in enumerate(events_df.iterrows(), start=1):
        event_id = idx
        timestamp = event['timestamp']
        event_type = event['type']
        
        print(f"  [{idx}/{len(events_df)}] {event_type.capitalize()} at {timestamp.strftime('%H:%M:%S')}", end='')
        
        # Find video file for this timestamp
        video_path = find_video_for_timestamp(video_dir, timestamp, date_str, video_extensions)
        
        if video_path is None:
            print(" - Video not found, skipping")
            stats['skipped'] += 1
            continue
        
        print(f" - Video: {video_path.name}", end='')
        
        # Calculate offset within video
        match = re.search(r'(\d{8})T(\d{6})', video_path.name)
        video_start = datetime.strptime(f"{match.group(1)}T{match.group(2)}", '%Y%m%dT%H%M%S')
        offset_seconds = (timestamp - video_start).total_seconds()
        
        # Calculate clip start/end times
        clip_start = max(0, offset_seconds - clip_before)
        clip_end = offset_seconds + clip_after
        
        # Generate output filename
        arrival_with_fish = event.get('arrival_with_fish', None) if event_type == 'arrival' else None
        output_filename = format_event_filename(
            station, date_str, timestamp, event_type, event_id, arrival_with_fish
        )
        output_path = date_output_dir / output_filename
        
        # Skip if already exists
        if output_path.exists():
            print(" - Already exists, skipping")
            stats['skipped'] += 1
            continue
        
        # Create overlay text
        overlay_text = format_overlay_text(event, event_id)
        
        # Extract clip
        success = extract_clip_with_overlay(
            video_path, clip_start, clip_end, output_path, overlay_text
        )
        
        if success:
            print(f" - Created: {output_filename}")
            stats['extracted'] += 1
        else:
            print(f" - Failed")
            stats['failed'] += 1
    
    return stats

def process_daily_analysis_directory(daily_analysis_dir: Path, video_dir: Path,
                                     output_dir: Path, station: str,
                                     event_types: Optional[List[str]] = None,
                                     fish_only: bool = False, clip_before: int = 5,
                                     clip_after: int = 10) -> dict:
    """
    Process all dates in a daily_analysis directory structure.
    
    Expects structure: daily_analysis/{station}/{date}/csv/daily_events_{date}.csv
    """
    print(f"Searching for event CSVs in: {daily_analysis_dir}")
    
    # Find all daily_events CSV files
    csv_pattern = daily_analysis_dir / "*/csv/daily_events_*.csv"
    csv_files = sorted(glob.glob(str(csv_pattern)))
    
    if not csv_files:
        print(f"No event CSV files found matching pattern: {csv_pattern}")
        return {'total_dates': 0, 'total_events': 0, 'extracted': 0, 'failed': 0, 'skipped': 0}
    
    print(f"Found {len(csv_files)} date(s) to process\n")
    print("="*80)
    
    # Aggregate statistics
    total_stats = {'total_dates': len(csv_files), 'total_events': 0, 
                  'extracted': 0, 'failed': 0, 'skipped': 0}
    
    # Process each CSV file
    for csv_path in csv_files:
        stats = extract_events_from_csv(
            Path(csv_path), video_dir, output_dir, station,
            event_types, fish_only, clip_before, clip_after
        )
        
        total_stats['total_events'] += stats['total']
        total_stats['extracted'] += stats['extracted']
        total_stats['failed'] += stats['failed']
        total_stats['skipped'] += stats['skipped']
    
    return total_stats

def main():
    parser = argparse.ArgumentParser(
        description="Extract video clips for events from daily_events CSV files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Extract all events for one date
  python extract_event_clips.py \\
      --events-csv daily_analysis/TRI3/20250630/csv/daily_events_20250630.csv \\
      --video-dir /mnt/BSP_NAS2_vol4/Video/Video2025/TRI3/ \\
      --output-dir clips/TRI3/

  # Extract only arrivals with fish for one date
  python extract_event_clips.py \\
      --events-csv daily_analysis/TRI3/20250630/csv/daily_events_20250630.csv \\
      --video-dir /mnt/BSP_NAS2_vol4/Video/Video2025/TRI3/ \\
      --output-dir clips/TRI3/ \\
      --event-types arrival \\
      --fish-only

  # Process entire station (all dates)
  python extract_event_clips.py \\
      --daily-analysis-dir daily_analysis/TRI3/ \\
      --video-dir /mnt/BSP_NAS2_vol4/Video/Video2025/TRI3/ \\
      --output-dir clips/TRI3/ \\
      --station TRI3 \\
      --event-types arrival
        """
    )
    
    # Input options (mutually exclusive)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--events-csv', type=str,
                            help='Path to single daily_events CSV file')
    input_group.add_argument('--daily-analysis-dir', type=str,
                            help='Path to daily_analysis directory to process all dates')
    
    # Required arguments
    parser.add_argument('--video-dir', type=str, required=True,
                       help='Directory containing original video files (searches recursively)')
    
    parser.add_argument('--output-dir', type=str, required=True,
                       help='Output directory for extracted clips')
    
    parser.add_argument('--station', type=str, required=True,
                       help='Station name (e.g., TRI3, FAR3)')
    
    # Filtering options
    parser.add_argument('--event-types', nargs='+', choices=['arrival', 'departure'],
                       help='Event types to extract (default: both)')
    
    parser.add_argument('--fish-only', action='store_true',
                       help='Only extract arrivals with fish detections')
    
    # Clip parameters
    parser.add_argument('--clip-before', type=int, default=5,
                       help='Seconds to include before event (default: 5)')
    
    parser.add_argument('--clip-after', type=int, default=10,
                       help='Seconds to include after event (default: 10)')
    
    parser.add_argument('--video-extensions', nargs='+', default=['.mkv', '.mp4', '.avi'],
                       help='Video file extensions to search for')
    
    args = parser.parse_args()
    
    # Convert paths
    video_dir = Path(args.video_dir)
    output_dir = Path(args.output_dir)
    
    if not video_dir.exists():
        print(f"Error: Video directory not found: {video_dir}")
        sys.exit(1)
    
    # Check for required tools
    try:
        subprocess.run(['ffmpeg', '-version'], capture_output=True, check=True)
        subprocess.run(['ffprobe', '-version'], capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("Error: ffmpeg and ffprobe are required. Please install ffmpeg.")
        sys.exit(1)
    
    print("="*80)
    print("EVENT CLIP EXTRACTION")
    print("="*80)
    print(f"Station: {args.station}")
    print(f"Video directory: {video_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Event types: {args.event_types if args.event_types else 'all'}")
    print(f"Fish only: {args.fish_only}")
    print(f"Clip padding: {args.clip_before}s before, {args.clip_after}s after")
    print("="*80)
    
    # Process events
    if args.events_csv:
        # Single CSV file
        csv_path = Path(args.events_csv)
        if not csv_path.exists():
            print(f"Error: Events CSV not found: {csv_path}")
            sys.exit(1)
        
        stats = extract_events_from_csv(
            csv_path, video_dir, output_dir, args.station,
            args.event_types, args.fish_only, args.clip_before, 
            args.clip_after, args.video_extensions
        )
        
        # Print summary
        print("\n" + "="*80)
        print("EXTRACTION COMPLETE")
        print("="*80)
        print(f"Total events: {stats['total']}")
        print(f"Extracted: {stats['extracted']}")
        print(f"Failed: {stats['failed']}")
        print(f"Skipped: {stats['skipped']}")
        
    else:
        # Process entire directory
        daily_analysis_dir = Path(args.daily_analysis_dir)
        if not daily_analysis_dir.exists():
            print(f"Error: Daily analysis directory not found: {daily_analysis_dir}")
            sys.exit(1)
        
        stats = process_daily_analysis_directory(
            daily_analysis_dir, video_dir, output_dir, args.station,
            args.event_types, args.fish_only, args.clip_before, args.clip_after
        )
        
        # Print summary
        print("\n" + "="*80)
        print("BATCH EXTRACTION COMPLETE")
        print("="*80)
        print(f"Dates processed: {stats['total_dates']}")
        print(f"Total events: {stats['total_events']}")
        print(f"Extracted: {stats['extracted']}")
        print(f"Failed: {stats['failed']}")
        print(f"Skipped: {stats['skipped']}")
    
    print("="*80)
    print(f"\nClips saved to: {output_dir}")

if __name__ == '__main__':
    main()
