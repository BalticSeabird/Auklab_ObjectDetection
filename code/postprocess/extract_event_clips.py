#!/usr/bin/env python3
"""
extract_event_clips.py

Extract video clips for events from daily_events CSV files.
Creates short clips around arrival/departure events with text overlays and YOLO detection overlays.

Usage:
    # Process multiple stations for entire year
    python code/postprocess/extract_event_clips.py --stations BONDEN6 TRI3 FAR3 --model-version 6080 --year 2025 --model-path models/auklab_model_xlarge_combined_6080_v1.pt

    # Extract only arrivals with fish for multiple stations
    python code/postprocess/extract_event_clips.py --stations BONDEN6 FAR3 FAR6 ROST2 ROST6 TRI3 --model-version 6080 --year 2025 --event-types arrival --model-path models/auklab_model_xlarge_combined_6080_v1.pt --clip-before 4 --clip-after 30 --no-skip

    # Process single station (legacy mode)
    python code/postprocess/extract_event_clips.py --station TRI3 --model-version 6080 --year 2025 --model-path models/auklab_model_xlarge_combined_6080_v1.pt


Features:
- Extracts clips around events with configurable before/after padding
- Uses event_id from CSV for filenames
- Separates arrivals with fish and without fish into different folders
- Runs YOLO detection on clips and saves detections as CSV
- Adds bounding boxes to video clips
- Optional video compression
- Filters by event type (arrival, departure, or both)
- Handles video files organized by date in subdirectories
- Batch processing of multiple dates

Output Structure:
    {base_output}/event_data/
        TRI3/
            20250630/
                arrival_with_fish/
                    video/
                        6080_TRI3_20250630_122345_arr.mp4
                    csv/
                        6080_TRI3_20250630_122345_arr_detections.csv
                arrival_no_fish/
                    video/
                        6080_TRI3_20250630_122530_arr.mp4
                    csv/
                        6080_TRI3_20250630_122530_arr_detections.csv
                departure/
                    video/
                        6080_TRI3_20250630_123015_dep.mp4
                    csv/
                        6080_TRI3_20250630_123015_dep_detections.csv
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
import cv2
import numpy as np

try:
    from ultralytics import YOLO
    HAS_YOLO = True
except ImportError:
    HAS_YOLO = False
    print("Warning: ultralytics not installed. YOLO detection will be disabled.")
    print("Install with: pip install ultralytics")

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
    # Expected formats:
    # - Standard: STATION_YYYYMMDDTHHMMSS_*.ext
    # - XProtect: ##_STATION_(IP)_YYYY-MM-DD_HH.MM.SS_######.ext
    for video_path in video_files:
        # Try standard format first
        match = re.search(r'(\d{8})T(\d{6})', video_path.name)
        if match:
            video_date = match.group(1)
            video_time = match.group(2)
            video_start = datetime.strptime(f"{video_date}T{video_time}", '%Y%m%dT%H%M%S')
        else:
            # Try XProtect format
            match = re.search(r'(\d{4})-(\d{2})-(\d{2})_(\d{2})\.(\d{2})\.(\d{2})', video_path.name)
            if not match:
                continue
            year, month, day, hour, minute, second = match.groups()
            video_start = datetime(int(year), int(month), int(day), int(hour), int(minute), int(second))
        
        # Get video duration and check if timestamp falls within range
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

def run_yolo_on_clip(video_path: Path, model_path: Path, output_video_path: Path, 
                    detections_csv_path: Path, overlay_text: str) -> bool:
    """
    Run YOLO detection on video clip and add bounding boxes.
    
    Args:
        video_path: Input video clip path
        model_path: Path to YOLO model
        output_video_path: Output video with bounding boxes
        detections_csv_path: Output CSV file for detections
        overlay_text: Text to overlay on video
    
    Returns:
        True if successful, False otherwise
    """
    if not HAS_YOLO:
        print("  YOLO not available, skipping detection")
        return False
    
    try:
        # Load YOLO model
        model = YOLO(str(model_path))
        
        # Open video
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            print(f"  Error: Could not open video {video_path}")
            return False
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Setup video writer (uncompressed initially)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(output_video_path), fourcc, fps, (width, height))
        
        # Store detections
        all_detections = []
        frame_num = 0
        
        # Process each frame
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Run YOLO detection
            results = model(frame, verbose=False)
            
            # Draw bounding boxes
            annotated_frame = results[0].plot()
            
            # Add event info overlay
            text_escaped = overlay_text.replace(':', ' ')
            cv2.putText(annotated_frame, text_escaped, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.rectangle(annotated_frame, (5, 5), (width - 5, 45), (0, 0, 0), -1)
            cv2.putText(annotated_frame, text_escaped, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
            
            # Write frame
            out.write(annotated_frame)
            
            # Store detections
            for box in results[0].boxes:
                all_detections.append({
                    'frame': frame_num,
                    'class': results[0].names[int(box.cls)],
                    'confidence': float(box.conf.cpu()),
                    'xmin': float(box.xyxy[0][0].cpu()),
                    'ymin': float(box.xyxy[0][1].cpu()),
                    'xmax': float(box.xyxy[0][2].cpu()),
                    'ymax': float(box.xyxy[0][3].cpu())
                })
            
            frame_num += 1
        
        # Release resources
        cap.release()
        out.release()
        
        # Save detections to CSV
        if all_detections:
            df = pd.DataFrame(all_detections)
            df.to_csv(detections_csv_path, index=False)
        else:
            # Create empty CSV with headers
            pd.DataFrame(columns=['frame', 'class', 'confidence', 'xmin', 'ymin', 'xmax', 'ymax']).to_csv(detections_csv_path, index=False)
        
        return True
        
    except Exception as e:
        print(f"  Error running YOLO: {e}")
        return False

def compress_video_h264(input_path: Path, output_path: Path, crf=28, preset='fast') -> bool:
    """
    Compress video using H.264 codec after YOLO processing.
    
    Args:
        input_path: Input video file
        output_path: Output compressed video
        crf: Quality (18=high quality, 23=default, 28=smaller file)
        preset: Encoding speed (ultrafast, fast, medium, slow)
    
    Returns:
        True if successful, False otherwise
    """
    cmd = [
        'ffmpeg',
        '-y',  # Overwrite output
        '-i', str(input_path),
        '-c:v', 'libx264',  # H.264 codec
        '-preset', preset,
        '-crf', str(crf),
        '-c:a', 'aac',  # Audio codec
        '-b:a', '128k',  # Audio bitrate
        '-movflags', '+faststart',  # Enable fast start
        str(output_path)
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"  Compression error: {e.stderr[:100]}")
        return False

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

def format_event_filename(event_id_from_csv: Optional[str] = None) -> str:
    """
    Create filename from event_id in CSV.
    
    Format: {event_id}.mp4
    Example: 6080_TRI3_20250630_122345_arr.mp4
    """
    if event_id_from_csv:
        return f"{event_id_from_csv}.mp4"
    else:
        # Fallback if no event_id in CSV
        return "event_unknown.mp4"

def format_overlay_text(event: pd.Series) -> str:
    """Create overlay text for video clip."""
    event_type = event['type'].upper()
    timestamp = event['timestamp'].strftime('%Y-%m-%d %H:%M:%S')
    event_id = event.get('event_id', 'unknown')
    
    text_parts = [
        f"Event: {event_id}",
        f"Type: {event_type}",
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
                            station: str, date_str: str, event_types: Optional[List[str]] = None,
                            fish_only: bool = False, clip_before: int = 5,
                            clip_after: int = 10, video_extensions: List[str] = None,
                            model_path: Optional[Path] = None, compress: bool = False) -> dict:
    """
    Extract clips for all events in a CSV file.
    
    Returns:
        Dictionary with statistics
    """
    if video_extensions is None:
        video_extensions = ['.mkv', '.mp4', '.avi']
    
    print(f"\nProcessing: {csv_path.name}")
    print(f"Date: {date_str}, Station: {station}")
    
    # Load and filter events
    events_df = load_events_csv(csv_path, event_types, fish_only)
    
    if len(events_df) == 0:
        print("  No events match the filter criteria")
        return {'total': 0, 'extracted': 0, 'failed': 0, 'skipped': 0}
    
    print(f"  Found {len(events_df)} events to extract")
    
    # Create output directory structure: station/date/event_type/(video|csv)
    date_output_dir = output_dir / station / date_str
    date_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Statistics
    stats = {'total': len(events_df), 'extracted': 0, 'failed': 0, 'skipped': 0}
    
    # Process each event
    for idx, (_, event) in enumerate(events_df.iterrows(), start=1):
        timestamp = event['timestamp']
        event_type = event['type']
        event_id_csv = event.get('event_id', None)
        
        print(f"  [{idx}/{len(events_df)}] {event_type.capitalize()} at {timestamp.strftime('%H:%M:%S')}", end='')
        
        # Determine subfolder based on event type and fish presence
        if event_type == 'arrival':
            has_fish = event.get('arrival_with_fish', False)
            if has_fish:
                subfolder = 'arrival_with_fish'
            else:
                subfolder = 'arrival_no_fish'
        else:
            subfolder = 'departure'
        
        # Create subfolder with video and csv subdirectories
        event_subfolder_dir = date_output_dir / subfolder
        video_dir_out = event_subfolder_dir / 'video'
        csv_dir_out = event_subfolder_dir / 'csv'
        video_dir_out.mkdir(parents=True, exist_ok=True)
        csv_dir_out.mkdir(parents=True, exist_ok=True)
        
        # Find video file for this timestamp
        video_path = find_video_for_timestamp(video_dir, timestamp, date_str, video_extensions)
        
        if video_path is None:
            print(" - Video not found, skipping")
            stats['skipped'] += 1
            continue
        
        print(f" - Video: {video_path.name}", end='')
        
        # Calculate offset within video - handle both naming formats
        match = re.search(r'(\d{8})T(\d{6})', video_path.name)
        if match:
            video_start = datetime.strptime(f"{match.group(1)}T{match.group(2)}", '%Y%m%dT%H%M%S')
        else:
            # XProtect format
            match = re.search(r'(\d{4})-(\d{2})-(\d{2})_(\d{2})\.(\d{2})\.(\d{2})', video_path.name)
            year, month, day, hour, minute, second = match.groups()
            video_start = datetime(int(year), int(month), int(day), int(hour), int(minute), int(second))
        
        offset_seconds = (timestamp - video_start).total_seconds()
        
        # Calculate clip start/end times
        clip_start = max(0, offset_seconds - clip_before)
        clip_end = offset_seconds + clip_after
        
        # Generate output filename from event_id
        output_filename = format_event_filename(event_id_csv)
        output_path = video_dir_out / output_filename
        temp_clip_path = video_dir_out / f"temp_{output_filename}"
        detections_csv_path = csv_dir_out / output_filename.replace('.mp4', '_detections.csv')
        
        # Skip if already exists
        if output_path.exists() and detections_csv_path.exists():
            print(" - Already exists, skipping")
            stats['skipped'] += 1
            continue
        
        # Create overlay text
        overlay_text = format_overlay_text(event)
        
        # Extract temporary clip without YOLO
        success = extract_clip_with_overlay(
            video_path, clip_start, clip_end, temp_clip_path, overlay_text
        )
        
        if not success:
            print(f" - Failed to extract clip")
            stats['failed'] += 1
            continue
        
        # Run YOLO detection on clip if model provided
        if model_path and model_path.exists():
            print(" - Running YOLO", end='')
            
            # YOLO creates uncompressed video in temp location
            yolo_temp_output = video_dir_out / f"yolo_temp_{output_filename}"
            
            yolo_success = run_yolo_on_clip(
                temp_clip_path, model_path, yolo_temp_output, detections_csv_path, 
                overlay_text
            )
            
            # Remove temp clip
            if temp_clip_path.exists():
                temp_clip_path.unlink()
            
            if not yolo_success:
                print(f" - YOLO failed")
                stats['failed'] += 1
                continue
            
            # Always compress YOLO output
            print(" - Compressing", end='')
            compress_success = compress_video_h264(yolo_temp_output, output_path, crf=28, preset='fast')
            
            # Remove uncompressed YOLO output
            if yolo_temp_output.exists():
                yolo_temp_output.unlink()
            
            if not compress_success:
                print(f" - Compression failed")
                stats['failed'] += 1
                continue
            
            print(f" - Created: {subfolder}/video/{output_filename}")
            stats['extracted'] += 1
        else:
            # No YOLO, just rename temp clip to final
            temp_clip_path.rename(output_path)
            # Create empty detections CSV
            pd.DataFrame(columns=['frame', 'class', 'confidence', 'xmin', 'ymin', 'xmax', 'ymax']).to_csv(detections_csv_path, index=False)
            print(f" - Created: {subfolder}/video/{output_filename} (no YOLO)")
            stats['extracted'] += 1
    
    return stats

def process_daily_analysis_directory(daily_analysis_dir: Path, video_dir: Path,
                                     output_dir: Path, station: str,
                                     event_types: Optional[List[str]] = None,
                                     fish_only: bool = False, clip_before: int = 5,
                                     clip_after: int = 10, model_path: Optional[Path] = None,
                                     compress: bool = False) -> dict:
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
        # Extract date from path
        date_str = extract_date_from_csv_path(str(csv_path))
        if not date_str:
            continue
            
        stats = extract_events_from_csv(
            Path(csv_path), video_dir, output_dir, station, date_str,
            event_types, fish_only, clip_before, clip_after,
            model_path=model_path, compress=compress
        )
        
        total_stats['total_events'] += stats['total']
        total_stats['extracted'] += stats['extracted']
        total_stats['failed'] += stats['failed']
        total_stats['skipped'] += stats['skipped']
    
    return total_stats

def is_date_already_processed(output_dir: Path, station: str, date_str: str, event_types: List[str], fish_only: bool) -> bool:
    """Check if a date has already been processed (has any output clips)"""
    date_output_dir = output_dir / station / date_str
    if not date_output_dir.exists():
        return False
    
    # Check for any video files in the expected subfolders
    subfolders = []
    if not event_types or 'arrival' in event_types:
        if fish_only:
            subfolders.append('arrival_with_fish')
        else:
            subfolders.extend(['arrival_with_fish', 'arrival_no_fish'])
    if not event_types or 'departure' in event_types:
        if not fish_only:  # departures don't have fish filtering
            subfolders.append('departure')
    
    for subfolder in subfolders:
        video_dir = date_output_dir / subfolder / 'video'
        if video_dir.exists() and list(video_dir.glob('*.mp4')):
            return True
    
    return False

def process_date_range(station: str, model_version: str, start_date: str, end_date: str,
                       base_inference_path: Path, base_video_path: Path, base_output_path: Path,
                       event_types: Optional[List[str]] = None,
                       fish_only: bool = False, clip_before: int = 5,
                       clip_after: int = 10, model_path: Optional[Path] = None,
                       skip_processed: bool = True) -> dict:
    """
    Process date range for a station.
    
    Constructs paths:
    - Events CSV: {base_inference_path}/summarized_inference/2025/{model_version}/{station}/{date}/csv/daily_events_{date}.csv
    - Videos: {base_video_path}/Video/Video2025/{station}/
    - Output: {base_output_path}/event_data/{station}/{date}/{event_type}/(video|csv)/
    """
    # Parse dates
    start = datetime.strptime(start_date, '%Y%m%d')
    end = datetime.strptime(end_date, '%Y%m%d')
    
    # Generate date list
    dates = []
    current = start
    while current <= end:
        dates.append(current.strftime('%Y%m%d'))
        current += timedelta(days=1)
    
    print(f"Processing {len(dates)} date(s) for station {station}")
    print(f"Date range: {start_date} to {end_date}")
    print("="*80)
    
    # Aggregate statistics
    total_stats = {'total_dates': 0, 'total_events': 0, 
                  'extracted': 0, 'failed': 0, 'skipped': 0}
    
    # Construct base paths
    inference_base = base_inference_path / "summarized_inference" / "2025" / model_version / station
    video_dir = base_video_path / "Video" / "Video2025" / station
    output_dir = base_output_path / "event_data"
    
    # Process each date
    for date_str in dates:
        # Construct CSV path
        csv_path = inference_base / date_str / "csv" / f"daily_events_{date_str}.csv"
        
        if not csv_path.exists():
            # Silently skip - no events CSV for this date
            continue
        
        # Check if already processed
        if skip_processed and is_date_already_processed(output_dir, station, date_str, event_types or ['arrival', 'departure'], fish_only):
            print(f"Skipping {date_str}: Already processed")
            continue
        
        total_stats['total_dates'] += 1
        
        stats = extract_events_from_csv(
            csv_path, video_dir, output_dir, station, date_str,
            event_types, fish_only, clip_before, clip_after,
            model_path=model_path, compress=True
        )
        
        total_stats['total_events'] += stats['total']
        total_stats['extracted'] += stats['extracted']
        total_stats['failed'] += stats['failed']
        total_stats['skipped'] += stats['skipped']
    
    return total_stats

def process_multiple_stations(stations: List[str], model_version: str, start_date: str, end_date: str,
                              base_inference_path: Path, base_video_path: Path, base_output_path: Path,
                              event_types: Optional[List[str]], fish_only: bool,
                              clip_before: int, clip_after: int, model_path: Optional[Path],
                              skip_processed: bool) -> dict:
    """Process multiple stations"""
    total_stats = {
        'stations_processed': 0,
        'total_dates': 0,
        'total_events': 0,
        'extracted': 0,
        'failed': 0,
        'skipped': 0
    }
    
    for idx, station in enumerate(stations, 1):
        print(f"\n[{idx}/{len(stations)}] Processing station: {station}")
        print("-" * 80)
        
        # Construct paths
        inference_base = base_inference_path / "summarized_inference" / "2025" / model_version / station
        video_dir = base_video_path / "Video" / "Video2025" / station
        
        print(f"Events CSV: {inference_base}")
        print(f"Videos: {video_dir}")
        print(f"Output: {base_output_path / 'event_data' / station}")
        
        if not video_dir.exists():
            print(f"WARNING: Video directory not found, skipping station {station}")
            continue
        
        # Process station
        stats = process_date_range(
            station, model_version, start_date, end_date,
            base_inference_path, base_video_path, base_output_path,
            event_types, fish_only, clip_before, clip_after,
            model_path, skip_processed
        )
        
        total_stats['stations_processed'] += 1
        total_stats['total_dates'] += stats['total_dates']
        total_stats['total_events'] += stats['total_events']
        total_stats['extracted'] += stats['extracted']
        total_stats['failed'] += stats['failed']
        total_stats['skipped'] += stats['skipped']
        
        print(f"Station {station} complete: {stats['total_dates']} dates, {stats['extracted']} clips extracted")
    
    return total_stats

def main():
    parser = argparse.ArgumentParser(
        description="Extract video clips for events with automatic path construction",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process multiple stations for entire year
  python extract_event_clips.py \\
      --stations BONDEN6 TRI3 FAR3 \\
      --model-version 6080 \\
      --year 2025 \\
      --model-path models/auklab_model_xlarge_combined_6080_v1.pt

  # Extract only arrivals with fish for multiple stations
  python extract_event_clips.py \\
      --stations BONDEN6 FAR3 \\
      --model-version 6080 \\
      --year 2025 \\
      --event-types arrival \\
      --fish-only

  # Process single station (legacy mode)
  python extract_event_clips.py \\
      --station TRI3 \\
      --model-version 6080 \\
      --year 2025
        """
    )
    
    # Station specification (mutually exclusive)
    station_group = parser.add_mutually_exclusive_group(required=True)
    station_group.add_argument('--stations', nargs='+',
                              help='Station names to process (e.g., BONDEN6 TRI3 FAR3)')
    station_group.add_argument('--station', type=str,
                              help='Single station name (legacy mode)')
    
    parser.add_argument('--model-version', type=str, required=True,
                       help='Model version (e.g., 6080, 4564)')
    
    # Date range options (mutually exclusive)
    date_group = parser.add_mutually_exclusive_group(required=True)
    date_group.add_argument('--year', type=int,
                           help='Process entire year (e.g., 2025)')
    date_group.add_argument('--start-date', type=str,
                           help='Start date in YYYYMMDD format (requires --end-date)')
    
    parser.add_argument('--end-date', type=str,
                       help='End date in YYYYMMDD format (use with --start-date)')
    
    # Base path arguments with defaults
    parser.add_argument('--base-inference-path', type=str,
                       default='/mnt/BSP_NAS2_work/auklab_model',
                       help='Base path for inference results (default: /mnt/BSP_NAS2_work/auklab_model)')
    
    parser.add_argument('--base-video-path', type=str,
                       default='/mnt/BSP_NAS2_vol4',
                       help='Base path for videos (default: /mnt/BSP_NAS2_vol4)')
    
    parser.add_argument('--base-output-path', type=str,
                       default='/mnt/BSP_NAS2_work/auklab_model',
                       help='Base path for output (default: /mnt/BSP_NAS2_work/auklab_model)')
    
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
    
    parser.add_argument('--model-path', type=str,
                       help='Path to YOLO model for detection on clips (e.g., models/auklab_model_xlarge_combined_6080_v1.pt)')
    
    parser.add_argument('--no-skip', action='store_true',
                       help='Reprocess already-completed dates (default: skip them)')
    
    args = parser.parse_args()
    
    # Validate date arguments
    if args.start_date and not args.end_date:
        print("Error: --end-date is required when using --start-date")
        sys.exit(1)
    if args.end_date and not args.start_date:
        print("Error: --start-date is required when using --end-date")
        sys.exit(1)
    
    # Determine date range
    if args.year:
        start_date = f"{args.year}0101"
        end_date = f"{args.year}1231"
    else:
        start_date = args.start_date
        end_date = args.end_date
    
    # Convert base paths
    base_inference_path = Path(args.base_inference_path)
    base_video_path = Path(args.base_video_path)
    base_output_path = Path(args.base_output_path)
    
    # Check for required tools
    try:
        subprocess.run(['ffmpeg', '-version'], capture_output=True, check=True)
        subprocess.run(['ffprobe', '-version'], capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("Error: ffmpeg and ffprobe are required. Please install ffmpeg.")
        sys.exit(1)
    
    # Check model path
    model_path = None
    if args.model_path:
        model_path = Path(args.model_path)
        if not model_path.exists():
            print(f"Warning: Model not found at {model_path}. YOLO detection will be skipped.")
            model_path = None
    
    print("="*80)
    print("EVENT CLIP EXTRACTION")
    print("="*80)
    
    if args.stations:
        # Multi-station mode
        print(f"Stations: {', '.join(args.stations)}")
        print(f"Model version: {args.model_version}")
        print(f"Date range: {start_date} to {end_date}")
        print(f"Event types: {args.event_types if args.event_types else 'all'}")
        print(f"Fish only: {args.fish_only}")
        print(f"Skip processed: {not args.no_skip}")
        print(f"Compression: Enabled (H.264 CRF28)")
        print("="*80)
        
        stats = process_multiple_stations(
            args.stations, args.model_version, start_date, end_date,
            base_inference_path, base_video_path, base_output_path,
            args.event_types, args.fish_only, args.clip_before, args.clip_after,
            model_path, skip_processed=not args.no_skip
        )
        
        print("\n" + "="*80)
        print("MULTI-STATION EXTRACTION COMPLETE")
        print("="*80)
        print(f"Stations processed: {stats['stations_processed']}")
        print(f"Dates processed: {stats['total_dates']}")
        print(f"Total events: {stats['total_events']}")
        print(f"Extracted: {stats['extracted']}")
        print(f"Failed: {stats['failed']}")
        print(f"Skipped: {stats['skipped']}")
        print("="*80)
    else:
        # Single station mode
        print(f"Station: {args.station}")
        print(f"Model version: {args.model_version}")
        print(f"Date range: {start_date} to {end_date}")
        print(f"Event types: {args.event_types if args.event_types else 'all'}")
        print(f"Fish only: {args.fish_only}")
        print(f"Clip padding: {args.clip_before}s before, {args.clip_after}s after")
        print(f"Skip processed: {not args.no_skip}")
        print(f"Compression: Enabled (H.264 CRF28)")
        print(f"Output: {base_output_path}/event_data/{args.station}/")
        print("="*80)
        
        stats = process_date_range(
            args.station, args.model_version, start_date, end_date,
            base_inference_path, base_video_path, base_output_path,
            args.event_types, args.fish_only, args.clip_before, args.clip_after,
            model_path, skip_processed=not args.no_skip
        )
        
        print("\n" + "="*80)
        print("EXTRACTION COMPLETE")
        print("="*80)
        print(f"Dates processed: {stats['total_dates']}")
        print(f"Total events: {stats['total_events']}")
        print(f"Extracted: {stats['extracted']}")
        print(f"Failed: {stats['failed']}")
        print(f"Skipped: {stats['skipped']}")
        print("="*80)
        print(f"\nClips saved to: {base_output_path}/event_data/{args.station}/")

if __name__ == '__main__':
    main()
