#!/usr/bin/env python3
"""
batch_analyze_days.py

Batch processing script to analyze seabird behavior across multiple days.
Groups CSV files by date, processes each day separately, and generates daily summary reports.

Usage:
    prod python3 code/postprocess/batch_analyze_days.py ../../../../../../mnt/BSP_NAS2_work/auklab_model/inference/2025/auklab_model_xlarge_combined_6080_v1/BONDEN6/ --station BONDEN6 --output_dir ../../../../../../mnt/BSP_NAS2_work/auklab_model/summarized_inference/2025/6080
    dev python3 code/postprocess/batch_analyze_days.py csv_detection_1fps/ --station TRI3 --output_dir dump/summarized_inference/
    
Features:
- Automatically groups files by date from filename timestamps
- Organizes output by station and date: daily_analysis/[station]/[date]/
- Processes all files for each day together
- Generates daily summary reports, plots, and aggregated CSV files
- Handles multiple observation periods per day
- Creates comparative analysis across days

Output Structure:
    daily_analysis/
        FAR3/
            20250630/
                daily_summary_20250630.txt
                csv/
                    daily_events_20250630.csv
                    daily_per_second_20250630.csv
                    ...
                plots/
                    daily_overview_20250630.png
        FAR1/
            20250701/
                ...
"""

import argparse
import os
import glob
import re
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# Import functions from event_detector
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from event_detector import (
    load_detections, 
    aggregate_per_second, 
    detect_arrivals_departures, 
    associate_fish_with_arrivals,
    compute_movement_metric, 
    detect_flapping,
    aggregate_per_minute,
    extract_timestamp_from_filename
)

def extract_date_from_filename(filename):
    """Extract date from filename pattern like FAR3_20250630T122002_raw.csv"""
    basename = os.path.basename(filename)
    match = re.search(r'(\d{8})T\d{6}', basename)
    if match:
        date_str = match.group(1)
        return datetime.strptime(date_str, '%Y%m%d').date()
    return None

def group_files_by_date(csv_files):
    """Group CSV files by their date"""
    date_groups = defaultdict(list)
    
    for filepath in csv_files:
        date = extract_date_from_filename(filepath)
        if date:
            date_groups[date].append(filepath)
        else:
            print(f"Warning: Could not extract date from {filepath}")
    
    return dict(date_groups)

def process_single_file(csv_path, fps=1, original_video_fps=25, conf_thresh=0.25, 
                       smooth_window_s=3, error_window_s=10, hold_seconds=8,
                       fish_window_s=5, movement_smoothing_s=5, 
                       flap_area_multiplier=3.0, flap_baseline_s=30):
    """Process a single CSV file and return all analysis results"""
    try:
        df, start_time = load_detections(csv_path, conf_thresh=conf_thresh)
        if len(df) == 0:
            print(f"Warning: No detections found in {csv_path}")
            return None
            
        print(f"Processing {os.path.basename(csv_path)}: {len(df)} detections")
        
        # Aggregate per second
        per_sec_df, positions, areas, counts = aggregate_per_second(
            df, fps, classes=('adult','chick','fish'), 
            original_video_fps=original_video_fps, start_time=start_time
        )
        
        # Detect events
        events, smoothed_counts = detect_arrivals_departures(
            per_sec_df, target_class='adult', smooth_window_s=smooth_window_s,
            error_window_s=error_window_s, hold_seconds=hold_seconds
        )
        
        # Associate fish with arrivals
        events = associate_fish_with_arrivals(
            events, per_sec_df, fish_window_s=fish_window_s, areas_by_class=areas
        )
        
        # Compute movement
        movement_df = compute_movement_metric(
            per_sec_df, classes=('adult',), smoothing_s=movement_smoothing_s, fps=fps
        )
        
        # Detect flapping
        flaps = detect_flapping(
            df, fps=fps, target_class='adult', area_multiplier=flap_area_multiplier,
            baseline_window_s=flap_baseline_s, original_video_fps=original_video_fps,
            start_time=start_time
        )
        
        # Per-minute aggregation
        per_min_df = aggregate_per_minute(per_sec_df, events, movement_df, flaps)
        
        return {
            'filename': os.path.basename(csv_path),
            'start_time': start_time,
            'events': events,
            'flaps': flaps,
            'per_second': per_sec_df,
            'movement': movement_df,
            'per_minute': per_min_df,
            'raw_data': df
        }
        
    except Exception as e:
        print(f"Error processing {csv_path}: {str(e)}")
        return None

def combine_daily_results(daily_results):
    """Combine results from multiple files for a single day"""
    if not daily_results:
        return None
    
    # Filter out None results
    valid_results = [r for r in daily_results if r is not None]
    if not valid_results:
        return None
    
    combined = {
        'files_processed': [r['filename'] for r in valid_results],
        'observation_periods': len(valid_results),
        'total_duration_minutes': 0,
        'date': None,
        'first_observation': None,
        'last_observation': None
    }
    
    # Combine all data
    all_events = []
    all_flaps = []
    all_per_second = []
    all_movement = []
    all_per_minute = []
    
    start_times = []
    end_times = []
    
    for result in valid_results:
        if result['start_time']:
            start_times.append(result['start_time'])
            # Estimate end time based on data length
            duration_seconds = len(result['per_second']) if not result['per_second'].empty else 0
            end_time = result['start_time'] + timedelta(seconds=duration_seconds)
            end_times.append(end_time)
        
        # Add file identifier to distinguish between observation periods
        file_prefix = result['filename'].replace('.csv', '').replace('_raw', '')
        
        # Events with file identifier
        for event in result['events']:
            event_copy = event.copy()
            event_copy['file'] = file_prefix
            event_copy['observation_period'] = result['filename']
            if result['start_time']:
                event_copy['absolute_timestamp'] = result['start_time'] + timedelta(seconds=event['second'])
            all_events.append(event_copy)
        
        # Flaps with file identifier
        for flap in result['flaps']:
            flap_copy = flap.copy()
            flap_copy['file'] = file_prefix
            flap_copy['observation_period'] = result['filename']
            if result['start_time']:
                flap_copy['absolute_timestamp'] = result['start_time'] + timedelta(seconds=flap['second'])
            all_flaps.append(flap_copy)
        
        # Per-second data with file identifier
        if not result['per_second'].empty:
            per_sec_copy = result['per_second'].copy()
            per_sec_copy['file'] = file_prefix
            per_sec_copy['observation_period'] = result['filename']
            all_per_second.append(per_sec_copy)
        
        # Movement data
        if not result['movement'].empty:
            mov_copy = result['movement'].copy()
            mov_copy['file'] = file_prefix
            mov_copy['observation_period'] = result['filename']
            all_movement.append(mov_copy)
        
        # Per-minute data
        if not result['per_minute'].empty:
            min_copy = result['per_minute'].copy()
            min_copy['file'] = file_prefix
            min_copy['observation_period'] = result['filename']
            all_per_minute.append(min_copy)
    
    # Calculate summary statistics
    if start_times:
        combined['first_observation'] = min(start_times)
        combined['last_observation'] = max(end_times) if end_times else max(start_times)
        combined['date'] = combined['first_observation'].date()
        
        # Calculate total observation time (sum of individual periods)
        total_seconds = sum(len(r['per_second']) for r in valid_results if not r['per_second'].empty)
        combined['total_duration_minutes'] = total_seconds / 60
    
    # Convert lists to DataFrames
    combined['events_df'] = pd.DataFrame(all_events)
    combined['flaps_df'] = pd.DataFrame(all_flaps)
    combined['per_second_df'] = pd.concat(all_per_second, ignore_index=True) if all_per_second else pd.DataFrame()
    combined['movement_df'] = pd.concat(all_movement, ignore_index=True) if all_movement else pd.DataFrame()
    combined['per_minute_df'] = pd.concat(all_per_minute, ignore_index=True) if all_per_minute else pd.DataFrame()
    
    return combined

def generate_daily_summary_report(combined_data, output_path):
    """Generate comprehensive daily summary report"""
    with open(output_path, 'w') as f:
        f.write("DAILY SEABIRD BEHAVIOR ANALYSIS SUMMARY\n")
        f.write("=" * 60 + "\n\n")
        
        # Date and observation info
        if combined_data['date']:
            f.write(f"DATE: {combined_data['date'].strftime('%Y-%m-%d')}\n")
        
        if combined_data['first_observation'] and combined_data['last_observation']:
            f.write(f"OBSERVATION PERIOD: {combined_data['first_observation'].strftime('%H:%M:%S')} to {combined_data['last_observation'].strftime('%H:%M:%S')}\n")
        
        f.write(f"NUMBER OF OBSERVATION PERIODS: {combined_data['observation_periods']}\n")
        f.write(f"TOTAL OBSERVATION TIME: {combined_data['total_duration_minutes']:.1f} minutes\n\n")
        
        f.write("FILES PROCESSED:\n")
        for filename in combined_data['files_processed']:
            f.write(f"  - {filename}\n")
        f.write("\n")
        
        # Event summary
        if not combined_data['events_df'].empty:
            events = combined_data['events_df']
            arrivals = events[events['type'] == 'arrival']
            departures = events[events['type'] == 'departure']
            fish_arrivals = arrivals[arrivals['arrival_with_fish'] == True] if 'arrival_with_fish' in arrivals.columns else pd.DataFrame()
            
            f.write("DAILY EVENT SUMMARY:\n")
            f.write(f"  Total Arrivals: {len(arrivals)}\n")
            f.write(f"  Total Departures: {len(departures)}\n")
            f.write(f"  Arrivals with Fish: {len(fish_arrivals)}\n")
            f.write(f"  Event Rate: {len(events) / combined_data['total_duration_minutes']:.2f} events per minute\n\n")
            
            # Events by observation period
            f.write("EVENTS BY OBSERVATION PERIOD:\n")
            for period in combined_data['files_processed']:
                period_events = events[events['observation_period'] == period]
                f.write(f"  {period}: {len(period_events)} events\n")
            f.write("\n")
            
            # Timeline of all events
            if 'absolute_timestamp' in events.columns:
                f.write("EVENT TIMELINE:\n")
                for _, event in events.sort_values('absolute_timestamp').iterrows():
                    timestamp = pd.to_datetime(event['absolute_timestamp']).strftime('%H:%M:%S')
                    f.write(f"  {timestamp} - {event['type'].title()} (in {event['observation_period']})\n")
                f.write("\n")
        
        # Movement summary
        if not combined_data['movement_df'].empty:
            movement = combined_data['movement_df'].dropna(subset=['movement_delta'])
            if not movement.empty:
                f.write("MOVEMENT SUMMARY:\n")
                f.write(f"  Average movement per second: {movement['movement_delta'].mean():.1f} pixels\n")
                f.write(f"  Maximum movement: {movement['movement_delta'].max():.1f} pixels\n")
                f.write(f"  Total seconds with movement data: {len(movement)}\n\n")
        
        # Flapping summary
        if not combined_data['flaps_df'].empty:
            flaps = combined_data['flaps_df']
            f.write("FLAPPING SUMMARY:\n")
            f.write(f"  Total flapping events: {len(flaps)}\n")
            f.write(f"  Average area multiplier: {flaps['multiplier'].mean():.2f}\n")
            f.write(f"  Maximum area multiplier: {flaps['multiplier'].max():.2f}\n")
            f.write(f"  Flapping rate: {len(flaps) / combined_data['total_duration_minutes']:.3f} events per minute\n\n")
        
        # Per-period summary
        if not combined_data['per_second_df'].empty:
            per_sec = combined_data['per_second_df']
            f.write("ACTIVITY SUMMARY:\n")
            f.write(f"  Average adults per second: {per_sec['count_adult'].mean():.1f}\n")
            f.write(f"  Peak adult count: {per_sec['count_adult'].max()}\n")
            f.write(f"  Total adult detections: {per_sec['count_adult'].sum()}\n")
            
            if 'count_chick' in per_sec.columns:
                f.write(f"  Average chicks per second: {per_sec['count_chick'].mean():.1f}\n")
                f.write(f"  Peak chick count: {per_sec['count_chick'].max()}\n")
            
            if 'count_fish' in per_sec.columns:
                f.write(f"  Total fish detections: {per_sec['count_fish'].sum()}\n")

def plot_daily_overview(combined_data, output_dir, date_str):
    """Create comprehensive daily overview plot"""
    fig = plt.figure(figsize=(20, 14))
    fig.suptitle(f'Daily Seabird Activity Overview - {date_str}', fontsize=20, fontweight='bold')
    
    gs = fig.add_gridspec(4, 3, hspace=0.4, wspace=0.3)
    
    # Daily statistics (top left)
    ax1 = fig.add_subplot(gs[0, 0])
    stats_text = [
        f"Date: {date_str}",
        f"Observation Periods: {combined_data['observation_periods']}",
        f"Total Duration: {combined_data['total_duration_minutes']:.1f} min"
    ]
    
    if not combined_data['events_df'].empty:
        events = combined_data['events_df']
        arrivals = len(events[events['type'] == 'arrival'])
        departures = len(events[events['type'] == 'departure'])
        stats_text.extend([
            f"Total Arrivals: {arrivals}",
            f"Total Departures: {departures}",
            f"Event Rate: {len(events)/combined_data['total_duration_minutes']:.2f}/min"
        ])
    
    if not combined_data['flaps_df'].empty:
        stats_text.append(f"Flapping Events: {len(combined_data['flaps_df'])}")
    
    ax1.text(0.05, 0.95, '\n'.join(stats_text), transform=ax1.transAxes, fontsize=11,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.axis('off')
    ax1.set_title('Daily Summary', fontweight='bold')
    
    # Event timeline across all observation periods (top middle/right)
    ax2 = fig.add_subplot(gs[0, 1:])
    if not combined_data['events_df'].empty and 'absolute_timestamp' in combined_data['events_df'].columns:
        events = combined_data['events_df']
        arrivals = events[events['type'] == 'arrival']
        departures = events[events['type'] == 'departure']
        
        if not arrivals.empty:
            arrival_times = pd.to_datetime(arrivals['absolute_timestamp'])
            ax2.scatter(arrival_times, [1]*len(arrival_times), c='green', s=80, 
                       alpha=0.8, label=f'Arrivals ({len(arrivals)})', marker='^')
        
        if not departures.empty:
            departure_times = pd.to_datetime(departures['absolute_timestamp'])
            ax2.scatter(departure_times, [0.5]*len(departure_times), c='red', s=80,
                       alpha=0.8, label=f'Departures ({len(departures)})', marker='v')
        
        ax2.set_ylabel('Event Type')
        ax2.set_xlabel('Time of Day')
        ax2.set_title('Event Timeline Across All Observation Periods')
        ax2.set_ylim(0, 1.5)
        ax2.set_yticks([0.5, 1])
        ax2.set_yticklabels(['Departure', 'Arrival'])
        
        # Set x-axis to always show full 24 hours (00:00 to 23:59)
        if combined_data['date']:
            start_of_day = pd.Timestamp(combined_data['date'])
            end_of_day = start_of_day + pd.Timedelta(days=1)
            ax2.set_xlim(start_of_day, end_of_day)
        
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        ax2.xaxis.set_major_locator(mdates.HourLocator(interval=2))
        ax2.tick_params(axis='x', rotation=45)
    
    # Activity by observation period (second row)
    ax3 = fig.add_subplot(gs[1, :])
    if not combined_data['per_second_df'].empty:
        per_sec = combined_data['per_second_df']
        
        # Plot activity for each observation period with single color
        periods = per_sec['observation_period'].unique()
        
        for period in periods:
            period_data = per_sec[per_sec['observation_period'] == period]
            if 'timestamp' in period_data.columns:
                x_data = pd.to_datetime(period_data['timestamp'])
                ax3.plot(x_data, period_data['count_adult'], color='steelblue', 
                        linewidth=1.5, alpha=0.7)
        
        ax3.set_ylabel('Adult Count')
        ax3.set_xlabel('Time of Day')
        ax3.set_title('Adult Activity Across All Observation Periods')
        
        # Set x-axis to always show full 24 hours (00:00 to 23:59)
        if combined_data['date']:
            start_of_day = pd.Timestamp(combined_data['date'])
            end_of_day = start_of_day + pd.Timedelta(days=1)
            ax3.set_xlim(start_of_day, end_of_day)
        
        ax3.grid(True, alpha=0.3)
        ax3.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        ax3.xaxis.set_major_locator(mdates.HourLocator(interval=2))
        ax3.tick_params(axis='x', rotation=45)
    
    # Flapping rate and fish arrivals timeline (bottom left)
    ax4 = fig.add_subplot(gs[2, 0])
    
    # Create a secondary y-axis for flapping rate
    ax4_twin = ax4.twinx()
    
    # Plot flapping rate as rolling mean (plot first, lower z-order)
    if not combined_data['flaps_df'].empty and 'absolute_timestamp' in combined_data['flaps_df'].columns:
        flaps = combined_data['flaps_df'].copy()
        flaps['timestamp'] = pd.to_datetime(flaps['absolute_timestamp'])
        
        # Create full day time range with 1-minute resolution
        if combined_data['date']:
            start_of_day = pd.Timestamp(combined_data['date'])
            end_of_day = start_of_day + pd.Timedelta(days=1)
            time_range = pd.date_range(start=start_of_day, end=end_of_day, freq='1min')
            
            # Count flapping events per minute
            flaps_per_min = flaps.groupby(pd.Grouper(key='timestamp', freq='1min')).size()
            flaps_per_min = flaps_per_min.reindex(time_range, fill_value=0)
            
            # Apply rolling mean (5-minute window)
            rolling_window = 5
            flaps_rolling = flaps_per_min.rolling(window=rolling_window, center=True, min_periods=1).mean()
            
            # Plot on secondary axis with lower z-order
            ax4_twin.plot(flaps_rolling.index, flaps_rolling.values, color='purple', 
                         linewidth=2, alpha=0.8, label=f'Flapping Rate (5-min avg)', zorder=1)
            ax4_twin.set_ylabel('Flapping Events per Minute', color='purple')
            ax4_twin.tick_params(axis='y', labelcolor='purple')
            ax4_twin.set_xlim(start_of_day, end_of_day)
    
    # Set twin axis to be behind primary axis
    ax4_twin.set_zorder(1)
    ax4.set_zorder(2)
    # Make primary axis background transparent so twin axis is visible
    ax4.patch.set_visible(False)
    
    # Plot fish arrivals on primary axis (higher z-order so they appear on top)
    if not combined_data['events_df'].empty and 'absolute_timestamp' in combined_data['events_df'].columns:
        events = combined_data['events_df']
        arrivals = events[events['type'] == 'arrival']
        
        if 'arrival_with_fish' in arrivals.columns:
            fish_arrivals = arrivals[arrivals['arrival_with_fish'] == True]
            if not fish_arrivals.empty:
                fish_times = pd.to_datetime(fish_arrivals['absolute_timestamp'])
                ax4.scatter(fish_times, [1]*len(fish_arrivals), c='orange', s=150, 
                           alpha=0.9, label=f'Fish Arrivals ({len(fish_arrivals)})', 
                           marker='*', edgecolors='darkgoldenrod', linewidths=1.5, zorder=10)
        
        ax4.set_ylabel('Fish Arrivals', color='orange')
        ax4.set_ylim(0, 2)
        ax4.set_yticks([1])
        ax4.set_yticklabels([''])
        ax4.tick_params(axis='y', labelcolor='orange')
        
        # Set x-axis to always show full 24 hours
        if combined_data['date']:
            start_of_day = pd.Timestamp(combined_data['date'])
            end_of_day = start_of_day + pd.Timedelta(days=1)
            ax4.set_xlim(start_of_day, end_of_day)
    
    ax4.set_xlabel('Time of Day')
    ax4.set_title('Flapping Rate and Fish Arrivals Timeline')
    ax4.grid(True, alpha=0.3)
    ax4.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    ax4.xaxis.set_major_locator(mdates.HourLocator(interval=2))
    ax4.tick_params(axis='x', rotation=45)
    
    # Combine legends from both axes (flapping first, then fish)
    lines2, labels2 = ax4_twin.get_legend_handles_labels()
    lines1, labels1 = ax4.get_legend_handles_labels()
    ax4.legend(lines2 + lines1, labels2 + labels1, loc='upper left')
    
    # Flapping events spatial distribution (bottom middle)
    ax5 = fig.add_subplot(gs[2, 1])
    if not combined_data['flaps_df'].empty:
        flaps = combined_data['flaps_df']
        # Plot with regular coordinates
        scatter = ax5.scatter(flaps['center_x'], flaps['center_y'], 
                            c=flaps['multiplier'], s=60, alpha=0.7, cmap='plasma')
        ax5.set_xlabel('X Position')
        ax5.set_ylabel('Y Position')
        ax5.set_title('Flapping Event Locations')
        # Invert y-axis so higher values (top of image) are at top of plot
        ax5.invert_yaxis()
        plt.colorbar(scatter, ax=ax5, label='Area Multiplier')
    
    # Activity distribution (bottom right)
    ax6 = fig.add_subplot(gs[2, 2])
    if not combined_data['per_second_df'].empty:
        per_sec = combined_data['per_second_df']
        ax6.hist(per_sec['count_adult'], bins=20, alpha=0.7, color='blue', edgecolor='black')
        ax6.axvline(per_sec['count_adult'].mean(), color='red', linestyle='--',
                   label=f'Mean: {per_sec["count_adult"].mean():.1f}')
        ax6.set_xlabel('Adult Count')
        ax6.set_ylabel('Frequency (seconds)')
        ax6.set_title('Adult Count Distribution')
        ax6.legend()
        ax6.grid(True, alpha=0.3)
    
    # Hourly activity summary (bottom full width)
    ax7 = fig.add_subplot(gs[3, :])
    if not combined_data['events_df'].empty and 'absolute_timestamp' in combined_data['events_df'].columns:
        events = combined_data['events_df']
        events['hour'] = pd.to_datetime(events['absolute_timestamp']).dt.hour
        
        hourly_arrivals = events[events['type'] == 'arrival']['hour'].value_counts().sort_index()
        hourly_departures = events[events['type'] == 'departure']['hour'].value_counts().sort_index()
        
        # Create complete hour range
        all_hours = range(24)
        arrival_counts = [hourly_arrivals.get(h, 0) for h in all_hours]
        departure_counts = [hourly_departures.get(h, 0) for h in all_hours]
        
        width = 0.35
        ax7.bar([h - width/2 for h in all_hours], arrival_counts, width, 
               label='Arrivals', alpha=0.8, color='green')
        ax7.bar([h + width/2 for h in all_hours], departure_counts, width,
               label='Departures', alpha=0.8, color='red')
        
        ax7.set_xlabel('Hour of Day')
        ax7.set_ylabel('Number of Events')
        ax7.set_title('Hourly Activity Pattern')
        ax7.set_xticks(all_hours)
        ax7.legend()
        ax7.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'daily_overview_{date_str}.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()

def main():
    parser = argparse.ArgumentParser(description='Batch analyze seabird behavior across multiple days')
    parser.add_argument('input_dir', help='Directory containing CSV files')
    parser.add_argument('--station', '-s', required=True,
                       help='Station name (e.g., FAR3, FAR1, etc.)')
    parser.add_argument('--output_dir', '-o', default='daily_analysis/', 
                       help='Output directory for daily analysis results')
    parser.add_argument('--fps', type=int, default=1, help='Sampling rate (samples per second)')
    parser.add_argument('--original_video_fps', type=int, default=25, 
                       help='Original video FPS')
    parser.add_argument('--conf_thresh', type=float, default=0.25, 
                       help='Confidence threshold for detections')
    parser.add_argument('--smooth_window_s', type=int, default=3)
    parser.add_argument('--error_window_s', type=int, default=10)
    parser.add_argument('--hold_seconds', type=int, default=8)
    parser.add_argument('--fish_window_s', type=int, default=5)
    parser.add_argument('--movement_smoothing_s', type=int, default=5)
    parser.add_argument('--flap_area_multiplier', type=float, default=3.0)
    parser.add_argument('--flap_baseline_s', type=int, default=30)
    
    args = parser.parse_args()
    
    # Find all CSV files
    csv_pattern = os.path.join(args.input_dir, "*.csv")
    csv_files = glob.glob(csv_pattern)
    
    if not csv_files:
        print(f"No CSV files found in {args.input_dir}")
        return
    
    print(f"Found {len(csv_files)} CSV files")
    
    # Group files by date
    date_groups = group_files_by_date(csv_files)
    print(f"Processing {len(date_groups)} days for station {args.station}:")
    for date, files in date_groups.items():
        print(f"  {date}: {len(files)} files")
    
    # Create output directory with station subfolder
    station_output_dir = os.path.join(args.output_dir, args.station)
    os.makedirs(station_output_dir, exist_ok=True)
    print(f"Output will be saved to: {station_output_dir}")
    
    # Process each day
    for date, files in date_groups.items():
        date_str = date.strftime('%Y%m%d')
        print(f"\nProcessing {args.station} - {date_str}...")
        
        # Create daily output directory under station folder
        daily_output_dir = os.path.join(station_output_dir, date_str)
        os.makedirs(daily_output_dir, exist_ok=True)
        
        # Process all files for this day
        daily_results = []
        for filepath in sorted(files):
            result = process_single_file(
                filepath, fps=args.fps, original_video_fps=args.original_video_fps,
                conf_thresh=args.conf_thresh, smooth_window_s=args.smooth_window_s,
                error_window_s=args.error_window_s, hold_seconds=args.hold_seconds,
                fish_window_s=args.fish_window_s, movement_smoothing_s=args.movement_smoothing_s,
                flap_area_multiplier=args.flap_area_multiplier, flap_baseline_s=args.flap_baseline_s
            )
            if result:
                daily_results.append(result)
        
        if not daily_results:
            print(f"No valid results for {date_str}")
            continue
        
        # Combine daily results
        combined_data = combine_daily_results(daily_results)
        if not combined_data:
            print(f"Failed to combine results for {date_str}")
            continue
        
        # Save combined CSV files
        csv_output_dir = os.path.join(daily_output_dir, 'csv')
        os.makedirs(csv_output_dir, exist_ok=True)
        
        if not combined_data['events_df'].empty:
            combined_data['events_df'].to_csv(
                os.path.join(csv_output_dir, f'daily_events_{date_str}.csv'), index=False)
        
        if not combined_data['flaps_df'].empty:
            combined_data['flaps_df'].to_csv(
                os.path.join(csv_output_dir, f'daily_flaps_{date_str}.csv'), index=False)
        
        if not combined_data['per_second_df'].empty:
            combined_data['per_second_df'].to_csv(
                os.path.join(csv_output_dir, f'daily_per_second_{date_str}.csv'), index=False)
        
        if not combined_data['movement_df'].empty:
            combined_data['movement_df'].to_csv(
                os.path.join(csv_output_dir, f'daily_movement_{date_str}.csv'), index=False)
        
        if not combined_data['per_minute_df'].empty:
            combined_data['per_minute_df'].to_csv(
                os.path.join(csv_output_dir, f'daily_per_minute_{date_str}.csv'), index=False)
        
        # Generate daily summary report
        report_path = os.path.join(daily_output_dir, f'daily_summary_{date_str}.txt')
        generate_daily_summary_report(combined_data, report_path)
        print(f"Generated daily report: {report_path}")
        
        # Generate daily overview plot
        plots_output_dir = os.path.join(daily_output_dir, 'plots')
        os.makedirs(plots_output_dir, exist_ok=True)
        plot_daily_overview(combined_data, plots_output_dir, date_str)
        print(f"Generated daily overview plot: {plots_output_dir}/daily_overview_{date_str}.png")
        
        print(f"Completed {args.station} - {date_str}: {len(daily_results)} observation periods processed")
    
    print(f"\nBatch processing complete for station {args.station}!")
    print(f"Results saved to {station_output_dir}")

if __name__ == "__main__":
    main()