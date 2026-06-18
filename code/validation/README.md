# WWF Video Clips Generation

This directory contains utilities for generating and managing video clips for the WWF citizen science app.

## `generate_wwf_clips.py`

Automatically generates batches of high-quality video clips from detected seabird arrival events with high fish detection counts. These clips are designed for crowdsourced validation and citizen science engagement.

### Features

- **Event Sampling**: Intelligently samples events from all station event databases
- **Quality Filtering**: Selects only events with sufficient fish detections (configurable threshold)
- **Batch Management**: Auto-increments batch numbers, with each batch stored in its own directory
- **SQLite Database**: Creates a per-batch database containing all event metadata
- **Video Extraction**: Creates short video clips without running object detection (fast processing)
- **Metadata Logging**: Generates summary JSON and CSV files for easy tracking
- **Cloud-Ready**: Output is structured for mirroring to cloud storage and linking to citizen apps

### Usage

**Basic usage with defaults (50 events, min 3 fish detections):**
```bash
cd /home/jonas/Documents/vscode/Auklab_OD
source .venv/bin/activate
python code/validation/generate_wwf_clips.py
```

**Custom parameters:**
```bash
python3 code/validation/generate_wwf_clips.py \
    --num-events 100 \
    --min-fish-detections 10 \
    --clip-before 4.0 \
    --clip-after 16.0 \
    --events-db-root data/events_db \
    --output-base /mnt/BSP_NAS2_work/temp
```

**Dry-run with test output:**
```bash
python code/validation/generate_wwf_clips.py \
    --num-events 10 \
    --min-fish-detections 1 \
    --output-base /tmp/wwf_test
```

### Command-Line Options

| Option | Default | Description |
|--------|---------|-------------|
| `--num-events` | 50 | Number of events to sample for this batch |
| `--min-fish-detections` | 3 | Minimum number of fish detections to include event |
| `--events-db-root` | `data/events_db` | Root directory containing per-station event databases |
| `--output-base` | `/mnt/BSP_NAS2_work` | Base output directory on NAS2_work |
| `--clip-before` | 2.0 | Seconds before event to include in clip |
| `--clip-after` | 8.0 | Seconds after event to include in clip |
| `--log-dir` | `logs/wwf_clips` | Log directory for operation logs |

### Output Structure

Each batch is organized as follows (with a unique name):

```
/mnt/BSP_NAS2_work/wwf_clips_brave_falcon_427/
├── wwf_clips_brave_falcon_427.db  # SQLite database with event metadata
├── video/
│   ├── event_id_1.mp4             # Video clip files
│   ├── event_id_2.mp4
│   └── ...
└── metadata/
    ├── batch_summary.json         # Batch metadata and statistics
    └── events_summary.csv         # Event data in CSV format
```

### Database Schema

The `wwf_clips_batch{X}.db` SQLite database contains a single `wwf_clips` table with the following key fields:

**Event Identifiers:**
- `event_id`: Unique event identifier
- `station`: Camera/station name
- `date`, `absolute_timestamp`: When the event occurred

**Detection Data:**
- `fish_detections_stage4`: Number of fish detected in stage4 classification
- `fish_avg_confidence_stage4`: Average confidence of fish detections
- `arrival_with_fish_stage2`: Whether stage2 detected fish (boolean)
- `is_actual_arrival`, `is_new_fish_arrival`: Validation flags

**Video Information:**
- `original_video_path`: Path to source video file
- `clip_path`: Path to generated clip file
- `second`: Time offset in seconds within the original video

**Staging Information:**
- `event_video_path`: Path to stage3 extracted clip (if available)
- `stage3_status`: Status of stage3 processing
- `stage4_rule_version`, `stage4_rule_hits`: Stage4 decision metadata

All fields from the per-station event databases are preserved for full traceability.

### Batch Management

The script generates **unique, non-reusable batch names** to prevent accidental overwrites:

**Naming Format:** `wwf_clips_{adjective}_{noun}_{number}`

**Examples:**
- `wwf_clips_brave_falcon_427`
- `wwf_clips_golden_eagle_891`
- `wwf_clips_open_bear_102`

Each batch is guaranteed to have a unique name that:
- ✅ Will never be reused across runs
- ✅ Is easily identifiable and memorable
- ✅ Prevents accidental batch overwrites
- ✅ Supports unlimited batch generation

No manual batch numbering needed—run the script as many times as you want!

### Logs

Operation logs are saved to `logs/wwf_clips/` with timestamps:
- `wwf_clips_20260507_151248.log`

Check logs for detailed information about:
- Events sampled from each station
- Video clip creation success/failure
- Database operations
- Any warnings or errors encountered

### Integration with Cloud Storage

The output batch directory can be easily synced to cloud storage:

```bash
# Example: rsync to cloud staging
rsync -avz --delete \
  /mnt/BSP_NAS2_work/wwf_clips_brave_falcon_427/ \
  cloud-storage:wwf/batch-falcon-427/
```

The batch can then be linked to the citizen science app for crowdsourced validation.

**Tip:** Use the unique batch name in your cloud storage path for easy tracking!

### Requirements

- Python 3.11+
- Required packages (already in .venv):
  - pandas
  - opencv-python
  - ffmpeg (system utility)
  - ffprobe (system utility)

### Troubleshooting

**No events found:**
- Check that `data/events_db` contains SQLite database files
- Lower `--min-fish-detections` threshold
- Verify events have `is_actual_arrival = 1`

**Video files not found:**
- Events must have `original_video_path` set in database
- Video files must be accessible at the stored path
- Some events may point to NAS paths that aren't currently mounted

**Clips not created:**
- Check logs in `logs/wwf_clips/` for detailed errors
- Verify ffmpeg/ffprobe are installed and accessible
- Check disk space on output location

**Database errors:**
- Ensure write permissions on output directory
- Check for corrupted database files in logs

### Performance Notes

- Event sampling: ~1 second
- Video clip extraction: ~1-5 seconds per clip (depends on clip length)
- For 50 clips: typically 1-3 minutes total
- Batch database: SQLite with WAL mode for concurrent access safety

### Example Workflow

1. **Generate a production batch:**
   ```bash
   python code/validation/generate_wwf_clips.py \
       --num-events 200 \
       --min-fish-detections 5
   # Creates: /mnt/BSP_NAS2_work/wwf_clips_brave_falcon_427/
   ```

2. **Verify the batch:**
   ```bash
   ls -lh /mnt/BSP_NAS2_work/wwf_clips_brave_falcon_427/
   sqlite3 /mnt/BSP_NAS2_work/wwf_clips_brave_falcon_427/wwf_clips_brave_falcon_427.db \
       "SELECT COUNT(*) FROM wwf_clips WHERE clip_path IS NOT NULL;"
   ```

3. **Generate another batch** (automatic unique naming):
   ```bash
   python code/validation/generate_wwf_clips.py \
       --num-events 200 \
       --min-fish-detections 5
   # Creates: /mnt/BSP_NAS2_work/wwf_clips_golden_eagle_891/
   ```

4. **Sync to cloud:**
   ```bash
   ./code/file_handling/sync_with_remote.sh \
       /mnt/BSP_NAS2_work/wwf_clips_brave_falcon_427/ \
       remote:wwf/batch-falcon-427/
   ```

5. **Link to citizen app:**
   - Update app configuration with batch URL
   - Citizens can now validate clips!
