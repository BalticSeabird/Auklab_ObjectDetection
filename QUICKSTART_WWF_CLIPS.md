# Quick Start: WWF Video Clips Generation

## Installation (One-Time Setup)
Already included in your `.venv`. No additional installation needed!

## Basic Usage

Navigate to your Auklab_OD directory and activate the virtual environment:

```bash
cd ~/Documents/vscode/Auklab_OD
source .venv/bin/activate
```

### Generate Your First Batch

```bash
python code/validation/generate_wwf_clips.py
```

This will:
- Sample 50 events with at least 3 fish detections
- Create a **unique batch** in `/mnt/BSP_NAS2_work/wwf_clips_{word}_{word}_{number}/`
- Example: `/mnt/BSP_NAS2_work/wwf_clips_brave_falcon_427/`
- Extract video clips with text overlays
- Generate a SQLite database with all metadata

### Generate with Custom Parameters

```bash
python code/validation/generate_wwf_clips.py \
  --num-events 100 \
  --min-fish-detections 5 \
  --clip-before 1.5 \
  --clip-after 10.0
```

### Test Before Production

```bash
python code/validation/generate_wwf_clips.py \
  --num-events 10 \
  --min-fish-detections 1 \
  --output-base /tmp/wwf_test
```

## Check Your Output

```bash
# List all batches (each with unique name)
ls -lh /mnt/BSP_NAS2_work/wwf_clips_*/

# View batch summary
cat /mnt/BSP_NAS2_work/wwf_clips_brave_falcon_427/metadata/batch_summary.json

# Query the database
sqlite3 /mnt/BSP_NAS2_work/wwf_clips_brave_falcon_427/wwf_clips_brave_falcon_427.db \
  "SELECT COUNT(*) FROM wwf_clips WHERE clip_path IS NOT NULL;"
```

## Verify Video Clips

```bash
# List generated clips
ls -lh /mnt/BSP_NAS2_work/wwf_clips_brave_falcon_427/video/

# Play a clip (if you have a player)
vlc /mnt/BSP_NAS2_work/wwf_clips_brave_falcon_427/video/6080_*.mp4
```

## Troubleshooting

### No clips created?
1. Check the logs:
   ```bash
   tail -50 logs/wwf_clips/*.log
   ```

2. Lower the `--min-fish-detections` threshold

3. Verify event databases exist:
   ```bash
   ls data/events_db/*.db
   ```

### Permission denied on NAS2_work?
Change output location:
```bash
python code/validation/generate_wwf_clips.py \
  --output-base /tmp/wwf_test
```

### Video files not found?
Some events may reference videos on network mounts that aren't currently accessible. The script logs these and continues with accessible videos.

## Next Steps

1. **Review batch metadata:**
   ```bash
   cat /mnt/BSP_NAS2_work/wwf_clips_brave_falcon_427/metadata/events_summary.csv
   ```

2. **Mirror to cloud storage:**
   ```bash
   rsync -avz --delete \
     /mnt/BSP_NAS2_work/wwf_clips_brave_falcon_427/ \
     your-cloud:wwf/batch-falcon-427/
   ```

3. **Link to citizen science app** with the batch folder URL

4. **Generate next batch** (automatic unique naming):
   ```bash
   python code/validation/generate_wwf_clips.py
   ```
   This will create a different batch like `wwf_clips_golden_eagle_891/`

## All Parameters

| Parameter | Default | What It Does |
|-----------|---------|--------------|
| `--num-events` | 50 | How many videos to include in batch |
| `--min-fish-detections` | 3 | Minimum fish detected in event |
| `--clip-before` | 2.0 | Seconds of context before event |
| `--clip-after` | 8.0 | Seconds of context after event |
| `--events-db-root` | data/events_db | Where to find event databases |
| `--output-base` | /mnt/BSP_NAS2_work | Where to save batches |
| `--log-dir` | logs/wwf_clips | Where to save logs |

## Batch Naming

Batches are named with **unique identifiers** that are never reused:

**Format:** `wwf_clips_{adjective}_{noun}_{number}`

**Examples:**
- `wwf_clips_brave_falcon_427`
- `wwf_clips_golden_eagle_891`
- `wwf_clips_open_bear_102`
- `wwf_clips_icy_reef_944`

This ensures you can:
- ✅ Run the script unlimited times without conflicts
- ✅ Track batches by memorable names
- ✅ Never accidentally overwrite previous batches

## Full Documentation

See `code/validation/README.md` for comprehensive documentation.

## Questions?

Check the logs directory for detailed operation logs:
```bash
tail -100 logs/wwf_clips/*.log
```
