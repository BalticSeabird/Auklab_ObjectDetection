# Using the Active Learning Pipeline with Date Subfolders

## ✅ Your Directory Structure is Fully Supported!

The pipeline **automatically searches recursively** through all subdirectories, so your date-organized structure works perfectly:

```
/mnt/BSP_NAS2_work/.../1FPS/TRI3/
├── 2025-06-28/
│   ├── TRI3_20250628T000000_raw.csv
│   ├── TRI3_20250628T001002_raw.csv
│   └── ...
├── 2025-06-29/
│   └── ...
└── 2025-07-01/
    └── ...

/mnt/BSP_NAS2_vol4/Video/Video2025/TRI3/
├── 2025-06-28/
│   ├── TRI3_20250628T000000.mkv
│   ├── TRI3_20250628T001002.mkv
│   └── ...
└── ...
```

## Quick Start

```bash
# 1. Use the ready-made TRI3 config
python code/active_learning/run_active_learning_pipeline.py \
    --config code/active_learning/config_TRI3.yaml

# 2. Or create your own for a different station
cp code/active_learning/config_TRI3.yaml code/active_learning/config_TRI4.yaml
# Edit paths for TRI4, then run:
python code/active_learning/run_active_learning_pipeline.py \
    --config code/active_learning/config_TRI4.yaml
```

## How It Works

1. **CSV Discovery**: Uses `rglob("*_raw.csv")` to recursively find all CSV files in all date subfolders
2. **Video Indexing**: Uses `rglob("*.mkv")` to recursively find all videos in all date subfolders
3. **Matching**: Matches CSV basenames to video basenames automatically (e.g., `TRI3_20250628T000000_raw.csv` → `TRI3_20250628T000000.mkv`)

## Examples

### Process 100 Random Files from All Dates
```yaml
sampling:
  enabled: true
  n_files: 100
  random_seed: 42
```

### Process Only June 2025
```yaml
sampling:
  enabled: true
  n_files: 100
  include_patterns:
    - ".*202506.*"
```

### Process Only Specific Dates
```yaml
sampling:
  enabled: true
  include_patterns:
    - ".*20250628.*"
    - ".*20250629.*"
    - ".*20250630.*"
```

### Process ALL Files (No Sampling)
```yaml
sampling:
  enabled: false
```

## What the Pipeline Does

1. **Discovers files**: Searches recursively through all date folders
2. **Samples**: Randomly selects specified number of CSVs (or all if sampling disabled)
3. **Analyzes**: Identifies problematic frames (edge detections, spikes, dips, etc.)
4. **Extracts**: Pulls those frames from matching video files
5. **Pre-annotates**: Runs your model on extracted frames for easier manual annotation

## Troubleshooting

**"Found 0 CSV files"**
- Check the `csv_dir` path is correct
- Ensure CSVs end with `_raw.csv`
- Try: `ls /mnt/BSP_NAS2_work/.../1FPS/TRI3/*/*.csv` to verify files exist

**"Video not found"**
- Verify video filenames match CSV basenames
- CSV: `TRI3_20250628T000000_raw.csv` → Video: `TRI3_20250628T000000.mkv`
- Check `video_dirs` paths
- Try: `find /mnt/BSP_NAS2_vol4/Video/Video2025/TRI3 -name "*.mkv" | head`

**"Found X CSV files but only Y videos indexed"**
- Some CSVs may not have matching videos (or vice versa) - this is OK
- The script will skip CSVs without videos during extraction
- Check video extensions in config match your files

## Next Steps

After the pipeline completes:
1. Check `data/active_learning_TRI3_batch1/frames/` for extracted frames
2. Check `data/active_learning_TRI3_batch1/frames/annotations/` for pre-annotations
3. Upload to Roboflow/CVAT for correction
4. Export corrected annotations
5. Add to training dataset and retrain model

## See Also

- [SCALING_GUIDE.md](SCALING_GUIDE.md) - Complete documentation
- [README.md](README.md) - Original active learning workflow
- [QUICK_REFERENCE.md](QUICK_REFERENCE.md) - Quick command reference
