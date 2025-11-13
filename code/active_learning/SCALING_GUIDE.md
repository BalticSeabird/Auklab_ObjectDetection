# Scalable Active Learning Pipeline - Setup Guide

## Overview

This enhanced pipeline supports large-scale active learning with:
- **Random sampling** of stations/dates from large datasets
- **Configurable paths** for CSV outputs and video files
- **Multi-directory video search** 
- **YAML-based configuration** for easy customization
- **Flexible filtering** by patterns (stations, dates, etc.)

## Quick Start

### 1. Create Your Configuration File

Copy the template and customize:

```bash
cp code/active_learning/config_template.yaml code/active_learning/my_config.yaml
```

Edit `my_config.yaml` with your paths:

```yaml
paths:
  csv_dir: "/path/to/your/csv_detection_outputs"
  video_dirs:
    - "/path/to/your/videos"
  output_dir: "data/active_learning_large_scale"

sampling:
  enabled: true
  n_files: 100  # Process 100 random files
  random_seed: 42

extraction:
  max_per_type: 100  # Extract up to 100 frames per problem type

pre_annotation:
  model_path: "models/auklab_model_xlarge_combined_4564_v1.pt"
  confidence: 0.25
```

### 2. Run the Pipeline

```bash
# Using config file (recommended)
python code/active_learning/run_active_learning_pipeline.py --config code/active_learning/my_config.yaml

# Or with command-line arguments
python code/active_learning/run_active_learning_pipeline.py \
    --csv-dir /path/to/csvs \
    --video-dirs /path/to/videos \
    --output-dir data/active_learning_batch \
    --sample-size 100
```

### 3. Review Results

The pipeline creates:
```
data/active_learning_large_scale/
├── problematic_frames.json          # Analysis results
└── frames/                          # Extracted frames
    ├── edge_detection/
    ├── spike/
    ├── dip/
    ├── high_count/
    ├── count_transition/
    ├── extraction_manifest.json
    ├── frames_list.csv
    └── annotations/                 # Pre-annotations (if enabled)
        ├── yolo/                    # YOLO format
        └── json/                    # Detailed JSON
```

## Advanced Usage

### Process Specific Stations or Dates

Use pattern filters in your config:

```yaml
sampling:
  enabled: true
  n_files: 50
  include_patterns:
    - ".*TRI3.*"           # Only TRI3 station
    - ".*202506.*"         # Only June 2025
  exclude_patterns:
    - ".*test.*"           # Exclude test files
```

### Run Specific Steps Only

```bash
# Only identify problems (don't extract)
python code/active_learning/run_active_learning_pipeline.py \
    --config my_config.yaml \
    --steps identify

# Only extract frames (assumes problematic_frames.json exists)
python code/active_learning/run_active_learning_pipeline.py \
    --config my_config.yaml \
    --steps extract

# Skip pre-annotation
python code/active_learning/run_active_learning_pipeline.py \
    --config my_config.yaml \
    --steps identify extract
```

### Process All Files (No Sampling)

```bash
python code/active_learning/run_active_learning_pipeline.py \
    --config my_config.yaml \
    --no-sampling
```

Or in config:
```yaml
sampling:
  enabled: false
```

## Configuration Reference

### Path Configuration

```yaml
paths:
  # Where your CSV detection outputs are stored
  # Can have subdirectories organized by station/date
  csv_dir: "/path/to/csv_outputs"
  
  # List of directories to search for video files
  # The script searches recursively in all directories
  video_dirs:
    - "/path/to/videos/station1"
    - "/path/to/videos/station2"
    - "/mnt/nas/more_videos"
  
  # Where to save all results
  output_dir: "data/active_learning_batch"
```

### Sampling Configuration

```yaml
sampling:
  enabled: true              # Enable/disable sampling
  strategy: "random"         # Currently only "random" supported
  n_files: 100              # Number of CSV files to sample
  random_seed: 42           # For reproducibility
  
  # Optional filters (regex patterns)
  include_patterns:
    - ".*TRI3.*"            # Only match files with "TRI3"
    - ".*2025.*"            # Only match files with "2025"
  
  exclude_patterns:
    - ".*test.*"            # Exclude test files
    - ".*backup.*"          # Exclude backups
```

### Detection Parameters

```yaml
detection:
  frame_width: 2688
  frame_height: 1520
  confidence_threshold: 0.25
  edge_margin: 100              # pixels from border
  max_spike_duration: 2         # seconds
  max_dip_duration: 3           # seconds
  high_count_threshold: 10      # birds
```

### Extraction Parameters

```yaml
extraction:
  max_per_type: 100             # Max frames per problem type
  priority: "diverse"           # "diverse" or "concentrated"
  video_extensions:
    - ".mkv"
    - ".mp4"
    - ".avi"
    - ".mov"
```

### Pre-annotation Parameters

```yaml
pre_annotation:
  model_path: "models/your_model.pt"
  confidence: 0.25              # Lower = more detections
  enabled: true                 # Set false to skip
```

## Workflow Comparison

### Old Workflow (Small Scale)
```bash
# Hardcoded paths, processes all files in csv_detection_1fps/
python code/active_learning/identify_problem_frames.py
python code/active_learning/extract_frames.py --video-dir video
python code/active_learning/pre_annotate_frames.py
```

### New Workflow (Large Scale)
```bash
# Configurable paths, random sampling, multi-directory support
python code/active_learning/run_active_learning_pipeline.py --config my_config.yaml
```

## Directory Structure Examples

The script **searches recursively** for files, so date subfolders are handled automatically!

### Example 1: Flat Structure
```
csv_outputs/
├── TRI3_20250628T000000_raw.csv
├── TRI3_20250628T001002_raw.csv
├── TRI3_20250629T000000_raw.csv
└── ...

videos/
├── TRI3_20250628T000000.mkv
├── TRI3_20250628T001002.mkv
└── ...
```

### Example 2: Organized by Station/Date (Automatic!)
```
csv_outputs/TRI3/
├── 2025-06-28/
│   ├── TRI3_20250628T000000_raw.csv
│   └── TRI3_20250628T001002_raw.csv
├── 2025-06-29/
│   ├── TRI3_20250629T000000_raw.csv
│   └── TRI3_20250629T001002_raw.csv
└── 2025-07-01/
    └── ...

videos/TRI3/
├── 2025-06-28/
│   ├── TRI3_20250628T000000.mkv
│   └── TRI3_20250628T001002.mkv
├── 2025-06-29/
│   └── ...
└── ...
```

### Example 3: Your Actual NAS Structure
```
/mnt/BSP_NAS2_work/.../1FPS/TRI3/
├── 2025-06-28/
│   ├── TRI3_20250628T000000_raw.csv
│   └── ...
├── 2025-06-29/
│   └── ...
└── ...

/mnt/BSP_NAS2_vol4/Video/Video2025/TRI3/
├── 2025-06-28/
│   ├── TRI3_20250628T000000.mkv
│   └── ...
└── ...
```

**All structures work!** The script uses `rglob()` which searches recursively through all subdirectories. Just point `csv_dir` to the top-level directory (e.g., `.../1FPS/TRI3/`) and it will find all CSVs in all date subfolders.

## Tips for Large Datasets

1. **Start with sampling**: Test with 50-100 files before processing thousands
   ```yaml
   sampling:
     enabled: true
     n_files: 50
   ```

2. **Use pattern filters**: Focus on specific stations or time periods
   ```yaml
   sampling:
     include_patterns:
       - ".*TRI3.*202506.*"  # TRI3 station, June 2025
   ```

3. **Monitor disk space**: Each extracted frame is ~500KB-2MB
   - 100 frames per type × 5 types = ~250-1000MB per run

4. **Batch annotation**: Extract multiple sets with different random seeds
   ```bash
   # Run 1
   python run_active_learning_pipeline.py --config config1.yaml  # seed=42
   
   # Run 2 (different seed)
   python run_active_learning_pipeline.py --config config2.yaml  # seed=123
   ```

5. **Keep track of results**: Use descriptive output directories
   ```yaml
   output_dir: "data/active_learning_TRI3_June_batch1"
   ```

## Troubleshooting

### "No CSV files found"
- Check `csv_dir` path is correct
- Ensure files end with `_raw.csv`
- Check include/exclude patterns aren't too restrictive

### "Video not found"
- Verify video filename matches CSV basename (without `_raw`)
- Check `video_dirs` paths
- Ensure video extensions are in config (`.mkv`, `.mp4`, etc.)

### "No detections" for many files
- Check `confidence_threshold` (try lowering to 0.20)
- Verify CSV files have actual detection data
- Check frame dimensions match your videos

### Memory issues with large datasets
- Reduce `n_files` in sampling
- Reduce `max_per_type` in extraction
- Process in multiple batches

## Next Steps After Pipeline Completes

1. **Review extracted frames**: Check that problem types make sense
2. **Upload to annotation tool**: Roboflow, CVAT, Label Studio, etc.
3. **Correct pre-annotations**: Much faster than annotating from scratch
4. **Export annotations**: In YOLOv8 format
5. **Merge with training data**: Add to existing dataset
6. **Retrain model**: Use updated dataset
7. **Evaluate improvement**: Re-run inference and event detection
8. **Iterate**: Run pipeline again with new model if needed

## Dependencies

```bash
pip install pyyaml opencv-python numpy pandas ultralytics
```

## Support

The new pipeline is fully compatible with the original workflow. You can still use the individual scripts if needed:

```bash
# Original scripts still work
python code/active_learning/identify_problem_frames.py
python code/active_learning/extract_frames.py
python code/active_learning/pre_annotate_frames.py
```
