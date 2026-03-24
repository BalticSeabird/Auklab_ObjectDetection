# Stage3 Active Learning Pipeline - Quick Guide

This guide shows how to run the new Stage3-based active learning pipeline, control how many frames are generated, and upload to an existing Roboflow project.

Important: this pipeline now scans Stage3 `_events.csv` files and samples from the original videos, not from boxed event clips.

## 1. Configure

Copy the template config and edit it:

```bash
cp config/config_stage3_branch.yaml my_stage3_branch.yaml
```

In `my_stage3_branch.yaml`, set:

- `paths.stage3_roots`: path(s) to your Stage3 event-data roots (`station/date/events/*_events.csv`)
- `paths.output_dir`: where this pipeline should write artifacts

## 2. Control Number of Generated Frames

Frame volume is controlled by:

- `sampling.max_events`: max number of fish-arrival events to sample
- `sampling.frames_per_event`: number of sampled frames per event
- Optional: `sampling.event_time_offsets_seconds` for explicit offsets around event time

### Upper-bound formula

```text
total_frames_max = max_events * frames_per_event
```

Example:

- `max_events: 80`
- `frames_per_event: 2`
- upper bound: `160` frames

Actual output can be lower if fewer eligible clips match your filters.

## 3. Run Index + Sample + Pre-Annotation

```bash
python3 code/active_learning/run_stage3_active_learning_branch.py \
  --config my_stage3_branch.yaml \
  --steps index sample annotate
```

This will:

1. Index Stage3 clips
2. Filter fish-arrival events
3. Find original videos from `original_video_path`
4. Extract frames around event times
5. Pre-annotate frames using your configured model and class thresholds

## 4. Upload to Existing Roboflow Project

Set these in config:

- `upload.enabled: true`
- `upload.workspace: <your_workspace>`
- `upload.project: <your_existing_project_slug>`
- `upload.batch_name_prefix: active_learning_stage3` (or your preferred prefix)

Set your API key:

```bash
export ROBOFLOW_API_KEY="your_api_key"
```

Then run upload only:

```bash
python3 code/active_learning/run_stage3_active_learning_branch.py \
  --config my_stage3_branch.yaml \
  --steps upload
```

## 5. One-Command Full Run

If upload is enabled in config:

```bash
python3 code/active_learning/run_stage3_active_learning_branch.py \
  --config my_stage3_branch.yaml \
  --steps index sample annotate upload
```

## 6. Output Locations

Under `paths.output_dir` (default: `data/active_learning_stage3_branch`):

- `stage3_event_index.json`
- `stage3_event_index.csv`
- `frames/` (sampled frames by category)
- `frames/extraction_manifest.json`
- `frames/annotations/` (YOLO + JSON pre-annotations)

## 7. Recommended First Run

Start small for validation:

- `max_events: 40`
- `frames_per_event: 2`
- narrow filters (`include_stations`, `date_from`, `date_to`)

Once quality looks good, increase quotas.

## Related Scripts

- `code/active_learning/run_stage3_active_learning_branch.py`
- `code/active_learning/index_stage3_clips.py`
- `code/active_learning/sample_frames_from_stage3_clips.py`
- `code/active_learning/pre_annotate_stage3_branch.py`
- `code/active_learning/upload_stage3_branch.py`
