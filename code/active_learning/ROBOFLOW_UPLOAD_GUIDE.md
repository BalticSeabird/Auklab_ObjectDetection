# Uploading Active Learning Batches to Roboflow - Quick Guide

## Setup (One-Time)

### 1. Install Roboflow Package
```bash
pip install roboflow
```

### 2. Get Your API Key
1. Go to https://app.roboflow.com
2. Click on your profile (top right)
3. Go to Settings → Roboflow API
4. Copy your API key

### 3. Set API Key (Recommended)
```bash
# Add to your ~/.bashrc or ~/.zshrc
export ROBOFLOW_API_KEY="your_key_here"

# Or set for current session
export ROBOFLOW_API_KEY="your_key_here"
```

## Quick Upload

### Simple Command (All Batches with Pre-Annotations)
```bash
python code/active_learning/upload_batches_simple.py \
    data/active_learning_TRI3_batch1/frames \
    --with-annotations
```

**Features:**
- ✅ **Progress bars** for each batch
- ✅ **Auto-resume** if connection drops
- ✅ Saves progress every 10 images
- ✅ Skips already uploaded files

### Resume After Interruption

If upload stops (network issue, Ctrl+C, etc.), just run the **same command again**:

```bash
# Upload was interrupted? Just run again:
python code/active_learning/upload_batches_simple.py \
    data/active_learning_TRI3_batch1/frames \
    --with-annotations

# Script will ask: "Resume from previous upload? (y/n)"
# Answer 'y' to continue where you left off
```

### Start Fresh (Ignore Previous Progress)

```bash
python code/active_learning/upload_batches_simple.py \
    data/active_learning_TRI3_batch1/frames \
    --with-annotations \
    --no-resume
```

### Upload Specific Batches Only
```bash
# Only fish detections
python code/active_learning/upload_batches_simple.py \
    data/active_learning_TRI3_batch1/frames \
    --batches fish

# Fish and edge detections
python code/active_learning/upload_batches_simple.py \
    data/active_learning_TRI3_batch1/frames \
    --batches fish edge_detection

# All batches without pre-annotations (annotate from scratch)
python code/active_learning/upload_batches_simple.py \
    data/active_learning_TRI3_batch1/frames
```

## What Happens During Upload

The script will:
1. **Check for previous progress** (if upload was interrupted before)
2. Connect to your Roboflow workspace and project
3. Show you a summary of what will be uploaded
4. Ask for confirmation (y/n)
5. Upload each batch separately with **progress bars**:
   - `edge_detection/` → "active_learning_edge_detection" batch
   - `spike/` → "active_learning_spike" batch
   - `dip/` → "active_learning_dip" batch
   - `high_count/` → "active_learning_high_count" batch
   - `count_transition/` → "active_learning_count_transition" batch
   - `fish/` → "active_learning_fish" batch
6. **Save progress** every 10 images (for resume capability)
7. Tag each image with its problem type
8. Include pre-annotations (if --with-annotations specified)
9. **Handle interruptions gracefully** - you can resume anytime

### Progress Tracking

You'll see a progress bar for each batch:
```
[1/6] Uploading: Fish Detections
--------------------------------------------------------------------------------
  Uploading fish: 100%|████████████████████| 45/45 [01:23<00:00,  1.85s/img]
  ✓ Batch complete - Uploaded: 45, Failed: 0
```

### Resume After Interruption

If upload stops mid-way:
- Progress is **automatically saved** in `.upload_state.json`
- Just **run the same command again**
- Script will detect previous progress and ask to resume
- Already uploaded files are **automatically skipped**

## After Upload - In Roboflow

### 1. View Your Batches
- Go to your project in Roboflow
- Click on "Images" tab
- Filter by batch name (e.g., "active_learning_fish")
- Or filter by tag (e.g., "fish")

### 2. Review Annotations
- Click on any image to review
- Correct any errors:
  - **False positives**: Delete incorrect boxes
  - **False negatives**: Add missing detections
  - **Wrong class**: Change bird ↔ fish classification
  - **Poor boxes**: Adjust bounding box size/position

### 3. Annotation Tips by Problem Type

**Edge Detections** (edge_detection batch)
- Focus on birds partially visible at frame borders
- Decide whether to include or exclude partial birds
- Be consistent with your annotation policy

**High Count Scenes** (high_count batch)
- These are crowded scenes - double-check all birds are detected
- Look for overlapping birds
- Ensure correct class labels (adult/chick/fish)

**Fish Frames** (fish batch)
- Verify fish vs bird classification
- Add any missed fish
- Important for multi-species model performance

**Spike/Dip Frames** (spike/dip batches)
- These had detection inconsistencies
- Check for false positives (causing spikes)
- Check for missed detections (causing dips)

**Count Transitions** (count_transition batch)
- Critical for event detection
- Ensure accurate counts at arrival/departure moments

### 4. Export When Done
1. Click "Generate" → "Export Dataset"
2. Choose format: "YOLOv8"
3. Download the corrected annotations
4. Extract to your training data directory

## Advanced Options

### Full Control Upload
```bash
python code/active_learning/upload_to_roboflow.py \
    --frames-dir data/active_learning_TRI3_batch1/frames \
    --api-key YOUR_KEY \
    --workspace ai-course-2024 \
    --project fish_seabirds_combined-625bd \
    --batches fish edge_detection \
    --use-annotations \
    --split train \
    --batch-name-prefix active_learning_TRI3
```

### Different Workspace/Project
```bash
python code/active_learning/upload_batches_simple.py \
    data/frames \
    --workspace YOUR_WORKSPACE \
    --project YOUR_PROJECT \
    --with-annotations
```

## Troubleshooting

### "roboflow package not installed"
```bash
pip install roboflow tqdm  # tqdm for progress bars
```

### Upload Interrupted / Connection Lost

**Don't worry!** Just run the same command again:
```bash
python code/active_learning/upload_batches_simple.py \
    data/active_learning_TRI3_batch1/frames \
    --with-annotations
```

The script will:
1. Detect previous progress
2. Ask if you want to resume (answer 'y')
3. Skip all already uploaded files
4. Continue from where it stopped

**Technical details:**
- Progress saved in `data/active_learning_TRI3_batch1/frames/.upload_state.json`
- Tracks: uploaded files, completed batches
- Auto-deleted after successful completion

### "API key required"
Make sure to:
```bash
export ROBOFLOW_API_KEY="your_actual_key"
```
Or the script will prompt you to enter it

### "No images found in batch"
- Check that frames were extracted: `ls data/frames_for_annotation/fish/`
- Ensure you're pointing to the correct directory

### Upload is slow

**Normal behavior:**
- Roboflow API typically processes 0.5-2 images/second
- For 100 images, expect 1-3 minutes per batch
- Includes image upload + processing + annotation conversion

**If it stops:**
- Check your network connection
- Press Ctrl+C to cancel gracefully
- Run again to resume from where it stopped

**Tips for large uploads:**
- Upload during off-peak hours
- Split into smaller batches using `--batches` option:
  ```bash
  # Upload one batch at a time
  python upload_batches_simple.py data/frames --batches fish
  python upload_batches_simple.py data/frames --batches edge_detection
  # etc.
  ```

### Images already exist in Roboflow
- Roboflow will skip duplicates (based on filename)
- If you want to re-upload, rename or delete in Roboflow first

## Workflow Summary

```
Extract frames → Pre-annotate → Upload to Roboflow → Review/Correct → Export → Retrain
     ↓              ↓                ↓                    ↓            ↓         ↓
   pipeline    pre_annotate   upload_batches      Roboflow UI    Download   train.py
```

## Next Steps After Upload

1. **Review in Roboflow** (~2-4 hours depending on batch size)
2. **Export corrected annotations**
3. **Combine with existing training data**:
   ```bash
   cp active_learning_batch/images/* dataset/seabird_fish*/images/train/
   cp active_learning_batch/labels/* dataset/seabird_fish*/labels/train/
   ```
4. **Retrain your model** with improved annotations
5. **Re-run event detection** and compare results
6. **Iterate** if needed with new samples

## Tips

- **Start with fish batch first** - Usually smallest, good for testing workflow
- **Review in small chunks** - 20-30 images at a time to stay focused
- **Be consistent** - Use same annotation standards as your training data
- **Save regularly** - Roboflow auto-saves, but good practice to export checkpoints
- **Use keyboard shortcuts** in Roboflow for faster annotation

## Support

If you encounter issues:
1. Check Roboflow documentation: https://docs.roboflow.com
2. Verify your API key is valid
3. Check your workspace/project names are correct
4. Ensure you have upload permissions on the project
