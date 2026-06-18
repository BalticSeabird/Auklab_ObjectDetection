# Stage4 Model Annotation & Retraining Guide

## Overview

This guide walks you through annotating new validation data and retraining the Stage4 model.

---

## Step 1: Sample and Prepare Events (DONE - Use the script)

```bash
python code/validation/sample_and_prepare_annotations.py \
    --num-events 200 \
    --output-dir data/annotation_batch \
    --annotation-csv data/class_validation_new_batch.csv
```

This creates:
- `data/annotation_batch/videos/` - All 200 event videos (named by event_id)
- `data/annotation_batch/detections_csv/` - Detection CSVs for each event
- `data/class_validation_new_batch.csv` - Template CSV ready for annotation

---

## Step 2: Annotate Videos (MANUAL WORK)

### Video Viewing
Browse the video files in `data/annotation_batch/videos/`. For each video:

1. **Watch the clip carefully** (typically 2-10 seconds)
2. **Look for the bird's arrival** (sudden motion into frame)
3. **Check for fish in the water** (look for motion, shapes, or behavior changes)

### Fill the CSV Template

Open `data/class_validation_new_batch.csv` and fill these columns for each row:

| Column | Value | Meaning |
|--------|-------|---------|
| `valid_arrival` | 0 or 1 | Is this a real bird arrival? (0=No, 1=Yes) |
| `valid_fish` | 0 or 1 | Are fish visible? (0=No, 1=Yes) |
| `valid_fish_arrival` | 0 or 1 | **Bird arrival WITH fish?** (0=No, 1=Yes) ⭐ **TARGET** |
| `valid_multiple_fish` | 0 or 1 | Multiple fish visible? (0=No, 1=Yes) |
| `comment` | Text | Optional: Why did you label it this way? |

**Key Focus**: `valid_fish_arrival` is the main target for Stage4 training. This is what the model will learn to predict.

### Example Annotations

```
Event: bird_arrives_with_fish.mp4
→ valid_arrival = 1 (yes, bird arrived)
→ valid_fish = 1 (yes, fish visible)
→ valid_fish_arrival = 1 (yes, arrival WITH fish - POSITIVE EXAMPLE)
→ comment = "Clear fish splash in frame 15-20"

Event: bird_arrives_no_fish.mp4
→ valid_arrival = 1 (yes, bird arrived)
→ valid_fish = 0 (no fish visible)
→ valid_fish_arrival = 0 (no, arrival but NO fish - NEGATIVE EXAMPLE)
→ comment = "Water too calm, no fish activity"

Event: false_motion.mp4
→ valid_arrival = 0 (no, just wave motion)
→ valid_fish = 0 (no fish)
→ valid_fish_arrival = 0 (not even a real arrival)
→ comment = "Just water turbulence, not a bird"
```

---

## Step 3: Merge Annotations into Main Database

After annotating, merge your new annotations with the existing `data/class_validation.csv`:

```bash
python code/validation/merge_annotations.py \
    --new-batch data/class_validation_new_batch.csv \
    --existing data/class_validation.csv \
    --output data/class_validation.csv
```

If that script doesn't exist, use this Python code:

```python
import pandas as pd

# Load both CSVs
existing = pd.read_csv("data/class_validation.csv", sep=";")
new = pd.read_csv("data/class_validation_new_batch.csv", sep=";")

# Filter out rows that were just templates (with empty valid_* columns)
new_annotated = new[new["valid_fish_arrival"].notna()].copy()

# Combine
merged = pd.concat([existing, new_annotated], ignore_index=True)

# Remove duplicates by event_id (keep new annotations)
merged = merged.drop_duplicates(subset=["event_id"], keep="last")

# Save
merged.to_csv("data/class_validation.csv", sep=";", index=False)
print(f"Merged {len(new_annotated)} new annotations. Total: {len(merged)} records")
```

---

## Step 4: Retrain the Stage4 Model

### 4A: Train with all data

```bash
python code/postprocess/train_stage4_tri3_model.py \
    --validation-csv data/class_validation.csv \
    --station TRI3 \
    --target valid_fish_arrival \
    --train-ratio 0.7 \
    --output-model models/stage4/tri3_fish_arrival_model.json \
    --output-report data/stage4_tri3_training_report.json
```

**Parameters:**
- `--validation-csv`: Path to your merged annotations file
- `--station`: Which station to train on (TRI3 recommended)
- `--target`: The label column to predict (`valid_fish_arrival`)
- `--train-ratio`: Training/test split (0.7 = 70% train, 30% test)
- `--output-model`: Where to save the trained model
- `--output-report`: Training metrics report (JSON)

### 4B: Check Training Results

After training, review `data/stage4_tri3_training_report.json`:

```bash
cat data/stage4_tri3_training_report.json | python -m json.tool
```

Look for:
- **Train accuracy**: Should be >75%
- **Test accuracy**: Should be >70% (if much lower, model might be overfitting)
- **Precision/Recall**: Check if model favors false positives or false negatives
- **Feature importance**: Which features matter most?

**Example good report:**
```json
{
  "train_accuracy": 0.78,
  "test_accuracy": 0.74,
  "model_threshold": 0.5,
  "feature_means": [...],
  "feature_stds": [...],
  "class_distribution": {
    "positive": 145,
    "negative": 348
  }
}
```

### 4C: Evaluate on Test Set

```bash
python code/postprocess/train_stage4_tri3_model.py \
    --validation-csv data/class_validation.csv \
    --target valid_fish_arrival \
    --output-report data/stage4_evaluation.json
```

---

## Step 5: Reassess All Events in Event Databases

Once you have a new trained model, you need to run Stage4 classification on all events in your event databases.

### 5A: Run Stage4 Batch Reassessment

```bash
python code/inference_system/stage4_batch_runner.py \
    --events-db-root data/events_db \
    --model-path models/stage4/tri3_fish_arrival_model.json \
    --config config/system_config.yaml \
    --stations TRI3,TRI6,ROST2,ROST3,ROST4,ROST5,ROST6,FAR3,FAR6,BONDEN5,BONDEN6
```

**What this does:**
1. Reads all events from event databases
2. Extracts Stage4 features from detection CSVs
3. Runs model prediction on each event
4. Updates `is_new_fish_arrival`, `fish_detections_stage4`, `stage4_model_score`, `stage4_decision_source` in database
5. Logs results to `logs/stage4_batch_runner.log`

### 5B: Monitor Reassessment Progress

```bash
tail -f logs/stage4_batch_runner.log
```

Look for:
- `Processing station: TRI3` - Which station is being processed
- `Events processed: 1234` - Progress counter
- Errors or warnings about missing files

### 5C: Verify Update Success

After reassessment completes, check the database:

```bash
# Count how many events got new stage4 scores
sqlite3 data/events_db/TRI3_events.db \
  "SELECT COUNT(*) as count FROM events WHERE stage4_decision_source = 'model'"

# Sample some updated events
sqlite3 data/events_db/TRI3_events.db \
  "SELECT event_id, is_new_fish_arrival, stage4_model_score, stage4_decision_source FROM events LIMIT 10"
```

---

## Step 6: Generate New WWF Clips (with improved model)

Now with the retrained model, generate fresh clips:

```bash
python code/validation/generate_wwf_clips.py \
    --num-events 300 \
    --min-fish-detections 30 \
    --events-db-root data/events_db \
    --output-base /mnt/BSP_NAS2_work/temp \
    --new-fish-only
```

The `--new-fish-only` flag now uses your updated Stage4 model! 🎉

---

## Troubleshooting

### Training fails: "No eligible events found"
- **Cause:** No annotations in `class_validation.csv` with `valid_fish_arrival` set
- **Fix:** Make sure you filled the CSV and merged it correctly

### Model produces same predictions as before
- **Cause:** The new training data is too similar to old data
- **Fix:** Focus annotation efforts on **hard examples** (currently misclassified events)

### Reassessment is slow
- **Cause:** Reading all detections CSVs takes time (with many events)
- **Tip:** You can run on individual stations:
  ```bash
  python code/inference_system/stage4_batch_runner.py \
      --events-db-root data/events_db \
      --stations TRI3
  ```

### Stage4 classifier still giving bad results
- **Strategy:** Annotate more examples, especially:
  - False positives from current model (high score but no fish)
  - False negatives from current model (low score but has fish)
  - Edge cases (partial fish, multiple arrivals, etc.)

---

## Quick Reference

| Step | Command | Purpose |
|------|---------|---------|
| 1 | `python code/validation/sample_and_prepare_annotations.py` | Sample 200 events |
| 2 | Manual | View videos and fill CSV |
| 3 | Python merge script | Combine with existing annotations |
| 4 | `python code/postprocess/train_stage4_tri3_model.py` | Retrain model |
| 5 | `python code/inference_system/stage4_batch_runner.py` | Update all events |
| 6 | `python code/validation/generate_wwf_clips.py --new-fish-only` | Generate new clips |

---

## Expected Results

**After annotation and retraining:**
- Model accuracy should improve by 5-15% (depending on annotation quality)
- False positive rate should decrease
- More accurate `valid_fish_arrival` predictions
- Better WWF clip batches with fewer false arrivals

**If you annotate 200 high-quality samples:**
- Expect ~150-180 to be useful (some are false arrivals)
- Should improve model from ~74% → ~78-82% accuracy
- Focus on **precision** (don't tag false arrivals as fish arrivals) to avoid flooding WWF with bad data
