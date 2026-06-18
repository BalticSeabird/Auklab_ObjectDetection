# Complete Workflow: Annotate, Retrain, Reassess Stage4

## Quick Start

```bash
# Step 1: Sample 200 events
python code/validation/sample_and_prepare_annotations.py \
    --num-events 200 \
    --output-dir data/annotation_batch

# Step 2: MANUAL - View videos and annotate CSV
# Browse: data/annotation_batch/videos/
# Edit: data/class_validation_new_batch.csv

# Step 3: Merge annotations
python code/validation/merge_annotations.py \
    --new-batch data/class_validation_new_batch.csv \
    --existing data/class_validation.csv \
    --output data/class_validation.csv

# Step 4: Retrain model
python code/postprocess/train_stage4_tri3_model.py \
    --validation-csv data/class_validation.csv \
    --output-model models/stage4/tri3_fish_arrival_model.json

# Step 5: Reassess all events
python code/inference_system/stage4_batch_runner.py \
    --events-db-root data/events_db \
    --model-path models/stage4/tri3_fish_arrival_model.json

# Step 6: Generate new clips with improved model
python code/validation/generate_wwf_clips.py \
    --num-events 300 \
    --new-fish-only
```

## What Was Created

| File | Purpose |
|------|---------|
| `code/validation/sample_and_prepare_annotations.py` | Sample 200 diverse events from databases |
| `code/validation/merge_annotations.py` | Merge new annotations with existing data |
| `code/inference_system/stage4_batch_runner.py` | Reassess all events with new model |
| `ANNOTATION_AND_RETRAINING_GUIDE.md` | Detailed step-by-step guide |
| `code/validation/generate_wwf_clips.py` | (Already exists, now with `--new-fish-only` flag) |
| `code/validation/generate_wwf_clips.py` | (Already updated with `--skip-stations` flag) |

## Key Improvements Made

### 1. Fixed `--new-fish-only` flag
- Was: Events with `is_new_fish_arrival=0` were included despite the flag
- Now: Properly filters to only `is_new_fish_arrival=1` events

### 2. Created Annotation Workflow
- Script to sample 200 strategically diverse events
- Copies videos to local folder for easy viewing
- Creates CSV template with all required metadata

### 3. Created Retraining Infrastructure
- Script to merge new annotations with existing data
- Batch reassessment script to update all events in databases
- Prevents data duplication and maintains data integrity

## Annotation Tips

**Focus on:**
- Hard cases (events the model currently gets wrong)
- Edge cases (partial fish, multiple birds, etc.)
- High-quality labels (be conservative with positive labels)

**Columns to fill:**
1. `valid_arrival` - Is this a real bird arrival? (0/1)
2. `valid_fish` - Are fish visible? (0/1)
3. **`valid_fish_arrival` - Bird arrival WITH fish?** (0/1) ⭐ **MAIN TARGET**
4. `valid_multiple_fish` - Multiple fish? (0/1)
5. `comment` - Why you labeled it that way (optional)

## Expected Improvement

- **Starting point:** ~74% accuracy
- **After 200 good annotations:** ~78-82% accuracy
- **Key metric:** Precision (avoid false positives flooding WWF)

## Troubleshooting

**Q: No videos copied?**
- Check that event videos exist: `ls /mnt/BSP_NAS2_work/auklab_model/event_data/*/`
- Verify detections CSVs exist for each event

**Q: Model not improving?**
- Ensure you're labeling `valid_fish_arrival` consistently
- Focus on hard examples (current misclassifications)
- Check training report for class imbalance

**Q: Events not reassessed?**
- Verify model path is correct
- Check logs: `tail logs/stage4_batch_reassessment_*.log`
- Ensure event databases have completed Stage3 processing

---

**Full documentation:** See `ANNOTATION_AND_RETRAINING_GUIDE.md`
