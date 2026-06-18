# Stage4 Improvement Strategy - Testing Plan

## ✅ What's Done

1. **Temporal features added** - Replaced simple early/late with dynamic arrival tracking
2. **Filtering script created** - Can test with/without problematic stations
3. **Code verified** - All syntax checks passed

---

## 📋 Testing Strategy (Recommended Order)

### Phase 1: Establish Baseline
```bash
# 1a. Sample 200 events (includes all stations)
python code/validation/sample_and_prepare_annotations.py \
    --num-events 200 \
    --output-dir data/annotation_batch

# 1b. Annotate the 200 events
# Watch videos in: data/annotation_batch/videos/
# Fill CSV: data/class_validation_new_batch.csv

# 1c. Merge annotations
python code/validation/merge_annotations.py \
    --new-batch data/class_validation_new_batch.csv \
    --existing data/class_validation.csv \
    --output data/class_validation.csv

# 1d. Train baseline model (all data including ROST2-4)
python code/postprocess/train_stage4_tri3_model.py \
    --validation-csv data/class_validation.csv \
    --output-model models/stage4/tri3_fish_arrival_baseline.json \
    --output-report data/stage4_baseline_report.json

# Note baseline accuracy from the report
```

### Phase 2: Test Without Problematic Stations
```bash
# 2a. Create filtered dataset (exclude ROST2-4)
python code/validation/filter_validation_data.py \
    --input data/class_validation.csv \
    --output data/class_validation_no_rost24.csv \
    --exclude-stations ROST2,ROST3,ROST4

# 2b. Train on high-quality data only
python code/postprocess/train_stage4_tri3_model.py \
    --validation-csv data/class_validation_no_rost24.csv \
    --output-model models/stage4/tri3_fish_arrival_quality.json \
    --output-report data/stage4_quality_report.json

# Compare accuracy with baseline
```

### Phase 3: Test Single Station Model
```bash
# 3a. Create TRI3-only dataset
python code/validation/filter_validation_data.py \
    --input data/class_validation.csv \
    --output data/class_validation_tri3_only.csv \
    --include-stations TRI3

# 3b. Train on TRI3 only
python code/postprocess/train_stage4_tri3_model.py \
    --validation-csv data/class_validation_tri3_only.csv \
    --output-model models/stage4/tri3_fish_arrival_tri3only.json \
    --output-report data/stage4_tri3_report.json

# This will likely have highest accuracy (but only works for TRI3)
```

---

## 📊 Metrics to Compare

| Test | Model File | Key Metrics | Decision |
|------|-----------|------------|----------|
| Baseline (all data) | `tri3_fish_arrival_baseline.json` | Train/test accuracy, precision, recall | Reference |
| Quality (no ROST) | `tri3_fish_arrival_quality.json` | Accuracy gain/loss | Does ROST hurt training? |
| TRI3 only | `tri3_fish_arrival_tri3only.json` | Max achievable accuracy | Upper bound |

**Compare reports:**
```bash
# View metrics
cat data/stage4_baseline_report.json | python -m json.tool | grep -E "(accuracy|precision|recall|f1)"
cat data/stage4_quality_report.json | python -m json.tool | grep -E "(accuracy|precision|recall|f1)"
cat data/stage4_tri3_report.json | python -m json.tool | grep -E "(accuracy|precision|recall|f1)"
```

---

## 🎯 Decision Tree

```
                        Baseline Accuracy
                             |
                _______________+_______________
               |                               |
           > 78%                          < 78%
               |                               |
               v                               v
        Good results!              Need investigation
               |                               |
        Does Quality                   Check quality
        model improve?               report for hints
               |
        ____YES__|__NO____
       |                  |
       v                  v
   Use quality      ROST2-4 helps!
   model           Use baseline
   (exclude ROST)   (include all)
```

---

## 🔍 What to Look For in Reports

**Key sections in training report JSON:**
- `train_accuracy` - Should be >75%
- `test_accuracy` - Should be >70%
- `precision` - High = fewer false positives (good for WWF)
- `recall` - High = catch more real arrivals
- `feature_importance` (if available) - Which features matter?

**Red flags:**
- `train >> test accuracy` = Overfitting (bad)
- `precision < 50%` = Lots of false positives (bad for WWF)
- `recall < 60%` = Missing real arrivals (bad)

---

## 🚀 After Testing - Choose Best Model

Based on results:

**Option A: Use quality model (no ROST2-4)**
```bash
# Reassess events with quality model
python code/inference_system/stage4_batch_runner.py \
    --events-db-root data/events_db \
    --model-path models/stage4/tri3_fish_arrival_quality.json \
    --stations TRI3,TRI6,FAR3,FAR6,BONDEN5,BONDEN6
    # Note: Skip ROST2-4 in --stations
```

**Option B: Use baseline (all stations)**
```bash
# Reassess with baseline model (works for all)
python code/inference_system/stage4_batch_runner.py \
    --events-db-root data/events_db \
    --model-path models/stage4/tri3_fish_arrival_baseline.json
    # All stations included
```

**Option C: Separate models per station (advanced)**
- Train separate models for ROST2-4 (low quality)
- Train separate models for others (high quality)
- Use appropriate model for each station during reassessment
- Most complex but potentially best results

---

## 📝 Quick Command Reference

**Sample & annotate:**
```bash
python code/validation/sample_and_prepare_annotations.py --num-events 200
# Manual: annotate data/class_validation_new_batch.csv
python code/validation/merge_annotations.py \
    --new-batch data/class_validation_new_batch.csv \
    --existing data/class_validation.csv \
    --output data/class_validation.csv
```

**Test scenarios:**
```bash
# Baseline
python code/postprocess/train_stage4_tri3_model.py \
    --validation-csv data/class_validation.csv \
    --output-model models/stage4/tri3_fish_arrival_baseline.json

# Without ROST2-4
python code/validation/filter_validation_data.py \
    --input data/class_validation.csv \
    --output data/class_validation_no_rost24.csv \
    --exclude-stations ROST2,ROST3,ROST4
python code/postprocess/train_stage4_tri3_model.py \
    --validation-csv data/class_validation_no_rost24.csv \
    --output-model models/stage4/tri3_fish_arrival_quality.json
```

**Reassess & generate clips:**
```bash
python code/inference_system/stage4_batch_runner.py \
    --events-db-root data/events_db \
    --model-path models/stage4/tri3_fish_arrival_quality.json

python code/validation/generate_wwf_clips.py \
    --num-events 300 \
    --new-fish-only
```

---

## ⏭️ Next Steps

1. **Annotate the 200 events** (manual work)
2. **Run Phase 1** to establish baseline
3. **Run Phase 2** to test quality impact
4. **Compare reports** and decide on best model
5. **Reassess all events** with chosen model
6. **Generate new clips** for WWF

Good luck! The temporal features should significantly improve bird and fish arrival detection!
