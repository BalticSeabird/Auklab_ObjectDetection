# Active Learning for Object Detection Improvement

This workflow helps you identify and annotate problematic frames to improve your object detection model, which will in turn improve event detection.

## 🚀 Quick Start (Large-Scale Pipeline)

**NEW**: For large datasets with random sampling, use the scalable pipeline:

```bash
# 1. Create your configuration
cp code/active_learning/config_example.yaml my_config.yaml
# Edit my_config.yaml with your paths

# 2. Run the full pipeline
python code/active_learning/run_active_learning_pipeline.py --config my_config.yaml
```

See **[SCALING_GUIDE.md](SCALING_GUIDE.md)** for complete documentation on the scalable pipeline.

---

## 🎯 Strategy

The core insight: **Many event detection failures stem from noisy object detections**. By identifying and retraining on problematic frames, you can:

1. ✅ Reduce edge detections (birds detected at frame borders)
2. ✅ Eliminate spike/dip artifacts (false positives/negatives)
3. ✅ Improve detection consistency in high-density scenes
4. ✅ Better capture count transitions (critical for event detection)

## 📋 Workflow

### Original Workflow (Small Scale, Hardcoded Paths)

For the original small-scale workflow, continue below. For **large-scale datasets with random sampling**, see the [Scalable Pipeline](#-quick-start-large-scale-pipeline) above.

### Step 1: Identify Problematic Frames

Run the frame identifier to analyze all detection CSVs:

```bash
./venv/bin/python code/active_learning/identify_problem_frames.py
```

This will:
- Analyze all CSV files in `csv_detection_1fps/`
- Identify frames with:
  - **Edge detections**: Birds detected near frame borders (often false positives)
  - **Spike frames**: Sudden count increases that disappear (detection errors)
  - **Dip frames**: Sudden count decreases that recover (missed detections)
  - **High count frames**: Scenes with many birds (harder to detect accurately)
  - **Count transitions**: Where bird count changes (critical for events)
  - **Fish detections**: Frames with fish present (class_id = 2)
- Save results to `data/problematic_frames.json`

**Expected output:**
```
Analyzing 31 CSV files...
================================================================================
[1/31] Processing TRI3_20250628T000000_raw... ✓ Found 45 problematic frames
[2/31] Processing TRI3_20250628T001002_raw... ✓ Found 23 problematic frames
...

SUMMARY STATISTICS
================================================================================
Files analyzed: 31
Files with problems: 28

Problem types:
  Edge detections: 547 frames
  Spike frames: 89 frames
  Dip frames: 124 frames
  High count frames: 67 frames
  Count transitions: 234 frames

Total problematic frames: 1061
```

### Step 2: Extract Frames for Annotation

Extract the most problematic frames from videos:

```bash
# Basic usage (extracts up to 100 frames per type, diverse sampling)
./venv/bin/python code/active_learning/extract_frames.py \
    --video-dir video \
    --output-dir data/frames_for_annotation \
    --max-per-type 100 \
    --priority diverse
```

**Options:**
- `--max-per-type`: Maximum frames to extract per problem type (default: 100)
- `--priority`:
  - `diverse`: Spread selection across different videos (recommended)
  - `concentrated`: Focus on worst videos first

**Output structure:**
```
data/frames_for_annotation/
├── annotation_manifest.json    # Complete metadata
├── frames_list.csv            # Simple list for quick reference
├── edge_detection/            # Frames with edge detections
│   ├── TRI3_20250628T030002_s0124_edge_detection.jpg
│   └── ...
├── spike/                     # Frames with spike artifacts
│   ├── TRI3_20250628T030002_s0037_spike.jpg
│   └── ...
├── dip/                       # Frames with dip artifacts
│   └── ...
├── high_count/                # Frames with many birds
│   └── ...
├── count_transition/          # Frames where count changes
│   └── ...
└── fish/                      # Frames with fish detections
    └── ...
```

### Step 3: Annotate the Frames

**Option A: Roboflow** (recommended - automated upload with pre-annotations)

Upload frames organized by problem type:

```bash
# Set your API key (once)
export ROBOFLOW_API_KEY="your_api_key_here"

# Upload all batches with pre-annotations
python code/active_learning/upload_batches_simple.py \
    data/frames_for_annotation \
    --with-annotations

# Or upload specific batches only (e.g., fish and edge detections)
python code/active_learning/upload_batches_simple.py \
    data/frames_for_annotation \
    --batches fish edge_detection
```

This will:
- Upload frames organized by problem type (edge_detection, spike, dip, high_count, count_transition, fish)
- Include pre-annotations from your model (faster than annotating from scratch)
- Create separate batches in Roboflow for each problem type
- Tag images by problem type for easy filtering

Then in Roboflow:
1. Review and correct the pre-annotations
2. Pay special attention to:
   - Edge detections: Partial birds at borders
   - High count frames: Overlapping birds
   - Fish frames: Fish vs bird classification
3. Export in YOLOv8 format when done

**Option B: CVAT / Label Studio**
**Option B: CVAT / Label Studio**
1. Manually upload frames from `data/frames_for_annotation/`
2. Import pre-annotations if your tool supports YOLO format
1. Import frames
2. Use bounding box tool to annotate birds
3. Export in YOLO format

**Annotation tips:**
- For edge detections: Pay special attention to partial birds at borders
- For high count frames: Be thorough - these are hardest cases
- For transitions: These frames are critical for event detection accuracy

### Step 4: Combine with Existing Dataset

Merge the new annotations with your existing training data:

```bash
# Assuming you exported from Roboflow/CVAT as 'active_learning_batch/'
cp -r active_learning_batch/images/* dataset/seabird_fish*/images/train/
cp -r active_learning_batch/labels/* dataset/seabird_fish*/labels/train/

# Update dataset YAML if needed
```

### Step 5: Retrain Object Detection Model

Use your existing training script with the augmented dataset:

```bash
./venv/bin/python code/model/train.py \
    --data dataset/your_updated_dataset.yaml \
    --epochs 50 \
    --batch 16 \
    --name model_with_active_learning
```

### Step 6: Evaluate Improvement

Run inference with new model and compare:

```bash
# Run new inference
./venv/bin/python code/model/run_inference.py \
    --model runs/detect/model_with_active_learning/weights/best.pt \
    --source video/

# Run event detection with new detections
./venv/bin/python code/postprocess/benchmark_state_based_detector.py
```

**Expected improvements:**
- Fewer edge detections → Less false positive events
- Fewer spikes/dips → Smoother count signals
- Better transition detection → Higher event detection accuracy
- Overall: Event detection F1 should increase from 61% to potentially 70%+

## 📊 Monitoring Impact

Track these metrics before and after retraining:

### Object Detection Level:
- Number of edge detections per video
- Spike/dip frequency
- Detection consistency (frame-to-frame)

### Event Detection Level:
- Overall F1-score
- Departure detection F1 (biggest problem in current model)
- False positive rate
- False negative rate

## 💡 Tips & Best Practices

1. **Start Small**: Begin with 50-100 frames per type to validate the workflow
2. **Focus on Failures**: Prioritize frame types causing the most event detection errors
3. **Iterate**: This is an iterative process - you may need 2-3 rounds
4. **Balance Dataset**: Don't oversample one frame type - maintain class balance
5. **Quality Over Quantity**: Better to annotate 100 frames well than 500 frames poorly

## 🔄 Iterative Improvement Loop

```
1. Current Model → 2. Identify Problems → 3. Extract Frames → 
4. Annotate → 5. Retrain → 6. Evaluate → (back to 1)
```

Each iteration should improve:
- Object detection quality
- Event detection accuracy
- Overall system reliability

## 📝 Expected Timeline

- **Analysis (Step 1)**: 5 minutes
- **Frame Extraction (Step 2)**: 10-15 minutes
- **Annotation (Step 3)**: 2-4 hours (for 500 frames)
- **Retraining (Step 5)**: 1-2 hours
- **Evaluation (Step 6)**: 30 minutes

**Total**: ~4-7 hours for one iteration

## 🎯 Success Criteria

You'll know the active learning worked when:
- ✅ Edge detection count drops by 50%+
- ✅ Spike/dip frequency reduces significantly
- ✅ Event detection F1 improves by 5-10 percentage points
- ✅ Departure detection F1 increases substantially (currently at 58.5%)
- ✅ False positive rate decreases

## 🆘 Troubleshooting

**Problem**: Can't find video files
- **Solution**: Check `--video-dir` path, ensure videos have expected extensions (.mp4, .avi, etc.)

**Problem**: Too many frames to annotate
- **Solution**: Reduce `--max-per-type` or use `--priority concentrated` to focus on worst cases

**Problem**: Extraction is slow
- **Solution**: Videos are large - expect ~1-2 min per video. Consider extracting from subset first.

## 📚 Related Files

- `identify_problem_frames.py`: Frame identification logic
- `extract_frames.py`: Frame extraction from videos
- `../postprocess/event_detector.py`: Event detection using these improved detections
- `../postprocess/state_based_detector.py`: State-based event detector that benefits from better detections

## 🚀 Future Enhancements

Potential improvements to this workflow:
- Uncertainty-based sampling (use model confidence scores)
- Hard negative mining (focus on false positives)
- Temporal consistency checks (multi-frame context)
- Automatic quality assessment of annotations
