# Active Learning - Quick Reference

## Purpose
Improve object detection model by identifying and annotating problematic frames → Better event detection (target: 61% → 70%+ F1).

## Quick Workflow

### 1. Identify Problem Frames (~5 min)
```bash
python code/active_learning/identify_problem_frames.py
```
→ Creates `data/problematic_frames.json`

### 2. Extract Frames (~15 min)
```bash
# Start with 50 per type (recommended for first iteration)
python code/active_learning/extract_frames.py --max-per-type 50 --priority diverse

# Or more for comprehensive coverage
python code/active_learning/extract_frames.py --max-per-type 100 --priority diverse
```
→ Saves to `data/frames_for_annotation/` (organized by problem type)

**Requirements**: Videos must be in `video/` directory (.mkv format)

### 3. Pre-Annotate (~2-5 min)
```bash
python code/active_learning/pre_annotate_frames.py --confidence 0.25
```
→ Creates annotations: `data/frames_for_annotation/annotations/`
- `yolo/*.txt` - YOLO format
- `json/*.json` - Detailed JSON with confidence scores

### 4. Manual Correction (2-4 hours)
- Upload frames + pre-annotations to Roboflow/CVAT
- Review and correct errors (faster than annotating from scratch)
- Pay attention to: edge detections, overlapping birds, partial visibility
- Export corrected annotations

### 5. Retrain Model (4-8 hours)
- Combine corrected annotations with existing training data
- Retrain object detection model

### 6. Evaluate (~1-2 hours)
- Re-run inference with new model
- Re-run event detection
- Compare F1 scores

## Problem Types

| Type | Description | Why It Matters |
|------|-------------|----------------|
| **Edge detections** | Birds near frame borders | Often false positives causing arrival/departure errors |
| **Spike frames** | Sudden count increases | Detection artifacts that create false arrivals |
| **Dip frames** | Sudden count decreases | Missed detections that create false departures |
| **High count** | Scenes with 10+ birds | Challenging to detect all birds accurately |
| **Count transitions** | Where count changes | Critical moments for event detection |

## Success Metrics

- 50%+ reduction in edge detection artifacts
- Event detection F1: **61.2% → 70%+**
- Maintain/improve departure F1 (currently **58.5%**)

## Tips

- **Start small**: 50-100 frames validates workflow before scaling
- **Diverse sampling**: Better coverage across videos/conditions
- **Lower confidence**: 0.25 threshold catches more detections for review
- **Iterate**: Multiple rounds compound improvements

## Dependencies

```bash
# If you get NumPy errors:
pip install "numpy<2"  # PyTorch compatibility
```

## File Locations

```
data/problematic_frames.json              # Problem frame identification
data/frames_for_annotation/               # Extracted frames
  ├── edge_detection/
  ├── spike/
  ├── dip/
  ├── high_count/
  ├── count_transition/
  └── annotations/
      ├── yolo/                          # YOLO format (.txt)
      └── json/                          # Detailed JSON
```

## Notes

- Video filenames must match detection CSV basenames
- Default model: `models/auklab_model_xlarge_combined_4564_v1.pt`
- Can specify different model with `--model path/to/model.pt`
- Use `--help` on any script for full options
