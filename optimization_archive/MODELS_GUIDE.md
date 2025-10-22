# Models Directory Organization

## Production-Ready Files ⭐

### `auklab_model_xlarge_combined_4564_v1_clean.trt` (271.6 MB)
**Status**: PRODUCTION READY - USE THIS ENGINE
**Performance**: 94.7 images/second at batch size 8
**Features**: 
- Clean generation without PyTorch contamination
- Batch support: 1-16 (optimized for 8)
- Input: 960x960 images
- Fully validated and tested

### `auklab_model_xlarge_combined_4564_v1_batch_fixed.onnx`
**Status**: SOURCE MODEL for clean engine
**Purpose**: ONNX model used to generate the clean TensorRT engine
**Keep**: Required for future engine regeneration

## Legacy/Debug Files (Can be archived or removed)

### Corrupted/Old TensorRT Engines:
- `auklab_model_xlarge_combined_4564_v1.trt` - Original corrupted engine
- `auklab_model_xlarge_combined_4564_v1_batch.trt` - Failed batch attempt  
- `auklab_model_xlarge_combined_4564_v1_batch_fixed.trt` - Pre-clean version

### Original Source Files:
- `auklab_model_xlarge_combined_4564_v1.pt` - PyTorch model
- `auklab_model_xlarge_combined_4564_v1.onnx` - Original ONNX export
- `auklab_model_xlarge_combined_4564_v1_batch.onnx` - Batch ONNX attempt

## Other Model Files (Unrelated to current optimization)

### YOLO Base Models:
- `yolo11l.pt`, `yolo11m.pt`, `yolo11n.pt`, `yolo11x.pt` - Base YOLO11 models

### Auklab Trained Models:
- Various `auklab_model_*` files for different datasets and sizes
- These are unrelated to the current dual RTX 4090 optimization project

## File Size Analysis
```bash
ls -lh models/*.trt
# auklab_model_xlarge_combined_4564_v1.trt           118M  (corrupted)
# auklab_model_xlarge_combined_4564_v1_batch.trt     119M  (corrupted) 
# auklab_model_xlarge_combined_4564_v1_batch_fixed.trt 119M (corrupted)
# auklab_model_xlarge_combined_4564_v1_clean.trt     272M  (WORKING ⭐)
```

## Cleanup Recommendations

### Keep (Essential):
1. `auklab_model_xlarge_combined_4564_v1_clean.trt` - Production engine
2. `auklab_model_xlarge_combined_4564_v1_batch_fixed.onnx` - Source model
3. All unrelated Auklab and YOLO models

### Archive (Move to optimization_archive):
1. All corrupted .trt files (for debugging history)
2. Intermediate .onnx files from testing

### Command to clean up:
```bash
# Move old/corrupted engines to archive
mkdir -p optimization_archive/old_engines
mv models/auklab_model_xlarge_combined_4564_v1.trt optimization_archive/old_engines/
mv models/auklab_model_xlarge_combined_4564_v1_batch.trt optimization_archive/old_engines/
mv models/auklab_model_xlarge_combined_4564_v1_batch_fixed.trt optimization_archive/old_engines/
mv models/auklab_model_xlarge_combined_4564_v1_batch.onnx optimization_archive/old_engines/
```

## Production Usage
For future inference, always use:
```python
engine_path = "models/auklab_model_xlarge_combined_4564_v1_clean.trt"
```

This engine is guaranteed to work with the data type fixes and delivers optimal performance.