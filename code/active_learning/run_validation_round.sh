#!/usr/bin/env bash
set -euo pipefail

# End-to-end validation round:
# 1) sample + pre-annotate frames
# 2) keep only images with pre-annotated objects
# 3) upload to Roboflow validation split
# 4) download labeled version for retraining

# Required env vars:
#   export ROBOFLOW_API_KEY="..."

CONFIG_PATH="config/config_active_learning_validation_smoke.yaml"
WORKSPACE="ai-course-2024"
PROJECT="fish_seabirds_combined-625bd"
BATCH_PREFIX="active_learning_validation"

# Set the Roboflow version once labels have been reviewed and generated.
ROBOFLOW_VERSION="11"
DOWNLOAD_FORMAT="yolov11"

OUTPUT_DIR="data/active_learning_validation_smoke"
FRAMES_DIR="${OUTPUT_DIR}/frames"
POSITIVE_DIR="${OUTPUT_DIR}/frames_positive"

if [[ -z "${ROBOFLOW_API_KEY:-}" ]]; then
  echo "ROBOFLOW_API_KEY is not set. Export it before running this script."
  exit 1
fi

echo "[1/5] Running active-learning sample+annotate pipeline"
python code/active_learning/run_active_learning_pipeline.py \
  --config "${CONFIG_PATH}" \
  --steps identify extract annotate

echo "[2/5] Selecting only frames with non-empty pre-annotations"
python code/active_learning/select_positive_preannotations.py \
  --frames-dir "${FRAMES_DIR}" \
  --output-dir "${POSITIVE_DIR}"

echo "[3/5] Uploading selected frames to Roboflow (valid split)"
python code/active_learning/upload_to_roboflow.py \
  --frames-dir "${POSITIVE_DIR}" \
  --use-annotations \
  --split valid \
  --workspace "${WORKSPACE}" \
  --project "${PROJECT}" \
  --api-key "${ROBOFLOW_API_KEY}" \
  --batch-name-prefix "${BATCH_PREFIX}"

echo "[4/5] Wait for annotation review in Roboflow, then set ROBOFLOW_VERSION in this script"

echo "[5/5] Downloading reviewed dataset version from Roboflow"
python code/dataset/roboflow_download.py \
  --workspace "${WORKSPACE}" \
  --project "${PROJECT}" \
  --version "${ROBOFLOW_VERSION}" \
  --format "${DOWNLOAD_FORMAT}"

echo "Validation round complete."
