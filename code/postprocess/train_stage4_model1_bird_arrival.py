#!/usr/bin/env python3
"""
Model 1: Bird Arrival Detector

Predicts if there's a TRUE bird arrival (not just existing bird movement).

Features:
- bird_appears_mid_clip: Did bird appear between 30-70%? (0 or 1)
- bird_count_peak: Maximum birds detected (0+)
- bird_arrival_timing_ratio: When peak occurred (0-1)
- bird_displacement: Total distance bird moved
- bird_mean_motion: Average frame-to-frame motion
- bird_path_efficiency: Displacement vs total path
- total_frames: Clip length
- bird_frames: Frames with bird detections

Target: is_actual_arrival (1=true arrival, 0=no arrival/false alarm)

Usage:
    python code/postprocess/train_stage4_model1_bird_arrival.py \
        --validation-csv data/class_validation_merged.csv \
        --station TRI3 \
        --output-model models/stage4/model1_bird_arrival.json \
        --output-report data/stage4_model1_report.json
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
CODE_ROOT = REPO_ROOT / "code"
import sys

if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from inference_system.stage4_modeling import MODEL_FEATURE_ORDER, extract_stage4_features


# Features for Model 1: Bird Arrival Detection
MODEL1_FEATURES = [
    "bird_appears_mid_clip",
    "bird_count_peak",
    "bird_arrival_timing_ratio",
    "bird_displacement",
    "bird_mean_motion",
    "bird_path_efficiency",
    "total_frames",
    "bird_frames",
]


@dataclass
class Stage4Model:
    """Serialized linear model for bird arrival detection."""

    model_type: str
    target_label: str
    feature_names: List[str]
    means: List[float]
    stds: List[float]
    weights: List[float]
    bias: float
    threshold: float
    metadata: Dict[str, object]

    def to_dict(self) -> dict:
        return {
            "model_type": self.model_type,
            "target_label": self.target_label,
            "feature_names": self.feature_names,
            "means": self.means,
            "stds": self.stds,
            "weights": self.weights,
            "bias": self.bias,
            "threshold": self.threshold,
            "metadata": self.metadata,
        }

    def predict_proba(self, feature_map: Dict[str, object]) -> float:
        """Predict probability using logistic regression."""
        if not self.feature_names:
            return 0.0

        z = self.bias
        for idx, fname in enumerate(self.feature_names):
            raw = float(feature_map.get(fname, 0.0))
            mean = self.means[idx] if idx < len(self.means) else 0.0
            std = self.stds[idx] if idx < len(self.stds) else 1.0
            if std <= 0.0:
                std = 1.0
            z += self.weights[idx] * ((raw - mean) / std)

        # Sigmoid
        import math
        if z >= 0:
            exp_neg = math.exp(-z)
            return 1.0 / (1.0 + exp_neg)
        exp_pos = math.exp(z)
        return exp_pos / (1.0 + exp_pos)


def load_validation_rows(csv_path: Path, station: str) -> pd.DataFrame:
    """Load and filter validation data for model 1 (bird arrival)."""
    df = pd.read_csv(csv_path, sep=";", dtype=str)
    df.columns = [c.strip() for c in df.columns]

    # Filter by station and event type
    df = df[df["station"].str.upper() == station.upper()].copy()
    df = df[df["event_type"].fillna("").str.lower() == "arrival"].copy()

    # Convert target
    df["valid_arrival"] = pd.to_numeric(df["valid_arrival"], errors="coerce")
    df = df[df["valid_arrival"].notna()].copy()

    print(f"Loaded {len(df)} arrival events from {station}")
    print(f"  Positive (true arrival): {int(df['valid_arrival'].sum())}")
    print(f"  Negative (no arrival): {len(df) - int(df['valid_arrival'].sum())}")

    return df


def load_features(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """Extract features and labels for model 1."""
    from inference_system.stage4_modeling import extract_stage4_features

    X_list = []
    y_list = []
    failed = 0

    for idx, row in df.iterrows():
        detections_csv = row.get("detections_csv_path")
        if not detections_csv or not Path(detections_csv).exists():
            failed += 1
            continue

        try:
            detections = pd.read_csv(detections_csv)
            features = extract_stage4_features(detections, stage2_flag=0)

            # Extract only model1 features
            feature_vector = [float(features.get(fname, 0.0)) for fname in MODEL1_FEATURES]
            X_list.append(feature_vector)
            y_list.append(int(row["valid_arrival"]))
        except Exception as e:
            failed += 1
            continue

    print(f"Extracted {len(X_list)} feature vectors ({failed} failed)")
    return np.array(X_list), np.array(y_list)


def train_logistic_regression(X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Train logistic regression using sklearn."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

    # Scale
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train_scaled, y_train)

    # Evaluate
    train_acc = model.score(X_train_scaled, y_train)
    test_acc = model.score(X_test_scaled, y_test)

    print(f"\nModel Performance:")
    print(f"  Train accuracy: {train_acc:.3f}")
    print(f"  Test accuracy: {test_acc:.3f}")

    return model.coef_[0], scaler.mean_, scaler.scale_, model.intercept_[0]


def main() -> int:
    parser = argparse.ArgumentParser(description="Train Model 1: Bird Arrival Detector")
    parser.add_argument("--validation-csv", type=Path, default=Path("data/class_validation_merged.csv"))
    parser.add_argument("--station", type=str, default="TRI3")
    parser.add_argument("--output-model", type=Path, default=Path("models/stage4/model1_bird_arrival.json"))
    parser.add_argument("--output-report", type=Path, default=Path("data/stage4_model1_report.json"))

    args = parser.parse_args()
    args.output_model.parent.mkdir(parents=True, exist_ok=True)
    args.output_report.parent.mkdir(parents=True, exist_ok=True)

    print(f"=" * 70)
    print(f"MODEL 1: BIRD ARRIVAL DETECTOR")
    print(f"=" * 70)

    # Load data
    df = load_validation_rows(args.validation_csv, args.station)
    if len(df) < 10:
        print("ERROR: Not enough labeled data")
        return 1

    # Extract features
    X, y = load_features(df)
    if len(X) < 5:
        print("ERROR: Not enough feature vectors")
        return 1

    # Train
    weights, means, stds, bias = train_logistic_regression(X, y)

    # Create model
    model = Stage4Model(
        model_type="logistic_regression",
        target_label="is_actual_arrival",
        feature_names=MODEL1_FEATURES,
        means=means.tolist(),
        stds=stds.tolist(),
        weights=weights.tolist(),
        bias=float(bias),
        threshold=0.5,
        metadata={
            "station": args.station,
            "feature_count": len(MODEL1_FEATURES),
            "training_samples": len(X),
            "positive_samples": int(y.sum()),
            "negative_samples": len(y) - int(y.sum()),
        },
    )

    # Save model
    with open(args.output_model, "w") as f:
        json.dump(model.to_dict(), f, indent=2)
    print(f"\nSaved model: {args.output_model}")

    # Save report
    with open(args.output_report, "w") as f:
        json.dump(model.metadata, f, indent=2)
    print(f"Saved report: {args.output_report}")

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
