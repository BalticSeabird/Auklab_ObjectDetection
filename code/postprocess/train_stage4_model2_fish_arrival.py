#!/usr/bin/env python3
"""
Model 2: Fish Arrival Detector

Predicts if there's fish arrival at a confirmed bird arrival.
Only used when Model 1 (bird arrival) returns positive.

Features:
- fish_count_increases: Did fish count go up? (0 or 1)
- fish_count_peak: Max fish count (0+)
- fish_arrival_timing_ratio: When peak occurred (0-1)
- fish_deceleration: Early motion - late motion (positive = slowed down)
- fish_movement_distance: Total distance traveled
- fish_bird_convergence_rate: Getting closer to bird?
- total_frames: Clip length
- fish_frames: Frames with fish detections
- arrival_with_fish_stage2: Stage2 flag

Target: is_new_fish_arrival (1=fish arrival, 0=no fish)

Usage:
    python code/postprocess/train_stage4_model2_fish_arrival.py \
        --validation-csv data/class_validation_merged.csv \
        --station TRI3 \
        --output-model models/stage4/model2_fish_arrival.json \
        --output-report data/stage4_model2_report.json
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

from inference_system.stage4_modeling import extract_stage4_features


# Features for Model 2: Fish Arrival Detection
MODEL2_FEATURES = [
    "fish_count_increases",
    "fish_count_peak",
    "fish_arrival_timing_ratio",
    "fish_deceleration",
    "fish_movement_distance",
    "fish_bird_convergence_rate",
    "total_frames",
    "fish_frames",
    "arrival_with_fish_stage2",
]


@dataclass
class Stage4Model:
    """Serialized linear model for fish arrival detection."""

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
    """Load and filter validation data for model 2 (fish arrival)."""
    df = pd.read_csv(csv_path, sep=";", dtype=str)
    df.columns = [c.strip() for c in df.columns]

    # Filter by station and event type
    df = df[df["station"].str.upper() == station.upper()].copy()
    df = df[df["event_type"].fillna("").str.lower() == "arrival"].copy()

    # Convert targets (for model 2, only use records where valid_arrival=1)
    df["valid_arrival"] = pd.to_numeric(df["valid_arrival"], errors="coerce")
    df["valid_fish_arrival"] = pd.to_numeric(df["valid_fish_arrival"], errors="coerce")

    # Only use records that are valid arrivals
    df = df[df["valid_arrival"].notna()].copy()
    df = df[df["valid_arrival"] == 1].copy()  # Only actual arrivals

    # Must have fish label
    df = df[df["valid_fish_arrival"].notna()].copy()

    print(f"Loaded {len(df)} confirmed arrival events from {station}")
    print(f"  With fish (true arrival): {int(df['valid_fish_arrival'].sum())}")
    print(f"  Without fish (false arrival): {len(df) - int(df['valid_fish_arrival'].sum())}")

    return df


def load_features(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """Extract features and labels for model 2."""
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
            stage2_flag = int(row.get("arrival_with_fish_stage2", 0))
            features = extract_stage4_features(detections, stage2_flag=stage2_flag)

            # Extract only model2 features
            feature_vector = [float(features.get(fname, 0.0)) for fname in MODEL2_FEATURES]
            X_list.append(feature_vector)
            y_list.append(int(row["valid_fish_arrival"]))
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
    parser = argparse.ArgumentParser(description="Train Model 2: Fish Arrival Detector")
    parser.add_argument("--validation-csv", type=Path, default=Path("data/class_validation_merged.csv"))
    parser.add_argument("--station", type=str, default="TRI3")
    parser.add_argument("--output-model", type=Path, default=Path("models/stage4/model2_fish_arrival.json"))
    parser.add_argument("--output-report", type=Path, default=Path("data/stage4_model2_report.json"))

    args = parser.parse_args()
    args.output_model.parent.mkdir(parents=True, exist_ok=True)
    args.output_report.parent.mkdir(parents=True, exist_ok=True)

    print(f"=" * 70)
    print(f"MODEL 2: FISH ARRIVAL DETECTOR")
    print(f"=" * 70)

    # Load data
    df = load_validation_rows(args.validation_csv, args.station)
    if len(df) < 5:
        print("ERROR: Not enough labeled data")
        return 1

    # Extract features
    X, y = load_features(df)
    if len(X) < 3:
        print("ERROR: Not enough feature vectors")
        return 1

    # Train
    weights, means, stds, bias = train_logistic_regression(X, y)

    # Create model
    model = Stage4Model(
        model_type="logistic_regression",
        target_label="is_new_fish_arrival",
        feature_names=MODEL2_FEATURES,
        means=means.tolist(),
        stds=stds.tolist(),
        weights=weights.tolist(),
        bias=float(bias),
        threshold=0.5,
        metadata={
            "station": args.station,
            "feature_count": len(MODEL2_FEATURES),
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
