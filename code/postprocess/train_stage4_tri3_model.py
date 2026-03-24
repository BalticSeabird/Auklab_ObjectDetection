#!/usr/bin/env python3
"""Train a lightweight Stage4 classifier for TRI3 fish-arrival validation.

This script uses manually validated labels from data/class_validation.csv and
linked Stage3 detection CSV files to train a logistic regression model with
simple numpy gradient descent. It also searches a compact threshold rule set as
an interpretable baseline.
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

from inference_system.stage4_modeling import MODEL_FEATURE_ORDER, extract_stage4_features, feature_vector_from_map


@dataclass
class DatasetBundle:
    frame: pd.DataFrame
    feature_names: List[str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train TRI3 Stage4 fish-arrival classifier")
    parser.add_argument("--validation-csv", type=Path, default=REPO_ROOT / "data" / "class_validation.csv")
    parser.add_argument("--station", type=str, default="TRI3")
    parser.add_argument("--target", type=str, default="valid_fish_arrival")
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--min-date", type=str, default="")
    parser.add_argument("--max-date", type=str, default="")
    parser.add_argument(
        "--output-model",
        type=Path,
        default=REPO_ROOT / "models" / "stage4" / "tri3_fish_arrival_model.json",
    )
    parser.add_argument(
        "--output-report",
        type=Path,
        default=REPO_ROOT / "data" / "stage4_tri3_training_report.json",
    )
    return parser.parse_args()


def load_validation_rows(args: argparse.Namespace) -> pd.DataFrame:
    frame = pd.read_csv(args.validation_csv, sep=";", dtype=str)
    frame.columns = [c.strip() for c in frame.columns]

    frame = frame[frame["station"].str.upper() == args.station.upper()].copy()
    frame = frame[frame["event_type"].fillna("").str.lower() == "arrival"].copy()

    for col in ["valid_arrival", "valid_fish", "valid_fish_arrival", "valid_multiple_fish"]:
        if col in frame.columns:
            frame[col] = pd.to_numeric(frame[col], errors="coerce")

    frame = frame[frame[args.target].notna()].copy()
    frame = frame[frame["valid_arrival"].notna()].copy()

    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    if args.min_date:
        frame = frame[frame["date"] >= pd.to_datetime(args.min_date)]
    if args.max_date:
        frame = frame[frame["date"] <= pd.to_datetime(args.max_date)]

    frame["arrival_with_fish_stage2"] = pd.to_numeric(frame["arrival_with_fish_stage2"], errors="coerce").fillna(0.0)
    frame = frame.sort_values(["date", "video_id", "second"], ascending=True)
    return frame


def extract_training_features(rows: pd.DataFrame) -> DatasetBundle:
    feature_rows: List[Dict[str, object]] = []

    for _, row in rows.iterrows():
        detections_csv = Path(str(row.get("detections_csv_path", "")))
        if not detections_csv.exists():
            continue

        try:
            detections = pd.read_csv(detections_csv)
        except Exception:
            continue

        stage2_flag = int(float(row.get("arrival_with_fish_stage2", 0) or 0))
        feats = extract_stage4_features(detections, stage2_flag=stage2_flag)

        feats["event_id"] = str(row.get("event_id", ""))
        feats["date"] = row.get("date")
        feats["video_id"] = str(row.get("video_id", ""))
        feats["target"] = int(float(row.get("valid_fish_arrival", 0) or 0))
        feats["valid_arrival"] = int(float(row.get("valid_arrival", 0) or 0))
        feats["valid_fish"] = int(float(row.get("valid_fish", 0) or 0))
        feats["valid_multiple_fish"] = int(float(row.get("valid_multiple_fish", 0) or 0)) if pd.notna(row.get("valid_multiple_fish")) else 0
        feature_rows.append(feats)

    dataset = pd.DataFrame(feature_rows)
    if dataset.empty:
        return DatasetBundle(frame=dataset, feature_names=list(MODEL_FEATURE_ORDER))

    dataset = dataset[dataset["valid_arrival"].isin([0, 1])].copy()
    dataset = dataset.sort_values(["date", "video_id", "event_id"]).reset_index(drop=True)
    return DatasetBundle(frame=dataset, feature_names=list(MODEL_FEATURE_ORDER))


def train_val_split(dataset: pd.DataFrame, train_ratio: float) -> Tuple[pd.DataFrame, pd.DataFrame]:
    split_idx = int(max(1, min(len(dataset) - 1, round(len(dataset) * train_ratio))))
    train_df = dataset.iloc[:split_idx].copy()
    val_df = dataset.iloc[split_idx:].copy()
    return train_df, val_df


def logistic_train(X: np.ndarray, y: np.ndarray, *, lr: float = 0.05, epochs: int = 6000, l2: float = 0.01) -> Tuple[np.ndarray, float]:
    w = np.zeros(X.shape[1], dtype=np.float64)
    b = 0.0

    for _ in range(epochs):
        logits = X @ w + b
        probs = 1.0 / (1.0 + np.exp(-np.clip(logits, -40.0, 40.0)))
        err = probs - y
        grad_w = (X.T @ err) / len(y) + l2 * w
        grad_b = float(err.mean())
        w -= lr * grad_w
        b -= lr * grad_b

    return w, b


def classify_probs(X: np.ndarray, w: np.ndarray, b: float) -> np.ndarray:
    logits = X @ w + b
    return 1.0 / (1.0 + np.exp(-np.clip(logits, -40.0, 40.0)))


def confusion_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2.0 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    accuracy = (tp + tn) / max(1, len(y_true))

    return {
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "accuracy": float(accuracy),
    }


def calibrate_threshold(y_true: np.ndarray, probs: np.ndarray) -> Tuple[float, Dict[str, float]]:
    best_threshold = 0.5
    best_metrics: Dict[str, float] = {"f1": -1.0}

    for threshold in np.linspace(0.1, 0.9, 81):
        pred = (probs >= threshold).astype(int)
        metrics = confusion_metrics(y_true, pred)
        if metrics["f1"] > best_metrics["f1"]:
            best_metrics = metrics
            best_threshold = float(threshold)

    return best_threshold, best_metrics


def evaluate_existing_rule(frame: pd.DataFrame) -> Dict[str, float]:
    if "is_new_fish_arrival" not in frame.columns:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "accuracy": 0.0, "tp": 0, "tn": 0, "fp": 0, "fn": 0}

    truth = frame["target"].astype(int).to_numpy()
    pred = pd.to_numeric(frame["is_new_fish_arrival"], errors="coerce").fillna(0).astype(int).to_numpy()
    return confusion_metrics(truth, pred)


def search_simple_rule(train_df: pd.DataFrame, val_df: pd.DataFrame) -> Dict[str, object]:
    best = {
        "f1": -1.0,
        "rule": None,
        "train_metrics": None,
        "val_metrics": None,
    }

    count_grid = [2, 3, 4, 6]
    first_grid = [0.05, 0.1, 0.15, 0.2, 0.25]
    dist_grid = [60.0, 80.0, 100.0, 140.0]
    conf_grid = [0.45, 0.55, 0.65, 0.75]

    for min_count in count_grid:
        for min_first in first_grid:
            for max_dist in dist_grid:
                for min_conf in conf_grid:
                    train_pred = (
                        (train_df["fish_detection_count"] >= min_count)
                        & (train_df["fish_first_frame_ratio"] >= min_first)
                        & (train_df["fish_bird_min_distance"] <= max_dist)
                        & (train_df["fish_avg_confidence"] >= min_conf)
                    ).astype(int)
                    train_metrics = confusion_metrics(train_df["target"].astype(int).to_numpy(), train_pred.to_numpy())

                    val_pred = (
                        (val_df["fish_detection_count"] >= min_count)
                        & (val_df["fish_first_frame_ratio"] >= min_first)
                        & (val_df["fish_bird_min_distance"] <= max_dist)
                        & (val_df["fish_avg_confidence"] >= min_conf)
                    ).astype(int)
                    val_metrics = confusion_metrics(val_df["target"].astype(int).to_numpy(), val_pred.to_numpy())

                    if val_metrics["f1"] > best["f1"]:
                        best = {
                            "f1": val_metrics["f1"],
                            "rule": {
                                "fish_detection_count_min": min_count,
                                "fish_first_frame_ratio_min": min_first,
                                "fish_bird_min_distance_max": max_dist,
                                "fish_avg_confidence_min": min_conf,
                            },
                            "train_metrics": train_metrics,
                            "val_metrics": val_metrics,
                        }

    return best


def main() -> None:
    args = parse_args()

    raw_rows = load_validation_rows(args)
    bundle = extract_training_features(raw_rows)
    dataset = bundle.frame
    if dataset.empty:
        raise RuntimeError("No training samples were built from validation rows and detection CSVs")

    train_df, val_df = train_val_split(dataset, args.train_ratio)
    if train_df.empty or val_df.empty:
        raise RuntimeError("Train/validation split failed; adjust --train-ratio or date filters")

    feature_names = bundle.feature_names
    X_train_raw = np.array([feature_vector_from_map(row, feature_names) for row in train_df.to_dict(orient="records")], dtype=np.float64)
    X_val_raw = np.array([feature_vector_from_map(row, feature_names) for row in val_df.to_dict(orient="records")], dtype=np.float64)
    y_train = train_df["target"].astype(int).to_numpy(dtype=np.float64)
    y_val = val_df["target"].astype(int).to_numpy(dtype=np.float64)

    means = X_train_raw.mean(axis=0)
    stds = X_train_raw.std(axis=0)
    stds[stds < 1e-6] = 1.0

    X_train = (X_train_raw - means) / stds
    X_val = (X_val_raw - means) / stds

    weights, bias = logistic_train(X_train, y_train)

    train_probs = classify_probs(X_train, weights, bias)
    val_probs = classify_probs(X_val, weights, bias)

    best_threshold, train_metrics = calibrate_threshold(y_train, train_probs)
    val_threshold_metrics = confusion_metrics(y_val.astype(int), (val_probs >= best_threshold).astype(int))
    val_best_threshold, val_best_metrics = calibrate_threshold(y_val, val_probs)

    rule_result = search_simple_rule(train_df, val_df)

    merged = dataset.merge(
        raw_rows[["event_id", "is_new_fish_arrival"]].copy(),
        on="event_id",
        how="left",
    )
    baseline_metrics = evaluate_existing_rule(merged)

    artifact = {
        "model_type": "logistic_regression",
        "target_label": args.target,
        "feature_names": feature_names,
        "means": means.tolist(),
        "stds": stds.tolist(),
        "weights": weights.tolist(),
        "bias": float(bias),
        "threshold": float(best_threshold),
        "metadata": {
            "station": args.station,
            "train_rows": int(len(train_df)),
            "val_rows": int(len(val_df)),
            "date_min": str(dataset["date"].min().date()),
            "date_max": str(dataset["date"].max().date()),
            "positive_rate_train": float(y_train.mean()),
            "positive_rate_val": float(y_val.mean()),
        },
    }

    report = {
        "dataset": {
            "station": args.station,
            "target": args.target,
            "rows_total": int(len(dataset)),
            "rows_train": int(len(train_df)),
            "rows_val": int(len(val_df)),
            "positives_total": int(dataset["target"].sum()),
            "negatives_total": int(len(dataset) - dataset["target"].sum()),
        },
        "baseline_stage4_rule_metrics": baseline_metrics,
        "logistic_model": {
            "threshold": float(best_threshold),
            "train_metrics": train_metrics,
            "val_metrics": val_threshold_metrics,
            "val_best_threshold": float(val_best_threshold),
            "val_best_metrics": val_best_metrics,
        },
        "simple_rule_search": rule_result,
        "model_output_path": str(args.output_model),
    }

    args.output_model.parent.mkdir(parents=True, exist_ok=True)
    args.output_report.parent.mkdir(parents=True, exist_ok=True)

    args.output_model.write_text(json.dumps(artifact, indent=2), encoding="utf-8")
    args.output_report.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
