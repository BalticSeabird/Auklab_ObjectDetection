"""Feature extraction and lightweight model inference for Stage4."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd


MODEL_FEATURE_ORDER = [
    "total_frames",
    "bird_frames",
    "fish_frames",
    "fish_detection_count",
    "fish_avg_confidence",
    "fish_confidence_std",
    "fish_presence_ratio",
    "fish_first_frame_ratio",
    "fish_last_frame_ratio",
    "fish_conf_late_minus_early",
    "fish_area_mean",
    "fish_area_trend",
    "bird_displacement",
    "bird_mean_motion",
    "bird_path_efficiency",
    "fish_bird_mean_distance",
    "fish_bird_min_distance",
    "fish_to_bird_first_frame_ratio_gap",
    "arrival_with_fish_stage2",
]


def _safe_float(value: object, default: float = 0.0) -> float:
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_int(value: object, default: int = 0) -> int:
    if value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


@dataclass
class Stage4ModelArtifact:
    """Serialized linear model for TRI3 fish-arrival scoring."""

    model_type: str
    target_label: str
    feature_names: List[str]
    means: List[float]
    stds: List[float]
    weights: List[float]
    bias: float
    threshold: float
    metadata: Dict[str, object]

    @classmethod
    def from_path(cls, path: Path) -> "Stage4ModelArtifact":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(
            model_type=str(payload.get("model_type", "logistic_regression")),
            target_label=str(payload.get("target_label", "valid_fish_arrival")),
            feature_names=[str(v) for v in payload.get("feature_names", [])],
            means=[float(v) for v in payload.get("means", [])],
            stds=[float(v) for v in payload.get("stds", [])],
            weights=[float(v) for v in payload.get("weights", [])],
            bias=float(payload.get("bias", 0.0)),
            threshold=float(payload.get("threshold", 0.5)),
            metadata=dict(payload.get("metadata", {})),
        )

    def predict_proba(self, feature_map: Dict[str, object]) -> float:
        if not self.feature_names:
            return 0.0
        z = self.bias
        for idx, feature_name in enumerate(self.feature_names):
            raw = _safe_float(feature_map.get(feature_name), 0.0)
            mean = self.means[idx] if idx < len(self.means) else 0.0
            std = self.stds[idx] if idx < len(self.stds) else 1.0
            if std <= 0.0:
                std = 1.0
            z += self.weights[idx] * ((raw - mean) / std)
        # Stable sigmoid.
        if z >= 0:
            exp_neg = math.exp(-z)
            return 1.0 / (1.0 + exp_neg)
        exp_pos = math.exp(z)
        return exp_pos / (1.0 + exp_pos)


def extract_stage4_features(detections: pd.DataFrame, *, stage2_flag: int = 0) -> Dict[str, object]:
    """Compute deterministic clip features used by Stage4 rules and model scoring."""
    if detections.empty:
        return {
            "total_frames": 0,
            "bird_frames": 0,
            "fish_frames": 0,
            "fish_detection_count": 0,
            "fish_avg_confidence": 0.0,
            "fish_confidence_std": 0.0,
            "fish_presence_ratio": 0.0,
            "fish_first_frame_ratio": 1.0,
            "fish_last_frame_ratio": 0.0,
            "fish_conf_late_minus_early": 0.0,
            "fish_area_mean": 0.0,
            "fish_area_trend": 0.0,
            "bird_displacement": 0.0,
            "bird_mean_motion": 0.0,
            "bird_path_efficiency": 0.0,
            "fish_bird_mean_distance": 9999.0,
            "fish_bird_min_distance": 9999.0,
            "fish_to_bird_first_frame_ratio_gap": 1.0,
            "arrival_with_fish_stage2": int(stage2_flag),
        }

    frame_max = int(detections["frame"].max()) if "frame" in detections.columns else -1
    total_frames = max(frame_max + 1, 0)

    classes = detections["class"].astype(str).str.lower()
    bird = detections[classes == "adult"].copy()
    fish = detections[classes == "fish"].copy()

    bird_by_frame = centroids_by_frame(bird)
    fish_by_frame = centroids_by_frame(fish)
    fish_frames = sorted(fish_by_frame.keys())

    bird_frames = sorted(bird_by_frame.keys())
    displacement, mean_motion, path_efficiency = motion_features(bird_by_frame)

    fish_detection_count = int(len(fish.index))
    fish_avg_confidence = 0.0
    fish_confidence_std = 0.0
    fish_conf_late_minus_early = 0.0
    fish_area_mean = 0.0
    fish_area_trend = 0.0

    if fish_detection_count > 0 and "confidence" in fish.columns:
        conf = pd.to_numeric(fish["confidence"], errors="coerce").fillna(0.0)
        fish_avg_confidence = float(conf.mean())
        fish_confidence_std = float(conf.std(ddof=0))

        half_frame = int(total_frames * 0.5)
        early = conf[fish["frame"] <= half_frame]
        late = conf[fish["frame"] > half_frame]
        early_mean = float(early.mean()) if not early.empty else fish_avg_confidence
        late_mean = float(late.mean()) if not late.empty else fish_avg_confidence
        fish_conf_late_minus_early = late_mean - early_mean

    if fish_detection_count > 0:
        area = (pd.to_numeric(fish["xmax"], errors="coerce") - pd.to_numeric(fish["xmin"], errors="coerce")) * (
            pd.to_numeric(fish["ymax"], errors="coerce") - pd.to_numeric(fish["ymin"], errors="coerce")
        )
        area = area.fillna(0.0)
        fish_area_mean = float(area.mean())

        frame_vals = pd.to_numeric(fish["frame"], errors="coerce").fillna(0.0)
        frame_centered = frame_vals - frame_vals.mean()
        denom = float((frame_centered * frame_centered).sum())
        if denom > 0.0:
            fish_area_trend = float((frame_centered * (area - area.mean())).sum() / denom)

    fish_first_ratio = 1.0
    fish_last_ratio = 0.0
    fish_presence_ratio = 0.0
    if total_frames > 0 and fish_frames:
        fish_first_ratio = float(fish_frames[0]) / float(total_frames)
        fish_last_ratio = float(fish_frames[-1]) / float(total_frames)
        fish_presence_ratio = float(len(fish_frames)) / float(total_frames)

    fish_bird_mean_distance, fish_bird_min_distance = fish_bird_distance_features(bird_by_frame, fish_by_frame)
    bird_first_ratio = 1.0
    if total_frames > 0 and bird_frames:
        bird_first_ratio = float(bird_frames[0]) / float(total_frames)

    return {
        "total_frames": total_frames,
        "bird_frames": len(bird_frames),
        "fish_frames": len(fish_frames),
        "fish_detection_count": fish_detection_count,
        "fish_avg_confidence": fish_avg_confidence,
        "fish_confidence_std": fish_confidence_std,
        "fish_presence_ratio": fish_presence_ratio,
        "fish_first_frame_ratio": fish_first_ratio,
        "fish_last_frame_ratio": fish_last_ratio,
        "fish_conf_late_minus_early": fish_conf_late_minus_early,
        "fish_area_mean": fish_area_mean,
        "fish_area_trend": fish_area_trend,
        "bird_displacement": displacement,
        "bird_mean_motion": mean_motion,
        "bird_path_efficiency": path_efficiency,
        "fish_bird_mean_distance": fish_bird_mean_distance,
        "fish_bird_min_distance": fish_bird_min_distance,
        "fish_to_bird_first_frame_ratio_gap": fish_first_ratio - bird_first_ratio,
        "arrival_with_fish_stage2": int(stage2_flag),
    }


def centroids_by_frame(df: pd.DataFrame) -> Dict[int, Tuple[float, float]]:
    if df.empty:
        return {}
    by_frame: Dict[int, List[Tuple[float, float]]] = {}
    for _, row in df.iterrows():
        frame = int(row["frame"])
        cx = (float(row["xmin"]) + float(row["xmax"])) / 2.0
        cy = (float(row["ymin"]) + float(row["ymax"])) / 2.0
        by_frame.setdefault(frame, []).append((cx, cy))

    centroids: Dict[int, Tuple[float, float]] = {}
    for frame, points in by_frame.items():
        x_mean = sum(p[0] for p in points) / len(points)
        y_mean = sum(p[1] for p in points) / len(points)
        centroids[frame] = (x_mean, y_mean)
    return centroids


def motion_features(centroids: Dict[int, Tuple[float, float]]) -> Tuple[float, float, float]:
    if len(centroids) < 2:
        return 0.0, 0.0, 0.0
    ordered = sorted(centroids.items(), key=lambda kv: kv[0])
    first = ordered[0][1]
    last = ordered[-1][1]
    dx = last[0] - first[0]
    dy = last[1] - first[1]
    displacement = (dx * dx + dy * dy) ** 0.5

    deltas: List[float] = []
    prev = ordered[0][1]
    for _, cur in ordered[1:]:
        ddx = cur[0] - prev[0]
        ddy = cur[1] - prev[1]
        deltas.append((ddx * ddx + ddy * ddy) ** 0.5)
        prev = cur
    path_length = float(sum(deltas))
    mean_motion = path_length / len(deltas) if deltas else 0.0
    path_efficiency = displacement / path_length if path_length > 0 else 0.0
    return displacement, mean_motion, path_efficiency


def fish_bird_distance_features(
    bird_by_frame: Dict[int, Tuple[float, float]],
    fish_by_frame: Dict[int, Tuple[float, float]],
) -> Tuple[float, float]:
    overlap = sorted(set(bird_by_frame.keys()) & set(fish_by_frame.keys()))
    if not overlap:
        return 9999.0, 9999.0

    distances: List[float] = []
    for frame in overlap:
        bird_c = bird_by_frame[frame]
        fish_c = fish_by_frame[frame]
        dx = bird_c[0] - fish_c[0]
        dy = bird_c[1] - fish_c[1]
        distances.append((dx * dx + dy * dy) ** 0.5)

    return float(sum(distances) / len(distances)), float(min(distances))


def feature_vector_from_map(feature_map: Dict[str, object], feature_names: Optional[List[str]] = None) -> List[float]:
    names = feature_names or MODEL_FEATURE_ORDER
    return [_safe_float(feature_map.get(name), 0.0) for name in names]


def parse_optional_int(value: object, default: int = 0) -> int:
    return _safe_int(value, default)
