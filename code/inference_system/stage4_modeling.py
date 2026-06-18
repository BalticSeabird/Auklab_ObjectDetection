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
    # Improved temporal features for bird arrival detection
    "bird_appears_mid_clip",
    "bird_count_peak",
    "bird_arrival_timing_ratio",
    # Improved temporal features for fish arrival behavior
    "fish_count_increases",
    "fish_count_peak",
    "fish_arrival_timing_ratio",
    "fish_deceleration",
    "fish_movement_distance",
    "fish_bird_convergence_rate",
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


def count_detections_in_phase(df: pd.DataFrame, total_frames: int, phase_start_ratio: float, phase_end_ratio: float) -> int:
    """Count unique detections in a time phase.

    Args:
        df: DataFrame with detections
        total_frames: Total number of frames
        phase_start_ratio: Start of phase as ratio (0-1)
        phase_end_ratio: End of phase as ratio (0-1)

    Returns:
        Number of detections in that phase
    """
    if df.empty or total_frames <= 0:
        return 0

    frame_start = int(total_frames * phase_start_ratio)
    frame_end = int(total_frames * phase_end_ratio)

    phase_detections = df[(df["frame"] >= frame_start) & (df["frame"] < frame_end)]
    return len(phase_detections)


def get_first_appearance_ratio(df: pd.DataFrame, total_frames: int) -> float:
    """Get when an object first appears as a ratio (0-1).

    Returns:
        Ratio of frame where object first appears (0=start, 1=end)
    """
    if df.empty or total_frames <= 0:
        return 1.0  # Default to end if no detections

    first_frame = int(df["frame"].min())
    return float(first_frame) / float(total_frames)


def object_appears_mid_clip(df: pd.DataFrame, total_frames: int, mid_start_ratio: float = 0.3, mid_end_ratio: float = 0.7) -> int:
    """Check if object first appears in the middle of the clip (not at start/end).

    Args:
        df: DataFrame with detections
        total_frames: Total number of frames
        mid_start_ratio: Start of "middle" phase (default 30%)
        mid_end_ratio: End of "middle" phase (default 70%)

    Returns:
        1 if first appearance is in middle range, 0 otherwise
    """
    if df.empty or total_frames <= 0:
        return 0

    first_frame = int(df["frame"].min())
    appearance_ratio = float(first_frame) / float(total_frames) if total_frames > 0 else 1.0

    # Check if appearance is in middle range (true arrival behavior)
    return 1 if (appearance_ratio >= mid_start_ratio and appearance_ratio <= mid_end_ratio) else 0


def get_count_timeline(df: pd.DataFrame, total_frames: int, num_phases: int = 3) -> List[int]:
    """Get detection count in each phase to track when counts peak.

    Divides clip into equal phases and returns count for each.
    """
    if df.empty or total_frames <= 0:
        return [0] * num_phases

    counts = []
    phase_width = 1.0 / num_phases
    for i in range(num_phases):
        start_ratio = i * phase_width
        end_ratio = (i + 1) * phase_width
        count = count_detections_in_phase(df, total_frames, start_ratio, end_ratio)
        counts.append(count)

    return counts


def count_increases_in_timeline(timeline: List[int]) -> int:
    """Check if count increases at any point in timeline.

    Returns 1 if any adjacent phases show an increase (e.g., 1→2 or 0→1)
    """
    for i in range(len(timeline) - 1):
        if timeline[i + 1] > timeline[i]:
            return 1
    return 0


def get_peak_timing_ratio(df: pd.DataFrame, total_frames: int, num_phases: int = 3) -> float:
    """Get when peak count occurs as a ratio (0=start, 1=end).

    Returns:
        Ratio indicating where in the clip the peak occurred
    """
    timeline = get_count_timeline(df, total_frames, num_phases)

    if not timeline or max(timeline) == 0:
        return 0.5  # Default to middle if no detections

    peak_phase = timeline.index(max(timeline))
    phase_width = 1.0 / num_phases

    # Return midpoint of peak phase
    peak_ratio = (peak_phase + 0.5) * phase_width
    return peak_ratio


def calculate_motion_in_phase(
    centroids: Dict[int, Tuple[float, float]],
    total_frames: int,
    phase_start_ratio: float,
    phase_end_ratio: float
) -> float:
    """Calculate average motion speed in a time phase.

    Returns:
        Average frame-to-frame distance in that phase
    """
    if not centroids or total_frames <= 0:
        return 0.0

    frame_start = int(total_frames * phase_start_ratio)
    frame_end = int(total_frames * phase_end_ratio)

    phase_frames = sorted([f for f in centroids.keys() if frame_start <= f < frame_end])
    if len(phase_frames) < 2:
        return 0.0

    deltas: List[float] = []
    for i in range(len(phase_frames) - 1):
        f1, f2 = phase_frames[i], phase_frames[i + 1]
        c1, c2 = centroids[f1], centroids[f2]
        dx = c2[0] - c1[0]
        dy = c2[1] - c1[1]
        deltas.append((dx * dx + dy * dy) ** 0.5)

    return float(sum(deltas) / len(deltas)) if deltas else 0.0


def calculate_object_distance(
    centroids: Dict[int, Tuple[float, float]],
    total_frames: int,
    phase_start_ratio: float,
    phase_end_ratio: float
) -> float:
    """Calculate distance traveled by object in a phase (first to last position).

    Returns:
        Euclidean distance from first to last position
    """
    if not centroids or total_frames <= 0:
        return 0.0

    frame_start = int(total_frames * phase_start_ratio)
    frame_end = int(total_frames * phase_end_ratio)

    phase_frames = sorted([f for f in centroids.keys() if frame_start <= f < frame_end])
    if len(phase_frames) < 2:
        return 0.0

    first_pos = centroids[phase_frames[0]]
    last_pos = centroids[phase_frames[-1]]

    dx = last_pos[0] - first_pos[0]
    dy = last_pos[1] - first_pos[1]
    return (dx * dx + dy * dy) ** 0.5


def calculate_distance_trend(
    bird_centroids: Dict[int, Tuple[float, float]],
    fish_centroids: Dict[int, Tuple[float, float]],
    total_frames: int
) -> float:
    """Calculate if fish and bird are converging (distance decreasing).

    Returns:
        early_avg_distance - late_avg_distance (positive = converging)
    """
    if not bird_centroids or not fish_centroids or total_frames <= 0:
        return 0.0

    # Early phase (0-50%)
    early_distance = calculate_mean_distance_between(bird_centroids, fish_centroids, total_frames, 0.0, 0.5)
    # Late phase (50-100%)
    late_distance = calculate_mean_distance_between(bird_centroids, fish_centroids, total_frames, 0.5, 1.0)

    # Positive value = they were far but got closer (converging)
    if early_distance == 9999.0 or late_distance == 9999.0:
        return 0.0

    return early_distance - late_distance


def calculate_mean_distance_between(
    bird_centroids: Dict[int, Tuple[float, float]],
    fish_centroids: Dict[int, Tuple[float, float]],
    total_frames: int,
    phase_start_ratio: float,
    phase_end_ratio: float
) -> float:
    """Calculate average distance between bird and fish in a phase."""
    frame_start = int(total_frames * phase_start_ratio)
    frame_end = int(total_frames * phase_end_ratio)

    overlap = sorted([f for f in set(bird_centroids.keys()) & set(fish_centroids.keys())
                     if frame_start <= f < frame_end])
    if not overlap:
        return 9999.0

    distances: List[float] = []
    for frame in overlap:
        bird_c = bird_centroids[frame]
        fish_c = fish_centroids[frame]
        dx = bird_c[0] - fish_c[0]
        dy = bird_c[1] - fish_c[1]
        distances.append((dx * dx + dy * dy) ** 0.5)

    return float(sum(distances) / len(distances)) if distances else 9999.0


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
            "bird_appears_mid_clip": 0,
            "bird_count_peak": 0,
            "bird_arrival_timing_ratio": 0.5,
            "fish_count_increases": 0,
            "fish_count_peak": 0,
            "fish_arrival_timing_ratio": 0.5,
            "fish_deceleration": 0.0,
            "fish_movement_distance": 0.0,
            "fish_bird_convergence_rate": 0.0,
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

    # Calculate improved temporal features for bird arrival (when does bird appear?)
    bird_appears_mid_clip = object_appears_mid_clip(bird, total_frames, mid_start_ratio=0.3, mid_end_ratio=0.7)
    bird_timeline = get_count_timeline(bird, total_frames, num_phases=3)
    bird_count_peak = max(bird_timeline) if bird_timeline else 0
    bird_arrival_timing_ratio = get_peak_timing_ratio(bird, total_frames, num_phases=3)

    # Calculate improved temporal features for fish arrival (when does fish appear/increase?)
    fish_timeline = get_count_timeline(fish, total_frames, num_phases=3)
    fish_count_increases = count_increases_in_timeline(fish_timeline)
    fish_count_peak = max(fish_timeline) if fish_timeline else 0
    fish_arrival_timing_ratio = get_peak_timing_ratio(fish, total_frames, num_phases=3)

    # Fish deceleration: early motion - late motion (positive = slowed down)
    fish_early_motion = calculate_motion_in_phase(fish_by_frame, total_frames, 0.0, 0.5)
    fish_late_motion = calculate_motion_in_phase(fish_by_frame, total_frames, 0.5, 1.0)
    fish_deceleration = fish_early_motion - fish_late_motion

    # Fish movement distance: how far the fish traveled overall
    fish_movement_distance = calculate_object_distance(fish_by_frame, total_frames, 0.0, 1.0)

    # Fish-bird convergence: are they getting closer?
    fish_bird_convergence_rate = calculate_distance_trend(bird_by_frame, fish_by_frame, total_frames)

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
        "bird_appears_mid_clip": bird_appears_mid_clip,
        "bird_count_peak": bird_count_peak,
        "bird_arrival_timing_ratio": bird_arrival_timing_ratio,
        "fish_count_increases": fish_count_increases,
        "fish_count_peak": fish_count_peak,
        "fish_arrival_timing_ratio": fish_arrival_timing_ratio,
        "fish_deceleration": fish_deceleration,
        "fish_movement_distance": fish_movement_distance,
        "fish_bird_convergence_rate": fish_bird_convergence_rate,
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
