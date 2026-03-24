#!/usr/bin/env python3
"""
Sample frames from original videos using indexed stage3 events.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import cv2
import numpy as np
import yaml


@dataclass
class SelectedEvent:
    event: Dict
    rank: float


def load_config(path: Path) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_index(index_path: Path) -> List[Dict]:
    with open(index_path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    if "events" in payload:
        return payload.get("events", [])
    # Backward compatibility with old index format.
    return payload.get("clips", [])


def _event_rank(event: Dict) -> float:
    # Prefer stronger fish signals when available, then fall back to event time.
    fish_count = int(event.get("fish_count", 0) or 0)
    second = int(event.get("event_second", event.get("second", 0)) or 0)
    return float(fish_count * 100000 + second)


def _select_max_events(cfg: Dict) -> int:
    sampling_cfg = cfg.get("sampling", {})
    if sampling_cfg.get("max_events") is not None:
        return int(sampling_cfg["max_events"])

    legacy = sampling_cfg.get("max_clips_per_category")
    if isinstance(legacy, dict):
        return int(sum(int(v) for v in legacy.values()))

    return 100


def _brightness_blur_probe(video_path: Path, target_second: float) -> Tuple[float, float]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return 0.0, 0.0

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    center_idx = int(max(target_second, 0.0) * fps)
    probe_idxs = [max(center_idx - int(fps), 0), center_idx, min(center_idx + int(fps), max(frame_count - 1, 0))]

    brightness_vals: List[float] = []
    blur_vals: List[float] = []

    for idx in probe_idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok or frame is None:
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        brightness_vals.append(float(np.mean(gray)))
        blur_vals.append(float(cv2.Laplacian(gray, cv2.CV_64F).var()))

    cap.release()

    if not brightness_vals:
        return 0.0, 0.0
    return float(np.mean(brightness_vals)), float(np.mean(blur_vals))


def score_low_quality(event: Dict) -> float:
    video_path = Path(event["original_video_path"])
    event_second = float(event.get("event_second", event.get("second", 0)) or 0.0)
    brightness, blur = _brightness_blur_probe(video_path, event_second)

    # Lower brightness and lower blur variance indicate difficult visibility.
    darkness = max(0.0, (60.0 - brightness) / 60.0)
    blur_penalty = max(0.0, (100.0 - blur) / 100.0)
    return darkness + blur_penalty


def select_events(index_rows: List[Dict], cfg: Dict) -> List[SelectedEvent]:
    seed = int(cfg.get("sampling", {}).get("random_seed", 42))
    random.seed(seed)

    ranked = [SelectedEvent(event=row, rank=_event_rank(row) + score_low_quality(row)) for row in index_rows if row.get("original_video_path")]
    if not ranked:
        return []

    ranked.sort(key=lambda x: x.rank, reverse=True)
    max_events = _select_max_events(cfg)
    enforce_diversity = bool(cfg.get("sampling", {}).get("enforce_diversity", True))

    if not enforce_diversity:
        return ranked[:max_events]

    taken: List[SelectedEvent] = []
    seen_station_date = set()
    for item in ranked:
        key = (item.event.get("station"), item.event.get("date"))
        if key not in seen_station_date:
            taken.append(item)
            seen_station_date.add(key)
        if len(taken) >= max_events:
            return taken

    if len(taken) < max_events:
        remaining = [r for r in ranked if r not in taken]
        taken.extend(remaining[: max_events - len(taken)])

    return taken


def _offsets_seconds(cfg: Dict) -> List[float]:
    sampling_cfg = cfg.get("sampling", {})
    if sampling_cfg.get("event_time_offsets_seconds"):
        return [float(x) for x in sampling_cfg.get("event_time_offsets_seconds", [])]

    frames_per_event = int(sampling_cfg.get("frames_per_event", sampling_cfg.get("frames_per_clip", 2)))
    before_sec = float(sampling_cfg.get("event_window_before_sec", 1.0))
    after_sec = float(sampling_cfg.get("event_window_after_sec", 1.0))

    if frames_per_event <= 1:
        return [0.0]

    return np.linspace(-before_sec, after_sec, frames_per_event, dtype=float).tolist()


def extract_frames(selected: List[SelectedEvent], output_frames_dir: Path, cfg: Dict) -> List[Dict]:
    output_frames_dir.mkdir(parents=True, exist_ok=True)
    extracted: List[Dict] = []
    offsets = _offsets_seconds(cfg)

    for item in selected:
        event = item.event
        video_path = Path(event["original_video_path"])
        if not video_path.exists():
            continue

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            continue

        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        duration_sec = frame_count / fps if fps > 0 else 0.0
        event_second = float(event.get("event_second", event.get("second", 0)) or 0.0)

        for offset in offsets:
            target_second = min(max(event_second + float(offset), 0.0), max(duration_sec - 1.0 / fps, 0.0))
            idx = int(target_second * fps)
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ok, frame = cap.read()
            if not ok or frame is None:
                continue

            second = float(idx / fps) if fps > 0 else 0.0
            rel_name = (
                f"{event['station']}_{event['date']}_{event['event_id']}_"
                f"f{int(idx):06d}_o{offset:+.1f}_fish.jpg"
            )
            out_dir = output_frames_dir / "fish"
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / rel_name

            cv2.imwrite(str(out_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])

            extracted.append(
                {
                    "frame_path": str(out_path),
                    "category": "fish",
                    "score": round(item.rank, 5),
                    "station": event["station"],
                    "date": event["date"],
                    "event_id": event["event_id"],
                    "source_video_path": event["original_video_path"],
                    "event_csv_path": event.get("event_csv_path"),
                    "event_second": event_second,
                    "offset_second": float(offset),
                    "frame_index": int(idx),
                    "second": round(second, 4),
                }
            )

        cap.release()

    return extracted


def save_manifest(extracted: List[Dict], output_frames_dir: Path) -> Tuple[Path, Path]:
    manifest = {
        "created": datetime.now().isoformat(),
        "total_frames": len(extracted),
        "frames_by_category": {},
        "frames": extracted,
    }

    for frame in extracted:
        category = frame["category"]
        manifest["frames_by_category"][category] = manifest["frames_by_category"].get(category, 0) + 1

    manifest_path = output_frames_dir / "extraction_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    csv_path = output_frames_dir / "frames_list.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "frame_path",
                "category",
                "score",
                "station",
                "date",
                "event_id",
                "source_video_path",
                "event_csv_path",
                "event_second",
                "offset_second",
                "frame_index",
                "second",
            ],
        )
        writer.writeheader()
        for row in extracted:
            writer.writerow(row)

    return manifest_path, csv_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Sample frames from original videos using indexed stage3 events")
    parser.add_argument("--config", type=str, required=True, help="Path to stage3 branch YAML config")
    parser.add_argument("--index-json", type=str, default=None, help="Override index json path")
    parser.add_argument("--output-frames-dir", type=str, default=None, help="Override frames output directory")
    args = parser.parse_args()

    cfg = load_config(Path(args.config))
    output_dir = Path(cfg["paths"]["output_dir"])
    index_json = Path(args.index_json) if args.index_json else output_dir / "stage3_event_index.json"
    output_frames_dir = Path(args.output_frames_dir) if args.output_frames_dir else output_dir / "frames"

    rows = load_index(index_json)
    print(f"Loaded {len(rows)} indexed events")

    selected = select_events(rows, cfg)
    print(f"Selected {len(selected)} events")

    extracted = extract_frames(selected, output_frames_dir, cfg)
    manifest_path, csv_path = save_manifest(extracted, output_frames_dir)

    print(f"Extracted frames: {len(extracted)}")
    print(f"Manifest: {manifest_path}")
    print(f"CSV list: {csv_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
