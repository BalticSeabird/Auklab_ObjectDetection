#!/usr/bin/env python3
"""
Index stage3 events for active learning.

Scans stage3 `_events.csv` files, filters fish-arrival events, and records
the corresponding original video paths and event times for frame sampling.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass, asdict
from pathlib import Path
import re
from typing import Dict, List, Optional, Tuple

import yaml


TRUE_STRINGS = {"1", "true", "yes", "y", "t"}


@dataclass
class EventRecord:
    station: str
    date: str
    event_id: str
    event_type: str
    event_second: int
    arrival_with_fish: bool
    fish_count: int
    original_video_path: str
    original_video_exists: bool
    event_video_path: Optional[str]
    absolute_timestamp: Optional[str]
    event_csv_path: str


def load_config(path: Path) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def discover_event_csv_files(stage3_roots: List[Path]) -> List[Path]:
    event_csv_files: List[Path] = []

    for root in stage3_roots:
        if not root.exists():
            continue
        for file_path in root.rglob("*_events.csv"):
            if file_path.is_file():
                event_csv_files.append(file_path)

    return sorted(event_csv_files)


def parse_station_date(csv_path: Path) -> Tuple[Optional[str], Optional[str]]:
    # Expected: .../{station}/{YYYYMMDD}/events/*_events.csv
    parts = csv_path.parts
    if "events" in parts:
        idx = parts.index("events")
        if idx >= 2:
            return parts[idx - 2], parts[idx - 1]

    match = re.search(r"([A-Za-z0-9]+)_(\d{8})T\d{6}_events\.csv$", csv_path.name)
    if match:
        return match.group(1), match.group(2)

    return None, None


def should_process_csv_path(csv_path: Path, cfg: Dict) -> bool:
    station, date = parse_station_date(csv_path)
    if not station or not date:
        return False

    filters = cfg.get("filters", {})
    include = set(filters.get("include_stations") or [])
    exclude = set(filters.get("exclude_stations") or [])
    date_from = filters.get("date_from")
    date_to = filters.get("date_to")

    if include and station not in include:
        return False
    if exclude and station in exclude:
        return False
    if date_from and date < str(date_from):
        return False
    if date_to and date > str(date_to):
        return False
    return True


def _safe_int(value: str, default: int = 0) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _as_bool(value: str) -> bool:
    if value is None:
        return False
    return str(value).strip().lower() in TRUE_STRINGS


def _resolve_path(path_text: str) -> Path:
    return Path(path_text).expanduser()


def _is_target_event(row: Dict, include_event_types: List[str], require_fish_arrival: bool) -> bool:
    event_type = str(row.get("type", "")).strip().lower()
    if include_event_types and event_type not in include_event_types:
        return False

    if not require_fish_arrival:
        return True

    arrival_with_fish = _as_bool(row.get("arrival_with_fish", ""))
    fish_count = _safe_int(row.get("fish_count", "0"), 0)
    return event_type == "arrival" and (arrival_with_fish or fish_count > 0)


def apply_filters(records: List[EventRecord], cfg: Dict) -> List[EventRecord]:
    filters = cfg.get("filters", {})
    include = set(filters.get("include_stations") or [])
    exclude = set(filters.get("exclude_stations") or [])
    date_from = filters.get("date_from")
    date_to = filters.get("date_to")

    result: List[EventRecord] = []
    for rec in records:
        if include and rec.station not in include:
            continue
        if exclude and rec.station in exclude:
            continue
        if date_from and rec.date < str(date_from):
            continue
        if date_to and rec.date > str(date_to):
            continue
        result.append(rec)

    return result


def save_index(records: List[EventRecord], output_dir: Path) -> Tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "stage3_event_index.json"
    csv_path = output_dir / "stage3_event_index.csv"

    payload = {
        "total_events": len(records),
        "events": [asdict(r) for r in records],
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    fieldnames = list(asdict(records[0]).keys()) if records else [
        "station",
        "date",
        "event_id",
        "event_type",
        "event_second",
        "arrival_with_fish",
        "fish_count",
        "original_video_path",
        "original_video_exists",
        "event_video_path",
        "absolute_timestamp",
        "event_csv_path",
    ]

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for rec in records:
            writer.writerow(asdict(rec))

    return json_path, csv_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Index stage3 event CSV files for active learning")
    parser.add_argument("--config", type=str, required=True, help="Path to stage3 branch YAML config")
    parser.add_argument("--output-dir", type=str, default=None, help="Override output directory")
    args = parser.parse_args()

    cfg = load_config(Path(args.config))
    output_dir = Path(args.output_dir or cfg["paths"]["output_dir"])
    stage3_roots = [Path(p) for p in cfg["paths"].get("stage3_roots", [])]
    filter_cfg = cfg.get("filters", {})
    include_event_types = [str(t).strip().lower() for t in filter_cfg.get("include_event_types", ["arrival"])]
    require_fish_arrival = bool(filter_cfg.get("require_fish_arrival", True))
    require_existing_video = bool(filter_cfg.get("require_existing_video", True))

    event_csv_files = discover_event_csv_files(stage3_roots)
    print(f"Discovered {len(event_csv_files)} event CSV files")

    records: List[EventRecord] = []
    for csv_path in event_csv_files:
        if not should_process_csv_path(csv_path, cfg):
            continue

        station, date = parse_station_date(csv_path)
        if not station or not date:
            continue

        with open(csv_path, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if not _is_target_event(row, include_event_types, require_fish_arrival):
                    continue

                original_video_path = _resolve_path(row.get("original_video_path", ""))
                has_video = original_video_path.exists()
                if require_existing_video and not has_video:
                    continue

                records.append(
                    EventRecord(
                        station=station,
                        date=date,
                        event_id=str(row.get("event_id", "")).strip(),
                        event_type=str(row.get("type", "")).strip().lower(),
                        event_second=_safe_int(row.get("second", "0"), 0),
                        arrival_with_fish=_as_bool(row.get("arrival_with_fish", "")),
                        fish_count=_safe_int(row.get("fish_count", "0"), 0),
                        original_video_path=str(original_video_path),
                        original_video_exists=has_video,
                        event_video_path=(str(row.get("event_video_path", "")).strip() or None),
                        absolute_timestamp=(str(row.get("absolute_timestamp", "")).strip() or None),
                        event_csv_path=str(csv_path),
                    )
                )

    records = apply_filters(records, cfg)
    json_path, csv_path = save_index(records, output_dir)

    print(f"Indexed events: {len(records)}")
    print(f"Wrote JSON index: {json_path}")
    print(f"Wrote CSV index: {csv_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
