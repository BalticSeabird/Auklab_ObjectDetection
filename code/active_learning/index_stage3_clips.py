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
import sqlite3
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


def discover_event_db_files(db_paths: List[Path]) -> List[Path]:
    db_files: List[Path] = []

    for path in db_paths:
        expanded = path.expanduser()
        if expanded.is_file() and expanded.suffix == ".db":
            db_files.append(expanded)
            continue
        if expanded.is_dir():
            db_files.extend(sorted(expanded.glob("*_events.db")))

    # De-duplicate while preserving order
    seen = set()
    unique_files: List[Path] = []
    for db_path in db_files:
        key = str(db_path.resolve())
        if key in seen:
            continue
        seen.add(key)
        unique_files.append(db_path)
    return unique_files


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


def _normalize_event_type(row: Dict) -> str:
    return str(row.get("type", row.get("event_type", ""))).strip().lower()


def _get_arrival_with_fish_flag(row: Dict) -> bool:
    if "arrival_with_fish" in row:
        return _as_bool(row.get("arrival_with_fish", ""))
    if "arrival_with_fish_stage2" in row:
        return _as_bool(row.get("arrival_with_fish_stage2", ""))
    return False


def _is_target_event(row: Dict, include_event_types: List[str], require_fish_arrival: bool) -> bool:
    event_type = _normalize_event_type(row)
    if include_event_types and event_type not in include_event_types:
        return False

    if not require_fish_arrival:
        return True

    arrival_with_fish = _get_arrival_with_fish_flag(row)
    fish_count = _safe_int(row.get("fish_count", "0"), 0)
    return event_type == "arrival" and (arrival_with_fish or fish_count > 0)


def _build_event_record_from_row(
    row: Dict,
    *,
    station: Optional[str],
    date: Optional[str],
    source_path: Path,
    require_existing_video: bool,
) -> Optional[EventRecord]:
    source_station = station or str(row.get("station", "")).strip()
    source_date = date or str(row.get("date", "")).strip()

    if not source_station or not source_date:
        return None

    original_video_text = str(row.get("original_video_path", "")).strip()
    if not original_video_text:
        return None

    original_video_path = _resolve_path(original_video_text)
    has_video = original_video_path.exists()
    if require_existing_video and not has_video:
        return None

    return EventRecord(
        station=source_station,
        date=source_date,
        event_id=str(row.get("event_id", "")).strip(),
        event_type=_normalize_event_type(row),
        event_second=_safe_int(row.get("second", row.get("event_second", "0")), 0),
        arrival_with_fish=_get_arrival_with_fish_flag(row),
        fish_count=_safe_int(row.get("fish_count", "0"), 0),
        original_video_path=str(original_video_path),
        original_video_exists=has_video,
        event_video_path=(str(row.get("event_video_path", "")).strip() or None),
        absolute_timestamp=(str(row.get("absolute_timestamp", "")).strip() or None),
        event_csv_path=str(source_path),
    )


def load_events_from_db_files(
    db_files: List[Path],
    cfg: Dict,
    include_event_types: List[str],
    require_fish_arrival: bool,
    require_existing_video: bool,
) -> List[EventRecord]:
    records: List[EventRecord] = []

    for db_path in db_files:
        if not db_path.exists():
            continue

        with sqlite3.connect(db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                """
                SELECT
                    station,
                    date,
                    event_id,
                    event_type,
                    second,
                    arrival_with_fish_stage2,
                    fish_count,
                    original_video_path,
                    event_video_path,
                    absolute_timestamp
                FROM events
                """
            ).fetchall()

        for row in rows:
            row_dict = dict(row)
            if not _is_target_event(row_dict, include_event_types, require_fish_arrival):
                continue

            rec = _build_event_record_from_row(
                row_dict,
                station=None,
                date=None,
                source_path=db_path,
                require_existing_video=require_existing_video,
            )
            if rec is not None:
                records.append(rec)

    return records


def load_events_from_csv_files(
    event_csv_files: List[Path],
    cfg: Dict,
    include_event_types: List[str],
    require_fish_arrival: bool,
    require_existing_video: bool,
) -> List[EventRecord]:
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

                rec = _build_event_record_from_row(
                    row,
                    station=station,
                    date=date,
                    source_path=csv_path,
                    require_existing_video=require_existing_video,
                )
                if rec is not None:
                    records.append(rec)

    return records


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
    db_paths = [Path(p) for p in cfg["paths"].get("events_db_paths", [])]
    if cfg["paths"].get("events_db"):
        db_paths.append(Path(cfg["paths"]["events_db"]))
    filter_cfg = cfg.get("filters", {})
    include_event_types = [str(t).strip().lower() for t in filter_cfg.get("include_event_types", ["arrival"])]
    require_fish_arrival = bool(filter_cfg.get("require_fish_arrival", True))
    require_existing_video = bool(filter_cfg.get("require_existing_video", True))

    event_csv_files = discover_event_csv_files(stage3_roots)
    event_db_files = discover_event_db_files(db_paths)
    print(f"Discovered {len(event_csv_files)} event CSV files")
    print(f"Discovered {len(event_db_files)} event DB files")

    records: List[EventRecord] = []
    records.extend(
        load_events_from_csv_files(
            event_csv_files,
            cfg,
            include_event_types,
            require_fish_arrival,
            require_existing_video,
        )
    )
    records.extend(
        load_events_from_db_files(
            event_db_files,
            cfg,
            include_event_types,
            require_fish_arrival,
            require_existing_video,
        )
    )

    deduped: Dict[Tuple[str, str], EventRecord] = {}
    for rec in records:
        deduped[(rec.station, rec.event_id)] = rec
    records = list(deduped.values())

    records = apply_filters(records, cfg)
    json_path, csv_path = save_index(records, output_dir)

    print(f"Indexed events: {len(records)}")
    print(f"Wrote JSON index: {json_path}")
    print(f"Wrote CSV index: {csv_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
