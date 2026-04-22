#!/usr/bin/env python3
"""
Sample still frames directly from existing clip files (no event CSV required).

Designed for clip-first active learning workflows where videos already exist as
separate files and you want to prioritize certain years and/or stations.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Optional

import cv2
import numpy as np


VIDEO_EXTS = (".mkv", ".mp4", ".avi", ".mov", ".MKV", ".MP4", ".AVI", ".MOV")


@dataclass
class ClipInfo:
    path: Path
    year: Optional[str]
    station: Optional[str]


def _extract_year(path: Path) -> Optional[str]:
    """Extract a 4-digit year from path components or filename."""
    text = str(path)
    match = re.search(r"(20\d{2})", text)
    return match.group(1) if match else None


def _extract_station(path: Path) -> Optional[str]:
    """Extract station name from filename (e.g., 'FAR3_SCALE' from 'FAR3_SCALE_2023-05-24_12_40.mp4')."""
    filename = path.stem  # Get filename without extension
    # Match everything before a date pattern (YYYY-MM-DD or YYYY_MM_DD)
    match = re.match(r"(.+?)_\d{4}[-_]\d{2}[-_]\d{2}", filename)
    return match.group(1) if match else None


def discover_clips(clip_roots: List[Path]) -> List[ClipInfo]:
    clips: List[ClipInfo] = []
    for root in clip_roots:
        if not root.exists():
            continue
        for ext in VIDEO_EXTS:
            for clip_path in root.rglob(f"*{ext}"):
                if clip_path.is_file():
                    clips.append(ClipInfo(
                        path=clip_path,
                        year=_extract_year(clip_path),
                        station=_extract_station(clip_path)
                    ))
    return sorted(clips, key=lambda c: str(c.path))


def prioritize_and_select_clips(
    clips: List[ClipInfo],
    max_clips: int,
    preferred_years: List[str],
    preferred_stations: List[str],
    seed: int,
) -> List[ClipInfo]:
    if not clips:
        return []

    random.seed(seed)

    # Three-tier priority system
    both = [
        c for c in clips
        if c.year in preferred_years and c.station in preferred_stations
    ]
    either = [
        c for c in clips
        if c not in both and (c.year in preferred_years or c.station in preferred_stations)
    ]
    other = [
        c for c in clips
        if c not in both and c not in either
    ]

    random.shuffle(both)
    random.shuffle(either)
    random.shuffle(other)

    ordered = both + either + other
    if max_clips <= 0 or max_clips >= len(ordered):
        return ordered
    return ordered[:max_clips]


def _target_frame_indices(frame_count: int, frames_per_clip: int) -> Iterable[int]:
    if frame_count <= 0:
        return []
    if frames_per_clip <= 1:
        return [max(frame_count // 2, 0)]

    # Sample across clip interior (avoid first/last edge frames).
    start = max(int(frame_count * 0.1), 0)
    end = max(int(frame_count * 0.9), start)
    return [int(x) for x in np.linspace(start, end, num=frames_per_clip)]


def extract_frames(
    selected_clips: List[ClipInfo],
    output_frames_dir: Path,
    frames_per_clip: int,
    category_name: str,
) -> List[dict]:
    output_category_dir = output_frames_dir / category_name
    output_category_dir.mkdir(parents=True, exist_ok=True)

    extracted: List[dict] = []

    for clip in selected_clips:
        cap = cv2.VideoCapture(str(clip.path))
        if not cap.isOpened():
            continue

        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        for idx in _target_frame_indices(frame_count, frames_per_clip):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ok, frame = cap.read()
            if not ok or frame is None:
                continue

            second = float(idx / fps) if fps > 0 else 0.0
            out_name = f"{clip.path.stem}_f{int(idx):06d}.jpg"
            out_path = output_category_dir / out_name

            cv2.imwrite(str(out_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])

            extracted.append(
                {
                    "frame_path": str(out_path),
                    "category": category_name,
                    "source_clip_path": str(clip.path),
                    "year": clip.year,
                    "station": clip.station,
                    "frame_index": int(idx),
                    "second": round(second, 4),
                }
            )

        cap.release()

    return extracted


def write_outputs(extracted: List[dict], output_frames_dir: Path) -> None:
    manifest = {
        "created": datetime.now().isoformat(),
        "total_frames": len(extracted),
        "frames": extracted,
    }

    manifest_path = output_frames_dir / "extraction_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    csv_path = output_frames_dir / "frames_list.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["frame_path", "category", "source_clip_path", "year", "station", "frame_index", "second"],
        )
        writer.writeheader()
        for row in extracted:
            writer.writerow(row)

    print(f"Manifest: {manifest_path}")
    print(f"CSV list: {csv_path}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Sample frames from a directory of existing clips")
    parser.add_argument(
        "--clip-roots",
        type=str,
        nargs="+",
        required=True,
        help="One or more root directories containing clip files",
    )
    parser.add_argument(
        "--output-frames-dir",
        type=str,
        default="data/active_learning_clips/frames",
        help="Output frames directory",
    )
    parser.add_argument(
        "--max-clips",
        type=int,
        default=120,
        help="Maximum number of clips to sample (<=0 means all discovered clips)",
    )
    parser.add_argument(
        "--frames-per-clip",
        type=int,
        default=2,
        help="Number of frames to sample from each selected clip",
    )
    parser.add_argument(
        "--preferred-years",
        type=str,
        nargs="*",
        default=["2025"],
        help="Years to prioritize first, e.g. 2025 2024",
    )
    parser.add_argument(
        "--preferred-stations",
        type=str,
        nargs="*",
        default=[],
        help="Station IDs to prioritize, e.g. 1064 1318",
    )
    parser.add_argument(
        "--category-name",
        type=str,
        default="metal_ring",
        help="Output category subfolder name (used by upload script as batch folder)",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="Random seed for clip ordering within priority groups",
    )

    args = parser.parse_args()

    clip_roots = [Path(p) for p in args.clip_roots]
    output_frames_dir = Path(args.output_frames_dir)
    output_frames_dir.mkdir(parents=True, exist_ok=True)

    clips = discover_clips(clip_roots)
    print(f"Discovered clips: {len(clips)}")

    # Debug: show extracted years and stations
    stations_found = set()
    years_found = set()
    for clip in clips:
        if clip.station:
            stations_found.add(clip.station)
        if clip.year:
            years_found.add(clip.year)
    print(f"Stations found: {sorted(stations_found)}")
    print(f"Years found: {sorted(years_found)}")

    selected = prioritize_and_select_clips(
        clips=clips,
        max_clips=args.max_clips,
        preferred_years=[str(y) for y in args.preferred_years],
        preferred_stations=[str(s) for s in args.preferred_stations],
        seed=args.random_seed,
    )
    print(f"Selected clips: {len(selected)}")

    extracted = extract_frames(
        selected_clips=selected,
        output_frames_dir=output_frames_dir,
        frames_per_clip=args.frames_per_clip,
        category_name=args.category_name,
    )

    print(f"Extracted frames: {len(extracted)}")
    write_outputs(extracted, output_frames_dir)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
