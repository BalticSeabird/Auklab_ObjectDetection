#!/usr/bin/env python3
"""
Select only images that have non-empty YOLO pre-annotations.

This is useful before upload when you want to send only frames that already
contain at least one predicted object.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG")


def has_non_empty_annotation(label_path: Path) -> bool:
    if not label_path.exists():
        return False
    return label_path.read_text(encoding="utf-8").strip() != ""


def copy_positive_samples(frames_dir: Path, output_dir: Path, batch_types: list[str] | None = None) -> dict:
    annotations_dir = frames_dir / "annotations" / "yolo"
    if not annotations_dir.exists():
        raise FileNotFoundError(f"Annotations directory not found: {annotations_dir}")

    available_types = [
        "edge_detection",
        "spike",
        "dip",
        "high_count",
        "count_transition",
        "fish",
    ]

    if batch_types is None:
        batch_types = [b for b in available_types if (frames_dir / b).exists()]
    else:
        batch_types = [b for b in batch_types if b in available_types and (frames_dir / b).exists()]

    (output_dir / "annotations" / "yolo").mkdir(parents=True, exist_ok=True)

    summary = {
        "total_images_seen": 0,
        "total_images_selected": 0,
        "selected_by_batch": {},
    }

    for batch in batch_types:
        src_batch_dir = frames_dir / batch
        dst_batch_dir = output_dir / batch
        dst_batch_dir.mkdir(parents=True, exist_ok=True)

        selected = 0
        images = [p for p in src_batch_dir.iterdir() if p.is_file() and p.suffix in IMAGE_EXTS]
        summary["total_images_seen"] += len(images)

        for img_path in images:
            label_path = annotations_dir / f"{img_path.stem}.txt"
            if not has_non_empty_annotation(label_path):
                continue

            shutil.copy2(img_path, dst_batch_dir / img_path.name)
            shutil.copy2(label_path, output_dir / "annotations" / "yolo" / label_path.name)
            selected += 1

        summary["selected_by_batch"][batch] = selected
        summary["total_images_selected"] += selected

    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Select frames that have non-empty YOLO pre-annotations")
    parser.add_argument("--frames-dir", required=True, type=str, help="Input frames directory")
    parser.add_argument("--output-dir", required=True, type=str, help="Output directory for selected frames")
    parser.add_argument(
        "--batches",
        nargs="+",
        choices=["edge_detection", "spike", "dip", "high_count", "count_transition", "fish"],
        default=None,
        help="Optional batch types to include",
    )
    args = parser.parse_args()

    frames_dir = Path(args.frames_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = copy_positive_samples(frames_dir=frames_dir, output_dir=output_dir, batch_types=args.batches)

    print("=" * 70)
    print("POSITIVE PRE-ANNOTATION SELECTION COMPLETE")
    print("=" * 70)
    print(f"Input frames dir: {frames_dir}")
    print(f"Output dir: {output_dir}")
    print(f"Images seen: {summary['total_images_seen']}")
    print(f"Images selected: {summary['total_images_selected']}")
    for batch, count in summary["selected_by_batch"].items():
        print(f"  {batch:20s}: {count}")
    print("=" * 70)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
