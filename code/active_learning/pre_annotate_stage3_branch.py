#!/usr/bin/env python3
"""
Pre-annotate stage3 branch frames with per-class confidence thresholds.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import cv2
import yaml

try:
    from ultralytics import YOLO
except ImportError as exc:
    raise SystemExit("ultralytics is required. Install with: pip install ultralytics") from exc


def load_config(path: Path) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _to_float(value) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def annotate_frames(frames_dir: Path, output_dir: Path, model_path: Path, class_thresholds: Dict[str, float]) -> Dict:
    model = YOLO(str(model_path))
    class_names = model.names

    images: List[Path] = []
    for ext in ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"):
        images.extend(frames_dir.rglob(ext))

    yolo_dir = output_dir / "yolo"
    yolo_dir.mkdir(parents=True, exist_ok=True)

    stats = {
        "total_images": len(images),
        "images_with_detections": 0,
        "total_detections": 0,
        "detections_by_class": {},
    }

    for img_path in images:
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        img_h, img_w = img.shape[:2]

        result = model(str(img_path), conf=0.01, verbose=False)[0]
        detections = []

        for box in result.boxes:
            cls_id = int(box.cls[0].cpu().numpy())
            cls_name = str(class_names[cls_id])
            conf = _to_float(box.conf[0].cpu().numpy())

            min_conf = _to_float(class_thresholds.get(cls_name, class_thresholds.get("default", 0.3)))
            if conf < min_conf:
                continue

            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().tolist()
            cx = ((x1 + x2) / 2.0) / img_w
            cy = ((y1 + y2) / 2.0) / img_h
            bw = (x2 - x1) / img_w
            bh = (y2 - y1) / img_h

            detections.append(
                {
                    "class_id": cls_id,
                    "class_name": cls_name,
                    "confidence": conf,
                    "bbox_normalized": {
                        "center_x": cx,
                        "center_y": cy,
                        "width": bw,
                        "height": bh,
                    },
                    "bbox_pixels": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
                }
            )

            stats["detections_by_class"][cls_name] = stats["detections_by_class"].get(cls_name, 0) + 1
            stats["total_detections"] += 1

        if detections:
            stats["images_with_detections"] += 1

        txt_path = yolo_dir / f"{img_path.stem}.txt"
        with open(txt_path, "w", encoding="utf-8") as f:
            for d in detections:
                b = d["bbox_normalized"]
                f.write(
                    f"{d['class_id']} {b['center_x']:.6f} {b['center_y']:.6f} {b['width']:.6f} {b['height']:.6f}\n"
                )

    return stats


def main() -> int:
    parser = argparse.ArgumentParser(description="Pre-annotate stage3 branch frames")
    parser.add_argument("--config", type=str, required=True, help="Path to stage3 branch YAML config")
    parser.add_argument("--frames-dir", type=str, default=None, help="Override frames directory")
    parser.add_argument("--output-dir", type=str, default=None, help="Override annotations directory")
    args = parser.parse_args()

    cfg = load_config(Path(args.config))
    output_root = Path(cfg["paths"]["output_dir"])

    frames_dir = Path(args.frames_dir) if args.frames_dir else output_root / "frames"
    annotations_dir = Path(args.output_dir) if args.output_dir else frames_dir / "annotations"

    model_path = Path(cfg.get("pre_annotation", {}).get("model_path", ""))
    class_thresholds = cfg.get("pre_annotation", {}).get("class_thresholds", {"default": 0.3})

    stats = annotate_frames(frames_dir, annotations_dir, model_path, class_thresholds)

    summary_path = annotations_dir / "pre_annotation_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    print(f"Annotated images: {stats['total_images']}")
    print(f"Images with detections: {stats['images_with_detections']}")
    print(f"Total detections: {stats['total_detections']}")
    print(f"Summary: {summary_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
