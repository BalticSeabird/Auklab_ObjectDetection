#!/usr/bin/env python3
"""
Upload stage3 branch frames to Roboflow using existing uploader logic.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Dict

import yaml

from upload_to_roboflow import RoboflowBatchUploader


def load_config(path: Path) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main() -> int:
    parser = argparse.ArgumentParser(description="Upload stage3 branch frames to Roboflow")
    parser.add_argument("--config", type=str, required=True, help="Path to stage3 branch YAML config")
    parser.add_argument("--frames-dir", type=str, default=None, help="Override frames directory")
    parser.add_argument("--batch-id", type=str, default=None, help="Optional custom batch id")
    parser.add_argument("--api-key", type=str, default=None, help="Roboflow API key (or use ROBOFLOW_API_KEY)")
    args = parser.parse_args()

    cfg = load_config(Path(args.config))
    upload_cfg = cfg.get("upload", {})
    output_dir = Path(cfg["paths"]["output_dir"])

    frames_dir = Path(args.frames_dir) if args.frames_dir else output_dir / "frames"
    api_key = args.api_key or os.environ.get("ROBOFLOW_API_KEY")
    workspace = upload_cfg.get("workspace", "")
    project = upload_cfg.get("project", "")

    if not api_key:
        raise SystemExit("Roboflow API key missing. Use --api-key or ROBOFLOW_API_KEY.")
    if not workspace or not project:
        raise SystemExit("upload.workspace and upload.project must be set in config.")

    uploader = RoboflowBatchUploader(api_key=api_key, workspace=workspace, project=project)

    uploader.upload_batches(
        frames_dir=frames_dir,
        batch_types=None,
        use_annotations=bool(upload_cfg.get("use_annotations", True)),
        split=upload_cfg.get("split", "train"),
        batch_name_prefix=upload_cfg.get("batch_name_prefix", "active_learning_stage3"),
        batch_id=args.batch_id,
        resume=bool(upload_cfg.get("resume", True)),
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
