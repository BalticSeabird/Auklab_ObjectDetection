#!/usr/bin/env python3
"""
Run stage3 active learning branch pipeline.

Steps:
1) index      - discover and score stage3 clips
2) sample     - sample still frames from selected clips
3) annotate   - pre-annotate sampled frames (optional)
4) upload     - upload frames and annotations to Roboflow (optional)
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Dict, List

import yaml


def load_config(path: Path) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def run_step(cmd: List[str]) -> int:
    print("\n" + "=" * 80)
    print("RUNNING:", " ".join(cmd))
    print("=" * 80)
    return subprocess.call(cmd)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Stage3 active-learning branch")
    parser.add_argument("--config", type=str, required=True, help="Path to stage3 branch YAML config")
    parser.add_argument(
        "--steps",
        nargs="+",
        choices=["index", "sample", "annotate", "upload"],
        default=["index", "sample", "annotate"],
        help="Pipeline steps to run",
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    cfg = load_config(config_path)

    output_dir = Path(cfg["paths"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    base_dir = Path(__file__).parent
    scripts = {
        "index": base_dir / "index_stage3_clips.py",
        "sample": base_dir / "sample_frames_from_stage3_clips.py",
        "annotate": base_dir / "pre_annotate_stage3_branch.py",
        "upload": base_dir / "upload_stage3_branch.py",
    }

    if "index" in args.steps:
        rc = run_step([sys.executable, str(scripts["index"]), "--config", str(config_path)])
        if rc != 0:
            return rc

    if "sample" in args.steps:
        rc = run_step([sys.executable, str(scripts["sample"]), "--config", str(config_path)])
        if rc != 0:
            return rc

    if "annotate" in args.steps and cfg.get("pre_annotation", {}).get("enabled", True):
        rc = run_step([sys.executable, str(scripts["annotate"]), "--config", str(config_path)])
        if rc != 0:
            return rc

    if "upload" in args.steps and cfg.get("upload", {}).get("enabled", False):
        rc = run_step([sys.executable, str(scripts["upload"]), "--config", str(config_path)])
        if rc != 0:
            return rc

    print("\n" + "=" * 80)
    print("STAGE3 ACTIVE LEARNING BRANCH COMPLETE")
    print("=" * 80)
    print(f"Output directory: {output_dir}")
    print("Suggested next step: validate sampled frames before large upload.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
