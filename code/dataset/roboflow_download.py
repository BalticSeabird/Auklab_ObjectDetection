#!/usr/bin/env python3
"""
Download a Roboflow dataset version for retraining.

# API1 - 2Z8LedwxqBlKAbVYyz8T
# ring_detection-497ko
# tag_detection-g6gi0
# fish_seabirds_combined-625bd

# API2 - X2yHJUrUKxkMNDPlzaAd
# Workspace: Research
# auklab_seabirdfish-9b8c2


Usage example:
  export ROBOFLOW_API_KEY="X2yHJUrUKxkMNDPlzaAd"
  python3 code/dataset/roboflow_download.py \
	  --workspace ai-course-2024 \
	  --project fish_seabirds_combined-625bd \
	  --version 12 \
	  --format yolov11
"""

"""
Usage example:
  export ROBOFLOW_API_KEY="2Z8LedwxqBlKAbVYyz8T"
  python3 code/dataset/roboflow_download.py \
	  --workspace research-x1kcu \
	  --project auklab_seabirdfish \
	  --version 1 \
	  --format yolov11
"""


from __future__ import annotations

import argparse
import os

from roboflow import Roboflow


def main() -> int:
	parser = argparse.ArgumentParser(description="Download a dataset version from Roboflow")
	parser.add_argument("--workspace", required=True, type=str, help="Roboflow workspace slug")
	parser.add_argument("--project", required=True, type=str, help="Roboflow project slug")
	parser.add_argument("--version", required=True, type=int, help="Roboflow dataset version number")
	parser.add_argument(
		"--format",
		default="yolov11",
		choices=["yolov11", "yolov8", "yolov5", "coco", "voc"],
		help="Download format",
	)
	parser.add_argument("--api-key", default=None, type=str, help="Roboflow API key (or use ROBOFLOW_API_KEY)")
	args = parser.parse_args()

	api_key = args.api_key or os.environ.get("ROBOFLOW_API_KEY")
	if not api_key:
		raise SystemExit("Missing API key. Use --api-key or set ROBOFLOW_API_KEY.")

	rf = Roboflow(api_key=api_key)
	project = rf.workspace(args.workspace).project(args.project)
	dataset = project.version(args.version).download(args.format)

	print(f"Downloaded dataset version: {args.version}")
	print(f"Format: {args.format}")
	print(f"Location: {dataset.location}")
	return 0


if __name__ == "__main__":
	raise SystemExit(main())