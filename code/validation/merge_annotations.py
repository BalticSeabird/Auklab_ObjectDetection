#!/usr/bin/env python3
"""
Merge newly annotated validation CSV with existing class_validation.csv.

Keeps the most recent annotation for each event_id (prioritizes new annotations).

Usage:
    python code/validation/merge_annotations.py \
        --new-batch data/class_validation_new_batch.csv \
        --existing data/class_validation.csv \
        --output data/class_validation.csv
"""

import argparse
import logging
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger(__name__)


def merge_annotations(new_csv: Path, existing_csv: Path, output_csv: Path) -> int:
    """Merge new annotations with existing validation data."""
    
    # Load existing data
    if existing_csv.exists():
        LOGGER.info(f"Loading existing annotations: {existing_csv}")
        existing = pd.read_csv(existing_csv, sep=";")
        LOGGER.info(f"  Loaded {len(existing)} records")
    else:
        LOGGER.warning(f"Existing file not found: {existing_csv}, starting fresh")
        existing = pd.DataFrame()

    # Load new data
    LOGGER.info(f"Loading new annotations: {new_csv}")
    new = pd.read_csv(new_csv, sep=";")
    LOGGER.info(f"  Loaded {len(new)} records")

    # Filter: only keep rows that have at least one label filled in
    new_annotated = new[
        (new["valid_arrival"].notna()) | 
        (new["valid_fish"].notna()) | 
        (new["valid_fish_arrival"].notna()) |
        (new["valid_multiple_fish"].notna())
    ].copy()
    
    LOGGER.info(f"  Found {len(new_annotated)} annotated records (not templates)")

    if new_annotated.empty:
        LOGGER.warning("No annotated records found in new batch!")
        return 1

    # Combine
    if existing.empty:
        merged = new_annotated.copy()
    else:
        merged = pd.concat([existing, new_annotated], ignore_index=True)

    # Remove duplicates by event_id, keeping the last (most recent) annotation
    LOGGER.info(f"Merged size: {len(merged)} (before dedup)")
    merged = merged.drop_duplicates(subset=["event_id"], keep="last")
    LOGGER.info(f"After removing duplicates: {len(merged)} records")

    # Save
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output_csv, sep=";", index=False)
    LOGGER.info(f"Saved merged annotations: {output_csv}")

    # Summary
    print("\n" + "=" * 80)
    print("MERGE SUMMARY")
    print("=" * 80)
    print(f"New annotated records: {len(new_annotated)}")
    print(f"Existing records: {len(existing) if not existing.empty else 0}")
    print(f"Total after merge: {len(merged)}")
    print(f"Output: {output_csv}")
    print("=" * 80 + "\n")

    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Merge annotation batches")
    parser.add_argument("--new-batch", type=Path, required=True, help="New annotations CSV")
    parser.add_argument("--existing", type=Path, default=Path("data/class_validation.csv"), help="Existing annotations")
    parser.add_argument("--output", type=Path, required=True, help="Output merged CSV")

    args = parser.parse_args()
    return merge_annotations(args.new_batch, args.existing, args.output)


if __name__ == "__main__":
    import sys
    sys.exit(main())
