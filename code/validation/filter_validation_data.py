#!/usr/bin/env python3
"""
Filter class_validation.csv to test model training with/without problematic stations.

This helps identify if certain stations (e.g., ROST2-4 with poor detection quality)
are hurting model training performance.

Usage:
    # Test without ROST2-4 (high-quality data only)
    python code/validation/filter_validation_data.py \
        --exclude-stations ROST2,ROST3,ROST4 \
        --output data/class_validation_no_rost24.csv

    # Test with only TRI3 (single station)
    python code/validation/filter_validation_data.py \
        --include-stations TRI3 \
        --output data/class_validation_tri3_only.csv

    # Then train with filtered data:
    python code/postprocess/train_stage4_tri3_model.py \
        --validation-csv data/class_validation_no_rost24.csv \
        --output-model models/stage4/tri3_fish_arrival_model_filtered.json \
        --output-report data/stage4_training_report_filtered.json
"""

import argparse
from pathlib import Path
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger(__name__)


def filter_validation_data(
    input_csv: Path,
    output_csv: Path,
    exclude_stations: list = None,
    include_stations: list = None,
) -> None:
    """Filter validation CSV by stations.

    Args:
        input_csv: Path to input class_validation.csv
        output_csv: Path to output filtered CSV
        exclude_stations: List of station codes to REMOVE
        include_stations: List of station codes to KEEP (if set, only these are kept)
    """
    LOGGER.info(f"Loading: {input_csv}")
    df = pd.read_csv(input_csv, sep=";")
    original_count = len(df)

    LOGGER.info(f"Original records: {original_count}")
    LOGGER.info(f"Unique stations: {sorted(df['station'].unique())}")

    # Apply filters
    if include_stations:
        LOGGER.info(f"Keeping only: {include_stations}")
        df = df[df["station"].isin(include_stations)]
    elif exclude_stations:
        LOGGER.info(f"Excluding: {exclude_stations}")
        df = df[~df["station"].isin(exclude_stations)]

    filtered_count = len(df)
    removed = original_count - filtered_count

    LOGGER.info(f"After filtering:")
    LOGGER.info(f"  Remaining records: {filtered_count} (removed {removed})")
    LOGGER.info(f"  Stations included: {sorted(df['station'].unique())}")

    # Count annotations
    has_labels = df[
        (df["valid_arrival"].notna()) |
        (df["valid_fish"].notna()) |
        (df["valid_fish_arrival"].notna())
    ]
    LOGGER.info(f"  Labeled records: {len(has_labels)}")

    # Save
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, sep=";", index=False)
    LOGGER.info(f"\nSaved: {output_csv}")

    # Summary by station
    print("\n" + "=" * 70)
    print("STATION BREAKDOWN")
    print("=" * 70)
    for station in sorted(df["station"].unique()):
        station_df = df[df["station"] == station]
        labeled = len(station_df[
            (station_df["valid_arrival"].notna()) |
            (station_df["valid_fish"].notna()) |
            (station_df["valid_fish_arrival"].notna())
        ])
        print(f"{station:15} {len(station_df):5} records ({labeled:3} labeled)")
    print("=" * 70 + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Filter class_validation.csv by stations"
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/class_validation.csv"),
        help="Input validation CSV",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output filtered CSV",
    )
    parser.add_argument(
        "--exclude-stations",
        type=str,
        default="",
        help="Comma-separated stations to REMOVE (e.g., ROST2,ROST3,ROST4)",
    )
    parser.add_argument(
        "--include-stations",
        type=str,
        default="",
        help="Comma-separated stations to KEEP (if set, only these; overrides exclude)",
    )

    args = parser.parse_args()

    exclude_list = [s.strip() for s in args.exclude_stations.split(",") if s.strip()] if args.exclude_stations else None
    include_list = [s.strip() for s in args.include_stations.split(",") if s.strip()] if args.include_stations else None

    if include_list and exclude_list:
        LOGGER.warning("Both --include-stations and --exclude-stations specified. Using include-stations only.")

    filter_validation_data(
        input_csv=args.input,
        output_csv=args.output,
        exclude_stations=exclude_list,
        include_stations=include_list,
    )

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
