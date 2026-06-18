#!/usr/bin/env python3
"""
Sample 200 relevant events from event databases and prepare them for annotation.

Steps:
1. Query events from all station databases
2. Sample diverse events (mix of current stage4 classifications)
3. Copy event videos and detections CSVs to annotation folder
4. Create annotation template CSV with metadata

Usage:
python3 code/validation/sample_and_prepare_annotations.py \
    --num-events 50 \
    --output-dir /mnt/BSP_NAS2_work/temp/stage4_annotation2 \
    --events-db-root data/events_db \
    --annotation-csv data/class_validation_new_batch02.csv
"""

import argparse
import logging
import sqlite3
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Tuple
import random
import shutil

import pandas as pd

PROJECT_CODE_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_CODE_DIR))

LOGGER = logging.getLogger(__name__)


def setup_logging(log_dir: Path) -> None:
    """Configure logging."""
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"sample_annotations_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout),
        ],
    )
    LOGGER.info(f"Logging to {log_file}")


class AnnotationSampler:
    """Sample and prepare events for annotation."""

    def __init__(
        self,
        events_db_root: Path,
        output_base: Path,
        num_events: int,
        annotation_csv: Path,
    ) -> None:
        """Initialize the sampler.

        Args:
            events_db_root: Root directory containing per-station event databases
            output_base: Base output directory for annotation materials
            num_events: Number of events to sample
            annotation_csv: Path to output annotation CSV template
        """
        self.events_db_root = Path(events_db_root)
        self.output_base = Path(output_base)
        self.num_events = num_events
        self.annotation_csv = Path(annotation_csv)

        self.video_dir = self.output_base / "videos"
        self.csv_dir = self.output_base / "detections_csv"

    def setup_directories(self) -> None:
        """Create necessary directories."""
        self.video_dir.mkdir(parents=True, exist_ok=True)
        self.csv_dir.mkdir(parents=True, exist_ok=True)
        LOGGER.info(f"Created directories: {self.video_dir}, {self.csv_dir}")

    def query_candidate_events(self) -> pd.DataFrame:
        """Query events from all station databases for sampling.

        Prioritizes events that are uncertain or important for training:
        - Mix of different stage4 decision sources
        - Various fish detection counts
        - All stations
        """
        if not self.events_db_root.exists():
            LOGGER.error(f"Events DB root does not exist: {self.events_db_root}")
            return pd.DataFrame()

        all_events = []
        db_files = sorted(self.events_db_root.glob("*_events.db"))
        LOGGER.info(f"Found {len(db_files)} event database files")

        for db_file in db_files:
            try:
                station = db_file.stem.replace("_events", "")
                conn = sqlite3.connect(db_file)
                conn.row_factory = sqlite3.Row

                # Query events with completed stage3 processing
                # Prioritize: actual arrivals, have detections, variety of fish counts
                query = """
                    SELECT *
                    FROM events
                    WHERE event_type = 'arrival'
                    AND is_actual_arrival = 1
                    AND stage3_status = 'completed'
                    AND detections_csv_path IS NOT NULL
                    AND event_video_path IS NOT NULL
                    AND original_video_path IS NOT NULL
                    ORDER BY RANDOM()
                """

                df = pd.read_sql_query(query, conn)
                conn.close()

                if not df.empty:
                    LOGGER.info(
                        f"Found {len(df)} candidate events in {station}"
                    )
                    all_events.append(df)
            except Exception as e:
                LOGGER.warning(f"Error reading events from {db_file}: {e}")
                continue

        if not all_events:
            LOGGER.warning("No candidate events found")
            return pd.DataFrame()

        combined = pd.concat(all_events, ignore_index=True)
        LOGGER.info(f"Total candidate events: {len(combined)}")

        return combined

    def sample_diverse_events(self, candidates: pd.DataFrame) -> pd.DataFrame:
        """Sample events with diversity across decision sources and fish counts."""
        if candidates.empty:
            return pd.DataFrame()

        # Sample with diversity across decision_source if available
        sample_size = min(self.num_events, len(candidates))

        # Try to balance across stations
        stations = candidates["station"].unique()
        per_station = max(1, sample_size // len(stations))

        sampled_list = []
        for station in stations:
            station_events = candidates[candidates["station"] == station]
            n_for_station = min(per_station, len(station_events))
            sampled_list.append(station_events.sample(n=n_for_station))

        sampled = pd.concat(sampled_list, ignore_index=True)

        # Fill up to target if needed
        if len(sampled) < sample_size:
            remaining = sample_size - len(sampled)
            remaining_candidates = candidates[~candidates.index.isin(sampled.index)]
            if len(remaining_candidates) >= remaining:
                sampled = pd.concat(
                    [sampled, remaining_candidates.sample(n=remaining)],
                    ignore_index=True
                )

        LOGGER.info(f"Sampled {len(sampled)} events for annotation")
        return sampled.head(sample_size)

    def copy_event_materials(self, events: pd.DataFrame) -> pd.DataFrame:
        """Copy video and detections CSV files for sampled events.

        Returns the events dataframe with updated local paths.
        """
        copied_events = []
        failures = 0

        for idx, (_, event) in enumerate(events.iterrows(), 1):
            event_id = str(event.get("event_id", ""))
            if not event_id:
                LOGGER.warning(f"Event {idx} has no event_id, skipping")
                failures += 1
                continue

            try:
                # Copy video file
                video_path = Path(str(event.get("event_video_path", "")))
                if not video_path.exists():
                    LOGGER.warning(f"Event {event_id}: video not found at {video_path}")
                    failures += 1
                    continue

                output_video = self.video_dir / f"{event_id}.mp4"
                shutil.copy2(video_path, output_video)
                LOGGER.info(f"[{idx}/{len(events)}] Copied video: {event_id}.mp4")

                # Copy detections CSV
                csv_path = Path(str(event.get("detections_csv_path", "")))
                if csv_path.exists():
                    output_csv = self.csv_dir / f"{event_id}_detections.csv"
                    shutil.copy2(csv_path, output_csv)
                    LOGGER.info(f"  Copied detections CSV: {event_id}_detections.csv")
                else:
                    LOGGER.warning(f"  Detections CSV not found at {csv_path}")

                # Track successful copy with local paths
                event_copy = event.copy()
                event_copy["event_video_path"] = str(output_video)
                event_copy["detections_csv_path"] = str(output_csv) if csv_path.exists() else ""
                copied_events.append(event_copy)

            except Exception as e:
                LOGGER.error(f"Error copying materials for {event_id}: {e}")
                failures += 1
                continue

        LOGGER.info(f"Successfully copied {len(copied_events)} events ({failures} failures)")
        return pd.DataFrame(copied_events) if copied_events else pd.DataFrame()

    def create_annotation_template(self, events: pd.DataFrame) -> None:
        """Create CSV template for annotation with required columns."""
        if events.empty:
            LOGGER.error("No events to create template from")
            return

        # Required columns for stage4 training
        template_data = {
            "id": range(1, len(events) + 1),
            "station": events["station"].values,
            "video_id": events["video_id"].values,
            "year": events["year"].values,
            "date": events["date"].values,
            "filename": events["filename"].values,
            "event_id": events["event_id"].values,
            "valid_arrival": [""] * len(events),  # TO BE FILLED BY ANNOTATOR
            "valid_fish": [""] * len(events),  # TO BE FILLED BY ANNOTATOR
            "valid_fish_arrival": [""] * len(events),  # TO BE FILLED BY ANNOTATOR (MAIN TARGET)
            "valid_multiple_fish": [""] * len(events),  # TO BE FILLED BY ANNOTATOR
            "comment": [""] * len(events),  # Optional annotation notes
            "event_type": ["arrival"] * len(events),
            "second": events["second"].values,
            "before_mean": events["before_mean"].values,
            "after_mean": events["after_mean"].values,
            "arrival_with_fish_stage2": events["arrival_with_fish_stage2"].values,
            "fish_count": events["fish_count"].values,
            "fish_mean_area": events["fish_mean_area"].values,
            "fish_max_area": events["fish_max_area"].values,
            "absolute_timestamp": events["absolute_timestamp"].values,
            "original_video_path": events["original_video_path"].values,
            "event_video_path": events["event_video_path"].values,
            "detections_csv_path": events["detections_csv_path"].values,
            "stage3_status": events["stage3_status"].values,
            "is_actual_arrival": events.get("is_actual_arrival", [0] * len(events)).values,
            "is_new_fish_arrival": events.get("is_new_fish_arrival", [0] * len(events)).values,
            "fish_detections_stage4": events.get("fish_detections_stage4", [0] * len(events)).values,
            "fish_avg_confidence_stage4": events.get("fish_avg_confidence_stage4", [0.0] * len(events)).values,
            "stage4_rule_version": events.get("stage4_rule_version", [""] * len(events)).values,
            "stage4_rule_hits": events.get("stage4_rule_hits", [""] * len(events)).values,
            "stage4_features": events.get("stage4_features", [""] * len(events)).values,
        }

        df = pd.DataFrame(template_data)
        df.to_csv(self.annotation_csv, sep=";", index=False)
        LOGGER.info(f"Created annotation template: {self.annotation_csv}")

        # Print instructions
        print("\n" + "=" * 80)
        print("ANNOTATION TEMPLATE CREATED")
        print("=" * 80)
        print(f"\nTemplate CSV: {self.annotation_csv}")
        print(f"Videos folder: {self.video_dir}")
        print(f"Detections CSVs: {self.csv_dir}")
        print(f"\nTotal events to annotate: {len(events)}")
        print("\nAnnotation Instructions:")
        print("-" * 80)
        print("For each event, fill in these columns (0=No, 1=Yes, empty=unsure):")
        print("  • valid_arrival: Is this a real bird arrival event?")
        print("  • valid_fish: Are fish visible in this clip?")
        print("  • valid_fish_arrival: Is this a bird arrival WITH fish? (MAIN TARGET)")
        print("  • valid_multiple_fish: Are there multiple fish visible?")
        print("  • comment: Optional notes about why you labeled it that way")
        print("\nVideo files are numbered by event_id for easy reference.")
        print("=" * 80 + "\n")

    def run(self) -> bool:
        """Execute the full sampling and preparation pipeline."""
        try:
            LOGGER.info("Starting annotation preparation pipeline")
            self.setup_directories()

            # Query candidates
            candidates = self.query_candidate_events()
            if candidates.empty:
                LOGGER.error("No candidate events found")
                return False

            # Sample diverse set
            sampled = self.sample_diverse_events(candidates)
            if sampled.empty:
                LOGGER.error("Failed to sample events")
                return False

            # Copy materials
            copied = self.copy_event_materials(sampled)
            if copied.empty:
                LOGGER.error("Failed to copy any events")
                return False

            # Create template
            self.create_annotation_template(copied)

            LOGGER.info("Annotation preparation complete!")
            return True

        except Exception as e:
            LOGGER.error(f"Error during preparation: {e}", exc_info=True)
            return False


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Sample and prepare events for annotation"
    )
    parser.add_argument(
        "--num-events",
        type=int,
        default=200,
        help="Number of events to sample (default: 200)",
    )
    parser.add_argument(
        "--events-db-root",
        type=Path,
        default=Path("data/events_db"),
        help="Root directory containing per-station event databases",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/annotation_batch"),
        help="Output directory for annotation materials",
    )
    parser.add_argument(
        "--annotation-csv",
        type=Path,
        default=Path("data/class_validation_new_batch.csv"),
        help="Path to output annotation template CSV",
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=Path("logs"),
        help="Log directory",
    )

    args = parser.parse_args()
    setup_logging(args.log_dir)

    sampler = AnnotationSampler(
        events_db_root=args.events_db_root,
        output_base=args.output_dir,
        num_events=args.num_events,
        annotation_csv=args.annotation_csv,
    )

    success = sampler.run()
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
