#!/usr/bin/env python3
"""
generate_wwf_clips.py

Generate video clips for WWF citizen science app from detected bird arrival events.
Samples events with high fish detection counts from event databases and creates
video clips suitable for crowdsourced validation and citizen engagement.


python3 code/validation/generate_wwf_clips.py \
    --num-events 300 \
    --events-db-root data/events_db \
    --min-fish-detections 120 \
    --output-base /mnt/BSP_NAS2_work/temp \
    --skip-stations ROST2,ROST3,ROST4,BONDEN5 \
    --new-fish-only \
    --clip-before 4.0 \
    --clip-after 11.0 


Output Structure:
    /mnt/BSP_NAS2_work/wwf_clips_{adjective}_{noun}_{number}/
        wwf_clips_{adjective}_{noun}_{number}.db     # SQLite DB with event metadata
        video/
            event_id_1.mp4
            event_id_2.mp4
            ...
        metadata/
            events_summary.csv

Features:
- Samples events from all available station event databases
- Filters events with high fish_detections_stage4 counts
- Creates SQLite database with all event metadata for the batch
- Extracts video clips without running object detection (fast processing)
- Generates unique, non-reusable batch names (never overwrites batches)
- Logs all operations for audit trail
"""

import argparse
import json
import logging
import sqlite3
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Tuple, List
import random

import pandas as pd

PROJECT_CODE_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_CODE_DIR))

# Imports after path is set
from inference_system.event_database import EventDatabase  # type: ignore  # noqa: E402
from postprocess.extract_event_clips import (  # type: ignore  # noqa: E402
    extract_clip_with_overlay,
    get_video_duration,
)

LOGGER = logging.getLogger(__name__)
NAS2_WORK_BASE = Path("/mnt/BSP_NAS2_work")

# Word lists for generating unique batch names
ADJECTIVES = [
    "arctic", "brave", "calm", "dawn", "eager", "frozen", "golden", "happy",
    "icy", "jolly", "keen", "lively", "misty", "noble", "open", "proud",
    "quiet", "royal", "swift", "trusted", "unique", "vivid", "wild", "yellow",
    "zesty", "active", "bright", "clear", "deep", "excellent", "fresh",
    "gentle", "honest", "ideal", "joyful", "kind", "lucky", "mighty", "nice",
    "old", "perfect", "quick", "rare", "strong", "true", "urgent", "vast",
    "warm", "xeric", "young", "zealous"
]

NOUNS = [
    "albatross", "bear", "coral", "dolphin", "eagle", "fish", "glacier",
    "heron", "island", "jaguar", "kelp", "lion", "mountain", "narwhal",
    "ocean", "penguin", "quail", "reef", "salmon", "tiger", "urchin",
    "vulture", "whale", "xenops", "yak", "zebra", "antelope", "badger",
    "cheetah", "deer", "elk", "falcon", "goose", "hawk", "ibex", "kestrel",
    "leopard", "moose", "newt", "osprey", "pelican", "raven", "seal",
    "tern", "uakari", "viper", "walrus", "yellowfin", "zorilla"
]


class WWFClipsGenerator:
    """Generate video clips for WWF citizen science app."""

    def __init__(
        self,
        events_db_root: Path,
        output_base: Path,
        num_events: int,
        min_fish_detections: int,
        clip_before: float = 2.0,
        clip_after: float = 8.0,
        skip_stations: Optional[List[str]] = None,
        new_fish_only: bool = False,
    ) -> None:
        """Initialize the generator.

        Args:
            events_db_root: Root directory containing per-station event databases
            output_base: Base output directory on NAS2_work
            num_events: Number of events to sample
            min_fish_detections: Minimum number of fish detections to include event
            clip_before: Seconds before event to include in clip
            clip_after: Seconds after event to include in clip
            skip_stations: List of station names to exclude from sampling
            new_fish_only: If True, only include events classified as new fish arrivals
        """
        self.events_db_root = Path(events_db_root)
        self.output_base = Path(output_base)
        self.num_events = num_events
        self.min_fish_detections = min_fish_detections
        self.clip_before = clip_before
        self.clip_after = clip_after
        self.skip_stations = set(skip_stations or [])
        self.new_fish_only = new_fish_only
        self.batch_name = self._generate_unique_batch_name()
        self.batch_dir = self.output_base / self.batch_name
        self.video_dir = self.batch_dir / "video"
        self.metadata_dir = self.batch_dir / "metadata"
        self.db_path = self.batch_dir / f"{self.batch_name}.db"

    def _generate_unique_batch_name(self) -> str:
        """Generate a unique batch name that is never reused.
        
        Format: wwf_clips_{adjective}_{noun}_{number}
        Example: wwf_clips_brave_falcon_427
        """
        max_attempts = 100
        for _ in range(max_attempts):
            adjective = random.choice(ADJECTIVES).lower()
            noun = random.choice(NOUNS).lower()
            number = random.randint(0, 999)
            
            batch_name = f"wwf_clips_{adjective}_{noun}_{number:03d}"
            batch_path = self.output_base / batch_name
            
            # If this batch name doesn't exist, use it
            if not batch_path.exists():
                LOGGER.info(f"Generated unique batch name: {batch_name}")
                return batch_name
        
        # Fallback if we somehow collide 100 times (extremely unlikely)
        raise RuntimeError("Could not generate unique batch name after 100 attempts")

    def setup_directories(self) -> None:
        """Create necessary directories."""
        self.batch_dir.mkdir(parents=True, exist_ok=True)
        self.video_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_dir.mkdir(parents=True, exist_ok=True)
        LOGGER.info(f"Created batch directory: {self.batch_dir}")

    def initialize_database(self) -> sqlite3.Connection:
        """Create SQLite database with schema matching events_db."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA foreign_keys=ON;")

        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS schema_migrations (
                version INTEGER PRIMARY KEY,
                applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS wwf_clips (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                station TEXT NOT NULL,
                video_id TEXT NOT NULL,
                year INTEGER NOT NULL,
                date TEXT NOT NULL,
                filename TEXT NOT NULL,
                event_id TEXT NOT NULL UNIQUE,
                event_type TEXT,
                second REAL,
                before_mean REAL,
                after_mean REAL,
                arrival_with_fish_stage2 INTEGER,
                fish_count REAL,
                fish_mean_area REAL,
                fish_max_area REAL,
                absolute_timestamp TEXT,
                original_video_path TEXT,
                event_video_path TEXT,
                detections_csv_path TEXT,
                stage3_status TEXT,
                is_actual_arrival INTEGER,
                is_new_fish_arrival INTEGER,
                fish_detections_stage4 INTEGER,
                fish_avg_confidence_stage4 REAL,
                stage4_rule_version TEXT,
                stage4_rule_hits TEXT,
                stage4_features TEXT,
                stage4_model_score REAL,
                stage4_decision_source TEXT,
                clip_path TEXT,
                clip_created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );

            CREATE INDEX IF NOT EXISTS idx_wwf_clips_event_id
            ON wwf_clips(event_id);

            CREATE INDEX IF NOT EXISTS idx_wwf_clips_station
            ON wwf_clips(station);

            CREATE INDEX IF NOT EXISTS idx_wwf_clips_fish_detections
            ON wwf_clips(fish_detections_stage4);
            """
        )
        conn.commit()
        LOGGER.info(f"Initialized database: {self.db_path}")
        return conn

    def sample_events(self) -> pd.DataFrame:
        """Sample events with high fish detection counts from all station DBs."""
        if not self.events_db_root.exists():
            LOGGER.error(f"Events DB root does not exist: {self.events_db_root}")
            return pd.DataFrame()

        all_events = []
        db_files = sorted(self.events_db_root.glob("*_events.db"))
        LOGGER.info(f"Found {len(db_files)} event database files")

        for db_file in db_files:
            try:
                station = db_file.stem.replace("_events", "")

                if station in self.skip_stations:
                    LOGGER.info(f"Skipping station {station} (in skip list)")
                    continue

                conn = sqlite3.connect(db_file)
                conn.row_factory = sqlite3.Row

                # Query events with sufficient fish detections
                where_clause = """
                    WHERE fish_detections_stage4 >= ?
                    AND is_actual_arrival = 1
                    AND original_video_path IS NOT NULL
                """

                if self.new_fish_only:
                    where_clause += "                    AND is_new_fish_arrival = 1\n"
                    LOGGER.info(f"Filtering for new fish arrivals only in {station}")

                query = f"""
                    SELECT *
                    FROM events
                    {where_clause}
                    ORDER BY fish_detections_stage4 DESC
                """

                df = pd.read_sql_query(
                    query, conn, params=(self.min_fish_detections,)
                )
                conn.close()

                if not df.empty:
                    LOGGER.info(
                        f"Found {len(df)} eligible events in {station} "
                        f"with >= {self.min_fish_detections} fish detections"
                    )
                    all_events.append(df)
            except Exception as e:
                LOGGER.warning(f"Error reading events from {db_file}: {e}")
                continue

        if not all_events:
            LOGGER.warning("No eligible events found across all databases")
            return pd.DataFrame()

        combined = pd.concat(all_events, ignore_index=True)
        LOGGER.info(
            f"Total eligible events across all stations: {len(combined)}"
        )

        # Sample events, ensuring we get high fish detection counts
        sample_size = min(self.num_events, len(combined))
        sampled = combined.sample(n=sample_size)  # Truly random - no fixed seed
        LOGGER.info(f"Sampled {sample_size} events for batch")

        return sampled

    def extract_clip(self, event: pd.Series, video_dir: Path) -> Optional[Path]:
        """Extract video clip for a single event.

        Args:
            event: Event data series
            video_dir: Output directory for video

        Returns:
            Path to created clip or None if extraction failed
        """
        try:
            event_id = str(event.get("event_id", ""))
            if not event_id:
                LOGGER.warning("Event has no event_id, skipping")
                return None

            # Get absolute timestamp
            abs_timestamp = event.get("absolute_timestamp")
            second_offset = event.get("second")

            if not abs_timestamp and second_offset is None:
                LOGGER.warning(
                    f"Event {event_id} has no timestamp or offset, skipping"
                )
                return None

            # Get original video path
            video_path = Path(str(event.get("original_video_path", "")))
            if not video_path.exists():
                LOGGER.warning(
                    f"Original video not found for {event_id}: {video_path}"
                )
                return None

            # Determine clip timing
            if abs_timestamp:
                # Parse absolute timestamp to get offset in video
                event_time = pd.to_datetime(abs_timestamp)
                try:
                    # Extract timestamp from video filename if available
                    # Format: STATION_YYYYMMDDTHHMMSS_*.ext
                    import re
                    match = re.search(r'(\d{8})T(\d{6})', video_path.name)
                    if match:
                        video_date = match.group(1)
                        video_time = match.group(2)
                        video_start = datetime.strptime(
                            f"{video_date}T{video_time}", '%Y%m%dT%H%M%S'
                        )
                        offset_seconds = (event_time - video_start).total_seconds()
                    else:
                        # Fallback: use second offset if available
                        offset_seconds = float(second_offset) if second_offset else None
                except Exception as e:
                    LOGGER.warning(
                        f"Error parsing video timestamp for {event_id}: {e}"
                    )
                    offset_seconds = float(second_offset) if second_offset else None
            else:
                offset_seconds = float(second_offset)

            if offset_seconds is None or offset_seconds < 0:
                LOGGER.warning(
                    f"Invalid offset for {event_id}: {offset_seconds}"
                )
                return None

            # Calculate clip boundaries
            clip_start = max(0.0, offset_seconds - self.clip_before)
            clip_end = offset_seconds + self.clip_after

            # Validate against video duration
            try:
                video_duration = get_video_duration(video_path)
                clip_end = min(video_duration, clip_end)
            except Exception as e:
                LOGGER.warning(
                    f"Could not get video duration for {event_id}: {e}"
                )
                # Continue anyway with clip_end as planned

            if clip_end - clip_start <= 0:
                LOGGER.warning(f"Invalid clip window for {event_id}")
                return None

            # Create output filename
            output_filename = f"{event_id}.mp4"
            output_path = video_dir / output_filename

            # Format overlay text
            fish_count = event.get("fish_detections_stage4", 0)
            timestamp_str = abs_timestamp or f"offset:{second_offset}s"
            overlay_text = f"{event_id} | {fish_count} fish | {timestamp_str}"

            # Extract clip with overlay
            success = extract_clip_with_overlay(
                video_path,
                clip_start,
                clip_end,
                output_path,
                overlay_text,
            )

            if success and output_path.exists():
                LOGGER.info(f"Created clip: {output_filename}")
                return output_path
            else:
                LOGGER.warning(f"Failed to create clip for {event_id}")
                return None

        except Exception as e:
            LOGGER.error(f"Error extracting clip for event: {e}")
            return None

    def insert_event_record(
        self, conn: sqlite3.Connection, event: pd.Series, clip_path: Optional[Path]
    ) -> None:
        """Insert event record into wwf_clips database."""
        try:
            event_id = str(event.get("event_id", ""))
            if not event_id:
                return

            # Build values tuple, matching the table schema
            values = (
                str(event.get("station", "")),
                str(event.get("video_id", "")),
                int(event.get("year", 0)) if event.get("year") else None,
                str(event.get("date", "")),
                str(event.get("filename", "")),
                event_id,
                str(event.get("event_type", "")),
                self._as_float(event.get("second")),
                self._as_float(event.get("before_mean")),
                self._as_float(event.get("after_mean")),
                self._as_int_bool(event.get("arrival_with_fish_stage2")),
                self._as_float(event.get("fish_count")),
                self._as_float(event.get("fish_mean_area")),
                self._as_float(event.get("fish_max_area")),
                str(event.get("absolute_timestamp", "")),
                str(event.get("original_video_path", "")),
                str(event.get("event_video_path", "")),
                str(event.get("detections_csv_path", "")),
                str(event.get("stage3_status", "")),
                self._as_int_bool(event.get("is_actual_arrival")),
                self._as_int_bool(event.get("is_new_fish_arrival")),
                int(event.get("fish_detections_stage4", 0))
                if event.get("fish_detections_stage4")
                else None,
                self._as_float(event.get("fish_avg_confidence_stage4")),
                str(event.get("stage4_rule_version", "")),
                str(event.get("stage4_rule_hits", "")),
                str(event.get("stage4_features", "")),
                self._as_float(event.get("stage4_model_score")),
                str(event.get("stage4_decision_source", "")),
                str(clip_path) if clip_path else None,
            )

            conn.execute(
                """
                INSERT INTO wwf_clips (
                    station, video_id, year, date, filename, event_id,
                    event_type, second, before_mean, after_mean,
                    arrival_with_fish_stage2, fish_count, fish_mean_area, fish_max_area,
                    absolute_timestamp, original_video_path, event_video_path,
                    detections_csv_path, stage3_status, is_actual_arrival,
                    is_new_fish_arrival, fish_detections_stage4, fish_avg_confidence_stage4,
                    stage4_rule_version, stage4_rule_hits, stage4_features,
                    stage4_model_score, stage4_decision_source, clip_path
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                values,
            )
        except Exception as e:
            LOGGER.error(f"Error inserting event record: {e}")

    def save_summary(self, events: pd.DataFrame, created_clips: int) -> None:
        """Save batch summary and event metadata."""
        summary = {
            "batch_name": self.batch_name,
            "created_at": datetime.now().isoformat(),
            "total_events_sampled": len(events),
            "clips_created": created_clips,
            "min_fish_detections_threshold": self.min_fish_detections,
            "clip_before_seconds": self.clip_before,
            "clip_after_seconds": self.clip_after,
        }

        # Save summary as JSON
        summary_path = self.metadata_dir / "batch_summary.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        LOGGER.info(f"Saved batch summary: {summary_path}")

        # Save events metadata as CSV
        events_csv = self.metadata_dir / "events_summary.csv"
        events.to_csv(events_csv, index=False)
        LOGGER.info(f"Saved events summary: {events_csv}")

    def run(self) -> bool:
        """Execute the full pipeline."""
        try:
            LOGGER.info(f"Starting WWF clips generation for batch {self.batch_name}")
            self.setup_directories()

            # Sample events
            sampled_events = self.sample_events()
            if sampled_events.empty:
                LOGGER.error("No events to process")
                return False

            # Initialize database
            db_conn = self.initialize_database()

            # Extract clips and populate database
            created_clips = 0
            processed_events = []

            for idx, (_, event) in enumerate(sampled_events.iterrows(), 1):
                LOGGER.info(
                    f"Processing event {idx}/{len(sampled_events)}: {event.get('event_id')}"
                )

                clip_path = self.extract_clip(event, self.video_dir)
                self.insert_event_record(db_conn, event, clip_path)

                if clip_path:
                    created_clips += 1
                    processed_events.append(event)

            db_conn.commit()
            db_conn.close()

            # Save summary
            self.save_summary(sampled_events, created_clips)

            LOGGER.info(
                f"Batch {self.batch_name} complete: {created_clips}/{len(sampled_events)} clips created"
            )
            print(f"\nBatch created successfully: {self.batch_dir}")
            print(f"Database: {self.db_path}")
            print(f"Clips created: {created_clips}/{len(sampled_events)}")

            return True

        except Exception as e:
            LOGGER.error(f"Error during batch generation: {e}", exc_info=True)
            return False

    @staticmethod
    def _as_float(value) -> Optional[float]:
        """Convert value to float or None."""
        if value is None:
            return None
        try:
            if pd.isna(value):
                return None
        except TypeError:
            pass
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _as_int_bool(value) -> Optional[int]:
        """Convert value to int boolean (0 or 1) or None."""
        if value is None:
            return None
        try:
            if pd.isna(value):
                return None
        except TypeError:
            pass
        return 1 if bool(value) else 0


def setup_logging(log_dir: Path) -> None:
    """Configure logging."""
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"wwf_clips_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout),
        ],
    )
    LOGGER.info(f"Logging to {log_file}")


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Generate video clips for WWF citizen science app"
    )
    parser.add_argument(
        "--num-events",
        type=int,
        default=50,
        help="Number of events to sample for this batch (default: 50)",
    )
    parser.add_argument(
        "--min-fish-detections",
        type=int,
        default=3,
        help="Minimum number of fish detections to include event (default: 3)",
    )
    parser.add_argument(
        "--events-db-root",
        type=Path,
        default=Path("data/events_db"),
        help="Root directory containing per-station event databases",
    )
    parser.add_argument(
        "--output-base",
        type=Path,
        default=NAS2_WORK_BASE,
        help="Base output directory on NAS2_work (default: /mnt/BSP_NAS2_work)",
    )
    parser.add_argument(
        "--clip-before",
        type=float,
        default=2.0,
        help="Seconds before event to include in clip (default: 2.0)",
    )
    parser.add_argument(
        "--clip-after",
        type=float,
        default=8.0,
        help="Seconds after event to include in clip (default: 8.0)",
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=Path("logs/wwf_clips"),
        help="Log directory (default: logs/wwf_clips)",
    )
    parser.add_argument(
        "--skip-stations",
        type=str,
        default="",
        help="Comma-separated list of station names to skip (e.g., ROST3,ROST4)",
    )
    parser.add_argument(
        "--new-fish-only",
        action="store_true",
        help="Only include events classified as new fish arrivals (is_new_fish_arrival=1)",
    )

    args = parser.parse_args()

    setup_logging(args.log_dir)

    skip_stations = [s.strip() for s in args.skip_stations.split(",") if s.strip()]

    generator = WWFClipsGenerator(
        events_db_root=args.events_db_root,
        output_base=args.output_base,
        num_events=args.num_events,
        min_fish_detections=args.min_fish_detections,
        clip_before=args.clip_before,
        clip_after=args.clip_after,
        skip_stations=skip_stations,
        new_fish_only=args.new_fish_only,
    )

    success = generator.run()
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
