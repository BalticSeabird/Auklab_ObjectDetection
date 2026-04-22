"""Standalone entry-point for running stage4 post-classification in batch mode."""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import pandas as pd

from .config_manager import Config, load_config
from .event_database import EventDatabase
from .stage4_postclassification import Stage4PostClassificationProcessor
from .state_manager import StateManager, VideoJob


@dataclass(slots=True)
class Stage4BatchMetrics:
    events_classified: int = 0
    events_failed: int = 0
    videos_processed: int = 0


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Re-run stage4 post-classification on events grouped by station/date"
    )
    parser.add_argument("--config", type=Path, default=Path("config/system_config.yaml"), help="Path to YAML config")
    parser.add_argument("--stations", nargs="*", help="Optional list of stations to include")
    parser.add_argument("--start-date", dest="start_date", help="Minimum date (YYYY-MM-DD)")
    parser.add_argument("--end-date", dest="end_date", help="Maximum date (YYYY-MM-DD)")
    parser.add_argument("--limit", type=int, help="Limit the number of station/date pairs to process")
    parser.add_argument("--force", action="store_true", help="Re-run already classified events")
    parser.add_argument("--retry-failed", action="store_true", help="Retry events that previously failed")
    parser.add_argument("--discover-only", action="store_true", help="Only list station/date pairs without processing")
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Root logging level",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO), format="[%(levelname)s] %(message)s")

    config = load_config(args.config)
    inference_state = StateManager(config.paths.state_db)
    inference_state.initialize_db()

    allowed_stations = [station.upper() for station in args.stations] if args.stations else None
    start_date, end_date = _parse_date_bounds(args.start_date, args.end_date)

    # Discover all station/date pairs from event databases
    batches = _discover_batches(config, allowed_stations, start_date, end_date, args.limit)
    if not batches:
        logging.info("No station/date pairs found matching the provided filters")
        return

    logging.info("Discovered %s station/date pair(s)", len(batches))
    if args.discover_only:
        for station, date_str in batches:
            logging.info("  %s on %s", station, date_str)
        return

    processor = Stage4BatchProcessor(config, inference_state)
    total_metrics = Stage4BatchMetrics()

    for station, date_str in batches:
        try:
            metrics = processor.process_batch(station, date_str, force=args.force, retry_failed=args.retry_failed)
            total_metrics.events_classified += metrics.events_classified
            total_metrics.events_failed += metrics.events_failed
            total_metrics.videos_processed += metrics.videos_processed
            logging.info(
                "Stage4 batch %s %s -> %s events classified (%s failed) across %s video(s)",
                station,
                date_str,
                metrics.events_classified,
                metrics.events_failed,
                metrics.videos_processed,
            )
        except Exception as exc:  # pragma: no cover - orchestration guard
            logging.exception("Stage4 batch %s %s failed", station, date_str)

    logging.info(
        "Stage4 batch processing complete: %s total events classified, %s failed",
        total_metrics.events_classified,
        total_metrics.events_failed,
    )


class Stage4BatchProcessor:
    """Coordinates post-classification for a single station/date batch."""

    def __init__(self, config: Config, inference_state: StateManager) -> None:
        self.config = config
        self.inference_state = inference_state
        self.classifier = Stage4PostClassificationProcessor(config)
        self.logger = logging.getLogger("stage4.processor")

    def process_batch(
        self, station: str, date_str: str, force: bool = False, retry_failed: bool = False
    ) -> Stage4BatchMetrics:
        """Process all events for a station/date pair."""
        event_db = EventDatabase.for_station(self.config, station)
        event_db.initialize()

        # Fetch events for this station/date, optionally filtering by classification status
        query_date = datetime.strptime(date_str, "%Y-%m-%d").date()
        events_df = self._fetch_events_for_date(event_db, station, query_date, force=force, retry_failed=retry_failed)

        if events_df.empty:
            self.logger.info("[%s %s] No events to classify", station, date_str)
            return Stage4BatchMetrics()

        # Group by video_id to get a job for each video
        videos_by_id = {}
        for video_id in events_df["video_id"].unique():
            job = self.inference_state.get_video(video_id)
            if job is None:
                self.logger.warning("[%s %s] No video metadata for %s", station, date_str, video_id)
                continue
            videos_by_id[video_id] = job

        if not videos_by_id:
            self.logger.info("[%s %s] No matching video metadata found", station, date_str)
            return Stage4BatchMetrics()

        classified = 0
        failed = 0
        for video_id, job in videos_by_id.items():
            video_events = events_df[events_df["video_id"] == video_id]
            for _, event in video_events.iterrows():
                event_id = str(event.get("event_id", ""))
                if not event_id:
                    continue

                try:
                    labels = self.classifier._classify_event(event)
                    event_db.update_stage4_labels(
                        station=station,
                        event_id=event_id,
                        is_actual_arrival=labels["is_actual_arrival"],
                        is_new_fish_arrival=labels["is_new_fish_arrival"],
                        fish_detections_stage4=labels["fish_detections_stage4"],
                        fish_avg_confidence_stage4=labels["fish_avg_confidence_stage4"],
                        rule_version=self.classifier.stage_cfg.rule_version,
                        rule_hits=labels["rule_hits"],
                        features=labels["features"],
                        model_score=labels.get("model_score"),
                        decision_source=str(labels.get("decision_source", "rules")),
                    )
                    classified += 1
                except Exception as exc:  # pragma: no cover
                    self.logger.exception("Failed to classify event %s: %s", event_id, exc)
                    failed += 1

        return Stage4BatchMetrics(
            events_classified=classified,
            events_failed=failed,
            videos_processed=len(videos_by_id),
        )

    def _fetch_events_for_date(
        self, event_db: EventDatabase, station: str, query_date: date, force: bool = False, retry_failed: bool = False
    ) -> pd.DataFrame:
        """Fetch events for a station on a given date, optionally filtering by classification status."""
        with event_db._connect() as conn:
            date_str = query_date.strftime("%Y-%m-%d")
            base_query = "SELECT * FROM events WHERE station = ? AND date = ?"
            params: List[str] = [station, date_str]

            if not force and not retry_failed:
                # Default: only unclassified events (stage4_rule_version is NULL)
                base_query += " AND stage4_rule_version IS NULL"
            elif retry_failed and not force:
                # Retry only events that have been classified (stage4_rule_version is NOT NULL)
                base_query += " AND stage4_rule_version IS NOT NULL"
            # If force=True, no filtering - reprocess everything

            base_query += " ORDER BY COALESCE(second, 0) ASC, event_id ASC"
            return pd.read_sql_query(base_query, conn, params=params)


def _discover_batches(
    config: Config,
    allowed_stations: Optional[Sequence[str]],
    start_date: Optional[date],
    end_date: Optional[date],
    limit: Optional[int],
) -> List[tuple[str, str]]:
    """Discover all station/date pairs with events in the event database."""
    events_db_root = Path(config.paths.events_db_root)
    if not events_db_root.exists():
        logging.warning("Event database root does not exist: %s", events_db_root)
        return []

    batches: List[tuple[str, str]] = []
    allowed_patterns = [station.upper() for station in allowed_stations] if allowed_stations else []
    ignored_patterns: List[str] = []
    if config.filters and getattr(config.filters, "ignored_stations", None):
        ignored_patterns = [s.upper() for s in config.filters.ignored_stations]

    for db_file in sorted(events_db_root.glob("*_events.db")):
        # Extract station name from filename (e.g., "ROST5_events.db" -> "ROST5")
        station = db_file.stem.replace("_events", "").upper()

        # Apply ignore list first
        if any(p and p in station for p in ignored_patterns):
            continue
        # Then apply optional allow filters using substring matching
        if allowed_patterns and not any(p and p in station for p in allowed_patterns):
            continue

        # Query the database for distinct dates
        try:
            event_db = EventDatabase(db_file)
            with event_db._connect() as conn:
                cursor = conn.execute(
                    "SELECT DISTINCT date FROM events WHERE date IS NOT NULL ORDER BY date DESC"
                )
                for (date_str,) in cursor.fetchall():
                    try:
                        parsed_date = datetime.strptime(date_str, "%Y-%m-%d").date()
                    except (ValueError, TypeError):
                        continue
                    if start_date and parsed_date < start_date:
                        continue
                    if end_date and parsed_date > end_date:
                        continue
                    batches.append((station, date_str))
                    if limit and len(batches) >= limit:
                        return batches
        except Exception as exc:  # pragma: no cover
            logging.warning("Failed to query event database %s: %s", db_file, exc)

    return batches


def _parse_date_bounds(start: Optional[str], end: Optional[str]) -> tuple[Optional[date], Optional[date]]:
    parsed_start = _parse_date(start) if start else None
    parsed_end = _parse_date(end) if end else None
    if parsed_start and parsed_end and parsed_end < parsed_start:
        raise ValueError("end-date must be greater than or equal to start-date")
    return parsed_start, parsed_end


def _parse_date(date_str: str) -> date:
    return datetime.strptime(date_str, "%Y-%m-%d").date()


if __name__ == "__main__":
    main()
