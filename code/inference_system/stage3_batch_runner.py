"""Standalone entry-point for running clip extraction (stage3) in batch mode."""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import pandas as pd

from .config_manager import Config, load_config
from .path_utils import get_model_display_name
from .stage3_clips import ClipExtractionProcessor
from .stage3_state_manager import (
    Stage3BatchRecord,
    Stage3BatchRegistration,
    Stage3BatchStatus,
    Stage3StateManager,
)
from .state_manager import StateManager, VideoJob


@dataclass(slots=True)
class Stage3BatchMetrics:
    clips_created: int = 0
    clips_failed: int = 0
    videos_processed: int = 0
    events_processed: int = 0


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run clip extraction batches grouped by station/date")
    parser.add_argument("--config", type=Path, default=Path("config/system_config.yaml"), help="Path to YAML config")
    parser.add_argument("--stations", nargs="*", help="Optional list of stations to include")
    parser.add_argument("--start-date", dest="start_date", help="Minimum date (YYYY-MM-DD)")
    parser.add_argument("--end-date", dest="end_date", help="Maximum date (YYYY-MM-DD)")
    parser.add_argument("--limit", type=int, help="Limit the number of batches to run")
    parser.add_argument("--force", action="store_true", help="Re-run completed batches")
    parser.add_argument("--retry-failed", action="store_true", help="Retry batches marked as failed")
    parser.add_argument("--discover-only", action="store_true", help="Only refresh the batch registry")
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
    stage3_state = Stage3StateManager(config.paths.stage3_state_db)
    stage3_state.initialize_db()

    inference_state = StateManager(config.paths.state_db)
    inference_state.initialize_db()

    allowed_stations = [station.upper() for station in args.stations] if args.stations else None
    start_date, end_date = _parse_date_bounds(args.start_date, args.end_date)

    discovered = _discover_batches(config, allowed_stations, start_date, end_date)
    if discovered:
        registered = stage3_state.register_batches(discovered)
        logging.info("Registered %s stage3 batch(es)", registered)
    else:
        logging.info("No new stage3 batches discovered")

    if args.discover_only:
        return

    statuses: List[Stage3BatchStatus] = [Stage3BatchStatus.PENDING]
    if args.retry_failed:
        statuses.append(Stage3BatchStatus.FAILED)
    if args.force:
        statuses.append(Stage3BatchStatus.COMPLETED)

    batches = stage3_state.fetch_batches(
        statuses=statuses,
        stations=allowed_stations,
        start_date=start_date.strftime("%Y-%m-%d") if start_date else None,
        end_date=end_date.strftime("%Y-%m-%d") if end_date else None,
        limit=args.limit,
    )
    if not batches:
        logging.info("No stage3 batches matched the provided filters")
        return

    processor = Stage3BatchProcessor(config, inference_state)
    for batch in batches:
        if args.force and batch.status == Stage3BatchStatus.COMPLETED:
            stage3_state.reset_batch(batch.station, batch.date)
        stage3_state.mark_batch_started(batch.station, batch.date)
        try:
            metrics = processor.process_batch(batch)
        except Exception as exc:  # pragma: no cover - orchestration guard
            logging.exception("Stage3 batch %s %s failed", batch.station, batch.date)
            stage3_state.mark_batch_failed(batch.station, batch.date, error_message=str(exc))
            continue
        stage3_state.mark_batch_completed(
            batch.station,
            batch.date,
            clips_created=metrics.clips_created,
            clips_failed=metrics.clips_failed,
        )
        logging.info(
            "Stage3 batch %s %s -> %s clips (%s failed) across %s video(s)",
            batch.station,
            batch.date,
            metrics.clips_created,
            metrics.clips_failed,
            metrics.videos_processed,
        )


class Stage3BatchProcessor:
    """Coordinates clip extraction for a single station/date batch."""

    def __init__(self, config: Config, inference_state: StateManager) -> None:
        self.config = config
        self.inference_state = inference_state
        self.clip_processor = ClipExtractionProcessor(config)
        self.logger = logging.getLogger("stage3.processor")

    def process_batch(self, batch: Stage3BatchRecord) -> Stage3BatchMetrics:
        events_csv = batch.events_csv
        if not events_csv.exists():
            raise FileNotFoundError(f"Events CSV missing for {batch.station} {batch.date}: {events_csv}")

        events_df = pd.read_csv(events_csv)
        filtered_events = self.clip_processor.filter_events(events_df)
        if filtered_events.empty:
            self.logger.info("[%s %s] No eligible events", batch.station, batch.date)
            return Stage3BatchMetrics()

        grouped_events = self._group_events_by_source(filtered_events)
        if not grouped_events:
            self.logger.info("[%s %s] No source groups found", batch.station, batch.date)
            return Stage3BatchMetrics()

        available_jobs = self.inference_state.get_videos_for_station_date(batch.station, batch.date)
        job_index = self._build_job_index(available_jobs)
        if not job_index:
            raise RuntimeError(f"No video metadata registered for {batch.station} on {batch.date}")

        total_created = 0
        total_failed = 0
        videos_processed = 0
        for source_key, subset in grouped_events.items():
            job = job_index.get(source_key)
            if job is None:
                self.logger.warning(
                    "[%s %s] Missing video metadata for observation '%s'", batch.station, batch.date, source_key
                )
                continue
            videos_processed += 1
            result = self.clip_processor.process_events_for_job(job, subset, event_csv_path=events_csv)
            metadata = result.metadata or {}
            total_created += int(metadata.get("clips_created", 0) or 0)
            total_failed += int(metadata.get("clips_failed", 0) or 0)

        return Stage3BatchMetrics(
            clips_created=total_created,
            clips_failed=total_failed,
            videos_processed=videos_processed,
            events_processed=len(filtered_events),
        )

    def _group_events_by_source(self, events_df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        if events_df.empty:
            return {}
        working = events_df.copy()
        if "file" in working.columns:
            working["_source_key"] = working["file"].astype(str).str.upper()
        elif "observation_period" in working.columns:
            working["_source_key"] = (
                working["observation_period"]
                .astype(str)
                .str.replace(".csv", "", regex=False)
                .str.replace("_raw", "", regex=False)
                .str.upper()
            )
        else:
            working["_source_key"] = "UNKNOWN"

        grouped: Dict[str, pd.DataFrame] = {}
        for source_key, subset in working.groupby("_source_key"):
            key = str(source_key).upper()
            if key in ("", "NAN"):
                continue
            grouped[key] = subset.drop(columns="_source_key")
        return grouped

    @staticmethod
    def _build_job_index(jobs: Sequence[VideoJob]) -> Dict[str, VideoJob]:
        index: Dict[str, VideoJob] = {}
        for job in jobs:
            key = Path(job.filename).stem.upper()
            index[key] = job
        return index


def _discover_batches(
    config: Config,
    allowed_stations: Optional[Sequence[str]],
    start_date: Optional[date],
    end_date: Optional[date],
) -> List[Stage3BatchRegistration]:
    root = Path(config.paths.event_analysis_output)
    model_folder = get_model_display_name(config)
    batches: List[Stage3BatchRegistration] = []
    if not root.exists():
        logging.warning("Event analysis root does not exist: %s", root)
        return batches

    allowed = {station.upper() for station in allowed_stations} if allowed_stations else None

    for year_dir in sorted(root.iterdir()):
        if not year_dir.is_dir() or not year_dir.name.isdigit():
            continue
        model_dir = year_dir / model_folder
        if not model_dir.exists():
            continue
        for station_dir in sorted(model_dir.iterdir()):
            if not station_dir.is_dir():
                continue
            station = station_dir.name.upper()
            if allowed and station not in allowed:
                continue
            for date_dir in sorted(station_dir.iterdir()):
                if not date_dir.is_dir():
                    continue
                try:
                    parsed_date = datetime.strptime(date_dir.name, "%Y%m%d").date()
                except ValueError:
                    continue
                if start_date and parsed_date < start_date:
                    continue
                if end_date and parsed_date > end_date:
                    continue
                csv_path = date_dir / "csv" / f"daily_events_{date_dir.name}.csv"
                if not csv_path.exists():
                    continue
                batches.append(
                    Stage3BatchRegistration(
                        station=station,
                        date=parsed_date.strftime("%Y-%m-%d"),
                        year=parsed_date.year,
                        events_csv=csv_path,
                    )
                )
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
