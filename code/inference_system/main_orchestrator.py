"""Entry point for the multi-stage Auklab inference pipeline."""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import List, Optional

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    __package__ = "inference_system"

from .config_manager import load_config
from .job_scheduler import JobScheduler
from .stage1_inference import VideoInferenceProcessor
from .stage2_events import EventDetectionProcessor
from .state_manager import ProcessingStage, StateManager
from .worker_pool import WorkerPoolManager


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Auklab multi-stage inference system")
    parser.add_argument("--config", type=Path, default=Path("config/system_config.yaml"), help="Path to YAML config")
    parser.add_argument("--stations", nargs="*", help="Optional subset of stations to process")
    parser.add_argument("--resume", action="store_true", help="Resume from existing state without rediscovery")
    parser.add_argument("--discover-only", action="store_true", help="Discover videos and exit")
    parser.add_argument(
        "--skip-discovery",
        action="store_true",
        help="Start workers without refreshing the video registry",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Root logging level",
    )
    parser.add_argument(
        "--stuck-timeout",
        type=int,
        default=3600,
        help="Seconds before in-progress jobs are considered stuck during resume",
    )
    args = parser.parse_args(argv)

    if args.skip_discovery and args.discover_only:
        parser.error("--skip-discovery cannot be combined with --discover-only")

    return args


def configure_logging(level: str, log_dir: Path | None = None) -> None:
    """Configure root logging for console and optional file output.

    When log_dir is provided, a main.log file will be created there and all
    logs from the inference system (including workers) will be written to it
    in addition to the console.
    """

    log_level = getattr(logging, level.upper(), logging.INFO)

    # Always log to the console
    formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] %(name)s: %(message)s")
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    handlers = [console_handler]

    # Optional file handler rooted at config.paths.log_dir
    if log_dir is not None:
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_dir / "main.log")
        file_handler.setFormatter(formatter)
        handlers.append(file_handler)

    logging.basicConfig(level=log_level, handlers=handlers)


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)

    # Load config first so we know where to place log files.
    config = load_config(args.config)
    configure_logging(args.log_level, config.paths.log_dir)

    logging.info("Starting inference orchestrator with config %s", args.config)
    state_mgr = StateManager(config.paths.state_db)
    state_mgr.initialize_db()

    if args.resume:
        logging.info("Resetting stuck jobs older than %ss", args.stuck_timeout)
        reset_count = state_mgr.reset_stuck_jobs(timeout_seconds=args.stuck_timeout)
        logging.info("Reset %s stuck stage entries", reset_count)
        failed_reset = state_mgr.reset_failed_jobs(stage=ProcessingStage.STAGE2)
        logging.info("Reset %s failed stage2 entries for retry", failed_reset)

    scheduler = JobScheduler(
        config,
        state_mgr,
        allowed_stations=args.stations,
    )

    if args.skip_discovery:
        logging.info("Skipping video discovery; using existing registry state")
    else:
        discovered = scheduler.discover_videos()
        logging.info("Discovered or refreshed %s video entries", discovered)

    # Always recalculate priorities from current config, even for videos that
    # were already present in the database from previous runs.
    scheduler.calculate_priorities()

    if args.discover_only:
        logging.info("Discovery-only mode complete")
        return

    processors = {
        ProcessingStage.STAGE1: VideoInferenceProcessor(config),
        ProcessingStage.STAGE2: EventDetectionProcessor(config),
    }

    worker_pool = WorkerPoolManager(config, state_mgr, scheduler, processors)
    worker_pool.start_workers()

    try:
        while True:
            summary = state_mgr.get_progress_summary()
            logging.info(
                "Progress: %s/%s completed, %s pending, %s in progress, %s failed",
                summary.completed_videos,
                summary.total_videos,
                summary.pending_videos,
                summary.in_progress_videos,
                summary.failed_videos,
            )
            if state_mgr.is_all_complete():
                break
            time.sleep(config.monitoring.update_interval_seconds)
    except KeyboardInterrupt:
        logging.info("Interrupted by user, shutting down workers")
    finally:
        worker_pool.stop_workers(graceful=True)

    logging.info("Processing complete")


if __name__ == "__main__":
    main(sys.argv[1:])
