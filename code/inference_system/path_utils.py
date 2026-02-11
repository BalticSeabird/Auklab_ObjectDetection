"""Helper utilities for deriving canonical file-system paths used by the pipeline."""

from __future__ import annotations

from pathlib import Path

from .config_manager import Config
from .state_manager import VideoJob


def _model_identifier(config: Config) -> str:
    return Path(config.paths.detection_model).stem


def _compact_date(job: VideoJob) -> str:
    return job.date.replace("-", "")


def get_detection_output_dir(config: Config, job: VideoJob) -> Path:
    return Path(config.paths.inference_output) / str(job.year) / _model_identifier(config) / job.station


def get_detection_csv_path(config: Config, job: VideoJob) -> Path:
    return get_detection_output_dir(config, job) / f"{job.filepath.stem}_raw.csv"


def get_event_output_dir(config: Config, job: VideoJob) -> Path:
    return (
        Path(config.paths.event_analysis_output)
        / str(job.year)
        / _model_identifier(config)
        / job.station
        / _compact_date(job)
    )


def get_event_csv_path(config: Config, job: VideoJob) -> Path:
    return get_event_output_dir(config, job) / "csv" / f"daily_events_{_compact_date(job)}.csv"


def get_clips_output_dir(config: Config, job: VideoJob, subfolder: str) -> Path:
    return Path(config.paths.clips_output) / job.station / _compact_date(job) / subfolder


def get_clips_date_root(config: Config, job: VideoJob) -> Path:
    return Path(config.paths.clips_output) / job.station / _compact_date(job)


def get_model_display_name(config: Config) -> str:
    """Return a short identifier for logging and filenames."""
    return _model_identifier(config)
