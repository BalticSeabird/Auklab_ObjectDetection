"""Configuration management for the integrated inference system."""

from __future__ import annotations

import argparse
import os
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from pydantic import BaseModel, Field, NonNegativeInt, PositiveInt, ValidationError, field_validator, model_validator


def _expand_env_vars(value: Any) -> Any:
    """Recursively expand environment variables and user-home markers inside the config tree."""
    if isinstance(value, str):
        return os.path.expandvars(os.path.expanduser(value))
    if isinstance(value, list):
        return [_expand_env_vars(item) for item in value]
    if isinstance(value, dict):
        return {key: _expand_env_vars(val) for key, val in value.items()}
    return value


class SystemSettings(BaseModel):
    name: str
    version: str


class GPUSettings(BaseModel):
    count: PositiveInt
    device_ids: List[NonNegativeInt]

    @model_validator(mode="after")
    def validate_device_ids(self) -> "GPUSettings":
        if len(self.device_ids) < self.count:
            msg = "device_ids must include at least as many entries as the configured GPU count"
            raise ValueError(msg)
        return self


class CPUSettings(BaseModel):
    worker_count: PositiveInt


class HardwareSettings(BaseModel):
    gpus: GPUSettings
    cpus: CPUSettings


class PathsConfig(BaseModel):
    video_base: Optional[Path] = None
    video_roots: Optional[List[Path]] = None
    inference_output: Path
    event_analysis_output: Path
    clips_output: Path
    detection_model: Path
    state_db: Path
    stage3_state_db: Path
    log_dir: Path

    @model_validator(mode="after")
    def validate_sources(self) -> "PathsConfig":
        if not self.video_sources:
            msg = "At least one video root must be defined (video_base or video_roots)."
            raise ValueError(msg)
        return self

    @property
    def video_sources(self) -> List[Path]:
        if self.video_roots and len(self.video_roots) > 0:
            return list(self.video_roots)
        if self.video_base:
            return [self.video_base]
        return []


class Stage1Config(BaseModel):
    frame_skip: PositiveInt
    batch_size: PositiveInt
    confidence_threshold: float = Field(ge=0.0, le=1.0)


class Stage2Config(BaseModel):
    fps: PositiveInt
    original_video_fps: PositiveInt
    confidence_threshold: float = Field(default=0.25, ge=0.0, le=1.0)
    smooth_window_s: PositiveInt
    error_window_s: PositiveInt
    hold_seconds: PositiveInt
    fish_window_s: PositiveInt
    movement_smoothing_s: PositiveInt
    flap_area_multiplier: float = Field(gt=0)
    flap_baseline_s: PositiveInt


class CompressionConfig(BaseModel):
    enabled: bool = True
    crf: NonNegativeInt = Field(le=51)
    preset: str = Field(default="fast")


class Stage3Config(BaseModel):
    clip_before: NonNegativeInt
    clip_after: NonNegativeInt
    event_types: List[str] = Field(default_factory=lambda: ["arrival", "departure"])
    fish_only: bool = False
    video_extensions: List[str] = Field(default_factory=lambda: [".mkv", ".mp4", ".avi"])
    compression: CompressionConfig = Field(default_factory=CompressionConfig)


class ProcessingConfig(BaseModel):
    stage1_video_inference: Stage1Config
    stage2_event_detection: Stage2Config
    stage3_clip_extraction: Stage3Config


class PriorityConfig(BaseModel):
    years: List[int]
    stations: List[str]


class DateRangeFilter(BaseModel):
    start: Optional[date] = None
    end: Optional[date] = None

    @field_validator("start", "end", mode="before")
    @classmethod
    def parse_date(cls, value: Optional[str]) -> Optional[date]:
        if value in (None, "", "null"):
            return None
        return datetime.strptime(value, "%Y-%m-%d").date()

    @model_validator(mode="after")
    def validate_range(self) -> "DateRangeFilter":
        if self.start and self.end and self.end < self.start:
            msg = "End date must be greater than or equal to start date"
            raise ValueError(msg)
        return self


class FilterConfig(BaseModel):
    date_range: Optional[DateRangeFilter] = None


class ErrorHandlingConfig(BaseModel):
    max_retries: NonNegativeInt
    retry_delay_seconds: PositiveInt
    skip_on_persistent_failure: bool = True


class MonitoringConfig(BaseModel):
    update_interval_seconds: PositiveInt
    log_performance_metrics: bool = True


class ResumeConfig(BaseModel):
    enabled: bool = True
    checkpoint_interval_seconds: PositiveInt


class Config(BaseModel):
    system: SystemSettings
    hardware: HardwareSettings
    paths: PathsConfig
    processing: ProcessingConfig
    priorities: PriorityConfig
    filters: Optional[FilterConfig] = None
    error_handling: ErrorHandlingConfig
    monitoring: MonitoringConfig
    resume: ResumeConfig

    @property
    def station_priority(self) -> List[str]:
        return list(self.priorities.stations)

    @property
    def year_priority(self) -> List[int]:
        return list(self.priorities.years)

    @property
    def video_sources(self) -> List[Path]:
        return self.paths.video_sources


class ConfigManager:
    """Helper responsible for loading and validating configuration files."""

    def __init__(self, config_path: Optional[Path] = None, config: Optional[Config] = None):
        self._config_path = Path(config_path) if config_path else None
        self._config = config

    @property
    def config(self) -> Config:
        if self._config is None:
            raise RuntimeError("Configuration has not been loaded")
        return self._config

    def load_config(self, config_path: Optional[Path] = None) -> Config:
        path = config_path or self._config_path
        if path is None:
            raise ValueError("Configuration path is not set")
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Configuration file not found: {path}")
        raw = path.read_text(encoding="utf-8")
        data = yaml.safe_load(raw) or {}
        expanded = _expand_env_vars(data)
        config = Config.model_validate(expanded)
        self._config = config
        self._config_path = path
        return config

    def validate_config(self) -> bool:
        _ = self.config  # Validation already occurs during load
        return True

    def get_station_priority(self) -> List[str]:
        return self.config.station_priority

    def get_processing_params(self) -> ProcessingConfig:
        return self.config.processing


def load_config(config_path: Path | str) -> Config:
    manager = ConfigManager(Path(config_path))
    return manager.load_config()


def generate_default_config() -> Dict[str, Any]:
    return {
        "system": {"name": "Auklab Video Inference System", "version": "1.0"},
        "hardware": {
            "gpus": {"count": 2, "device_ids": [0, 1]},
            "cpus": {"worker_count": 16},
        },
        "paths": {
            "video_roots": [
                "/mnt/BSP_NAS2_vol4/Video",
            ],
            "inference_output": "/mnt/BSP_NAS2_work/auklab_model/inference",
            "event_analysis_output": "/mnt/BSP_NAS2_work/auklab_model/summarized_inference",
            "clips_output": "/mnt/BSP_NAS2_work/auklab_model/event_data",
            "detection_model": "models/auklab_model_xlarge_combined_6080_v1.pt",
            "state_db": "data/processing_state.db",
            "stage3_state_db": "data/stage3_processing_state.db",
            "log_dir": "logs",
        },
        "processing": {
            "stage1_video_inference": {"frame_skip": 25, "batch_size": 32, "confidence_threshold": 0.25},
            "stage2_event_detection": {
                "fps": 1,
                "original_video_fps": 25,
                "confidence_threshold": 0.25,
                "smooth_window_s": 3,
                "error_window_s": 10,
                "hold_seconds": 8,
                "fish_window_s": 5,
                "movement_smoothing_s": 5,
                "flap_area_multiplier": 3.0,
                "flap_baseline_s": 30,
            },
            "stage3_clip_extraction": {
                "clip_before": 5,
                "clip_after": 10,
                "event_types": ["arrival", "departure"],
                "fish_only": False,
                "video_extensions": [".mkv", ".mp4", ".avi"],
                "compression": {"enabled": True, "crf": 28, "preset": "fast"},
            },
        },
        "priorities": {
            "years": [2025, 2024, 2023],
            "stations": ["BONDEN3", "BONDEN6", "TRI3", "FAR3", "FAR6", "ROST2", "ROST6"],
        },
        "filters": {"date_range": {"start": "2025-01-01", "end": "2025-12-31"}},
        "error_handling": {
            "max_retries": 2,
            "retry_delay_seconds": 60,
            "skip_on_persistent_failure": True,
        },
        "monitoring": {"update_interval_seconds": 60, "log_performance_metrics": True},
        "resume": {"enabled": True, "checkpoint_interval_seconds": 300},
    }


def dump_default_config(target_path: Optional[Path] = None) -> str:
    config_dict = generate_default_config()
    yaml_str = yaml.safe_dump(config_dict, sort_keys=False)
    if target_path:
        target_path.parent.mkdir(parents=True, exist_ok=True)
        target_path.write_text(yaml_str, encoding="utf-8")
    return yaml_str


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Configuration manager utility")
    parser.add_argument("--config", type=Path, default=Path("config/system_config.yaml"), help="Path to config file")
    parser.add_argument("--generate-default", action="store_true", help="Print a default configuration template")
    parser.add_argument("--output", type=Path, help="Destination when generating a default config")
    parser.add_argument("--validate", action="store_true", help="Validate the specified config file")
    return parser


def main() -> None:
    parser = _build_arg_parser()
    args = parser.parse_args()

    if args.generate_default:
        content = dump_default_config(args.output)
        if args.output is None:
            print(content)
        return

    manager = ConfigManager(args.config)
    try:
        manager.load_config()
    except (FileNotFoundError, ValidationError, ValueError) as exc:
        raise SystemExit(f"Failed to load configuration: {exc}") from exc

    if args.validate:
        print(f"Configuration '{args.config}' is valid")


if __name__ == "__main__":
    main()
