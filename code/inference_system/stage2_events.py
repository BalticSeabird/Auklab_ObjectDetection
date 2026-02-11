"""Stage 2 processor: convert detection CSVs into ecological event summaries."""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional

import matplotlib
import pandas as pd

matplotlib.use("Agg")  # Ensure headless plotting

PROJECT_CODE_DIR = Path(__file__).resolve().parents[1]
CODE_ROOT = PROJECT_CODE_DIR.parent
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from postprocess.batch_analyze_days import (  # type: ignore  # noqa: E402
    combine_daily_results,
    generate_daily_summary_report,
    plot_daily_overview,
    process_single_file,
)

from .config_manager import Config
from .path_utils import get_event_csv_path, get_event_output_dir, get_model_display_name
from .state_manager import VideoJob
from .worker_pool import (
    PermanentError,
    ProcessingMetrics,
    ProcessingResult,
    RecoverableError,
    StageProcessor,
    WorkerContext,
)
from .path_utils import get_detection_csv_path

LOGGER = logging.getLogger(__name__)


class EventDetectionProcessor(StageProcessor):
    """Analyze detection CSVs to derive events, movement, and plots."""

    def __init__(self, config: Config) -> None:
        self.config = config
        self.stage_cfg = config.processing.stage2_event_detection
        self.model_label = get_model_display_name(config)
        self.logger = logging.getLogger("stage2")

    def process(self, job: VideoJob, context: WorkerContext) -> Optional[ProcessingResult]:  # noqa: ARG002
        detections_csv = get_detection_csv_path(self.config, job)
        if not detections_csv.exists():
            raise RecoverableError(f"Detections CSV missing for {job.video_id}: {detections_csv}")

        event_root = get_event_output_dir(self.config, job)
        csv_dir = event_root / "csv"
        plots_dir = event_root / "plots"
        csv_dir.mkdir(parents=True, exist_ok=True)
        plots_dir.mkdir(parents=True, exist_ok=True)

        event_csv = get_event_csv_path(self.config, job)
        if event_csv.exists():
            self.logger.info("[%s] events already generated", job.video_id)
            return ProcessingResult(
                metadata={"events_csv": str(event_csv), "skipped": True},
                metrics=ProcessingMetrics(events_count=0),
                duration_seconds=0.0,
            )

        result = process_single_file(
            str(detections_csv),
            fps=self.stage_cfg.fps,
            original_video_fps=self.stage_cfg.original_video_fps,
            conf_thresh=self.stage_cfg.confidence_threshold,
            smooth_window_s=self.stage_cfg.smooth_window_s,
            error_window_s=self.stage_cfg.error_window_s,
            hold_seconds=self.stage_cfg.hold_seconds,
            fish_window_s=self.stage_cfg.fish_window_s,
            movement_smoothing_s=self.stage_cfg.movement_smoothing_s,
            flap_area_multiplier=self.stage_cfg.flap_area_multiplier,
            flap_baseline_s=self.stage_cfg.flap_baseline_s,
        )

        if not result:
            self.logger.warning("[%s] No detections available for event analysis", job.video_id)
            columns = ["event_id", "type", "timestamp", "absolute_timestamp"]
            csv_dir.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(columns=columns).to_csv(event_csv, index=False)
            return ProcessingResult(
                metadata={"events_csv": str(event_csv), "events": 0},
                metrics=ProcessingMetrics(events_count=0),
            )

        combined = combine_daily_results([result], station=job.station, model_name=self.model_label)
        if not combined:
            raise PermanentError(f"Failed to combine event data for {job.video_id}")

        date_str = (combined["date"].strftime("%Y%m%d") if combined.get("date") else job.date.replace("-", ""))
        csv_outputs = self._write_csv_outputs(combined, csv_dir, date_str)
        summary_path = event_root / f"daily_summary_{date_str}.txt"
        generate_daily_summary_report(combined, summary_path)
        plot_daily_overview(combined, plots_dir, date_str)

        events_count = len(combined["events_df"]) if not combined["events_df"].empty else 0
        events_path = csv_outputs.get("events", event_csv)
        metadata = {
            "events_csv": str(events_path),
            "flaps_csv": str(csv_outputs.get("flaps", "")),
            "per_second_csv": str(csv_outputs.get("per_second", "")),
            "movement_csv": str(csv_outputs.get("movement", "")),
            "per_minute_csv": str(csv_outputs.get("per_minute", "")),
        }

        return ProcessingResult(
            metadata=metadata,
            metrics=ProcessingMetrics(events_count=events_count),
        )

    def _write_csv_outputs(self, combined_data, csv_dir: Path, date_str: str) -> dict:
        outputs = {}
        if not combined_data["events_df"].empty:
            events_path = csv_dir / f"daily_events_{date_str}.csv"
            combined_data["events_df"].to_csv(events_path, index=False)
            outputs["events"] = events_path
        if not combined_data["flaps_df"].empty:
            flaps_path = csv_dir / f"daily_flaps_{date_str}.csv"
            combined_data["flaps_df"].to_csv(flaps_path, index=False)
            outputs["flaps"] = flaps_path
        if not combined_data["per_second_df"].empty:
            per_second_path = csv_dir / f"daily_per_second_{date_str}.csv"
            combined_data["per_second_df"].to_csv(per_second_path, index=False)
            outputs["per_second"] = per_second_path
        if not combined_data["movement_df"].empty:
            movement_path = csv_dir / f"daily_movement_{date_str}.csv"
            combined_data["movement_df"].to_csv(movement_path, index=False)
            outputs["movement"] = movement_path
        if not combined_data["per_minute_df"].empty:
            per_minute_path = csv_dir / f"daily_per_minute_{date_str}.csv"
            combined_data["per_minute_df"].to_csv(per_minute_path, index=False)
            outputs["per_minute"] = per_minute_path
        return outputs
