"""Stage 2 processor: detect events per video and extract annotated clips.

Each invocation operates on a single video's detection CSV and produces:
  - A per-video events CSV  ({video_stem}_events.csv)
  - Annotated mp4 clips for each detected event

Daily aggregation / summary reports are intentionally NOT performed here.
They belong to a separate, optional post-processing step once all videos for
a given day have been processed.
"""

from __future__ import annotations

import logging
import re
import sys
from datetime import timedelta
from pathlib import Path
from typing import Optional

import pandas as pd

PROJECT_CODE_DIR = Path(__file__).resolve().parents[1]
CODE_ROOT = PROJECT_CODE_DIR.parent
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from postprocess.batch_analyze_days import process_single_file  # type: ignore  # noqa: E402
from postprocess.extract_event_clips import format_event_filename  # type: ignore  # noqa: E402

from .config_manager import Config
from .event_database import EventDatabase
from .path_utils import get_clips_output_dir, get_detection_csv_path, get_station_event_db_path
from .state_manager import VideoJob
from .worker_pool import (
    ProcessingMetrics,
    ProcessingResult,
    RecoverableError,
    StageProcessor,
    WorkerContext,
)

LOGGER = logging.getLogger(__name__)

_EMPTY_EVENT_COLUMNS = [
    "event_id", "type", "second", "before_mean", "after_mean",
    "arrival_with_fish", "fish_count", "fish_mean_area", "fish_max_area",
    "absolute_timestamp", "event_video_path", "original_video_path",
]


class EventDetectionProcessor(StageProcessor):
    """Detect events for a single video and persist them to station SQLite."""

    def __init__(self, config: Config) -> None:
        self.config = config
        self.stage_cfg = config.processing.stage2_event_detection
        self.logger = logging.getLogger("stage2")
        self.model_name = self._derive_model_name(config)

    def process(self, job: VideoJob, context: WorkerContext) -> Optional[ProcessingResult]:  # noqa: ARG002
        detections_csv = get_detection_csv_path(self.config, job)
        if not detections_csv.exists():
            raise RecoverableError(f"Detections CSV missing for {job.video_id}: {detections_csv}")

        events_df = self._detect_events(job, detections_csv)
        event_db = EventDatabase.for_station(self.config, job.station)
        event_db.initialize()
        persisted_rows = event_db.upsert_events(station=job.station, job=job, events_df=events_df)

        events_count = len(events_df)

        return ProcessingResult(
            metadata={
                "events_db": str(get_station_event_db_path(self.config, job.station)),
                "events_count": events_count,
                "persisted_rows": persisted_rows,
            },
            metrics=ProcessingMetrics(events_count=events_count),
        )

    # ──────────────────────────────────────────────────────────────────────

    def _detect_events(
        self,
        job: VideoJob,
        detections_csv: Path,
    ) -> pd.DataFrame:
        """Run event detection and build a per-video event dataframe."""

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
            return pd.DataFrame(columns=_EMPTY_EVENT_COLUMNS)

        events = result.get("events", [])
        if not events:
            self.logger.info("[%s] No events detected", job.video_id)
            return pd.DataFrame(columns=_EMPTY_EVENT_COLUMNS)

        start_time = result.get("start_time")
        rows = []
        for event in events:
            row = dict(event)
            if start_time and "second" in event:
                row["absolute_timestamp"] = start_time + timedelta(seconds=int(event["second"]))
            rows.append(row)

        events_df = pd.DataFrame(rows)

        # Attach stable, human-readable event identifiers used for filenames.
        # Preferred format: {model}_{station}_{YYYYMMDD}_{HHMMSS}_{arr|dep}
        # Fallback (when timestamps cannot be parsed from filename):
        #   {model}_{station}_{video_stem}_s{second:06d}_{arr|dep}
        if not events_df.empty and "type" in events_df.columns:
            def _build_event_id(row: pd.Series) -> str:
                event_type_short = "arr" if row.get("type") == "arrival" else "dep"
                station_part = job.station if job.station else "station"

                if "absolute_timestamp" in events_df.columns and pd.notna(row.get("absolute_timestamp")):
                    try:
                        ts = pd.to_datetime(row["absolute_timestamp"])
                        timestamp_str = ts.strftime("%Y%m%d_%H%M%S")
                        return f"{self.model_name}_{station_part}_{timestamp_str}_{event_type_short}"
                    except Exception:
                        pass

                # Fallback ID when no valid absolute timestamp is available.
                second_value = row.get("second", 0)
                try:
                    second_int = int(float(second_value))
                except (TypeError, ValueError):
                    second_int = 0
                return (
                    f"{self.model_name}_{station_part}_{job.filepath.stem}_"
                    f"s{second_int:06d}_{event_type_short}"
                )

            events_df["event_id"] = events_df.apply(_build_event_id, axis=1)

        # Attach path information so each event row knows both the
        # original source video and the target path of the generated
        # event clip.
        if not events_df.empty:
            original_video_path = str(job.filepath.resolve())

            def _determine_subfolder(row: pd.Series) -> str:
                if row.get("type") == "arrival":
                    has_fish = bool(row.get("arrival_with_fish", False))
                    return "arrival_with_fish" if has_fish else "arrival_no_fish"
                return "departure"

            def _event_video_path(row: pd.Series) -> str:
                subfolder = _determine_subfolder(row)
                video_out_dir = get_clips_output_dir(self.config, job, f"{subfolder}/video")
                filename = format_event_filename(row.get("event_id"))
                return str((video_out_dir / filename).resolve())

            events_df["original_video_path"] = original_video_path
            events_df["event_video_path"] = events_df.apply(_event_video_path, axis=1)

            # Reorder columns to put identifiers and paths first when present
            preferred_prefix = [
                "event_id",
                "event_video_path",
                "original_video_path",
            ]
            prefix_cols = [col for col in preferred_prefix if col in events_df.columns]
            cols = prefix_cols + [col for col in events_df.columns if col not in prefix_cols]
            events_df = events_df[cols]

        self.logger.info("[%s] %d event(s) detected", job.video_id, len(events_df))
        return events_df

    @staticmethod
    def _derive_model_name(config: Config) -> str:
        """Derive a compact model identifier for use in event IDs.

        Examples:
          models/auklab_yolo26x_seabirdfish_6080_v1.pt -> "6080"
          models/yolo11x.pt -> "yolo11x"
        """
        stem = Path(config.paths.detection_model).stem

        # Prefer a 4-digit block (e.g. 6080) before an optional _vN suffix
        match = re.search(r"(\d{4})(?:_v\d+)?$", stem)
        if match:
            return match.group(1)

        # Otherwise, fall back to the trailing digit sequence if any
        match = re.search(r"(\d+)$", stem)
        if match:
            return match.group(1)

        # As a last resort, use the full stem
        return stem
