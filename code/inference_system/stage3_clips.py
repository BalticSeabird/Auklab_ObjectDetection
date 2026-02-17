"""Stage 3 processor: extract annotated video clips for detected events."""

from __future__ import annotations

import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd

PROJECT_CODE_DIR = Path(__file__).resolve().parents[1]
CODE_ROOT = PROJECT_CODE_DIR.parent
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from postprocess.event_detector import extract_timestamp_from_filename  # type: ignore  # noqa: E402
from postprocess.extract_event_clips import (  # type: ignore  # noqa: E402
    compress_video_h264,
    extract_clip_with_overlay,
    format_event_filename,
    format_overlay_text,
    get_video_duration,
    run_yolo_on_clip,
)

from .config_manager import Config
from .path_utils import get_clips_output_dir, get_event_csv_path
from .state_manager import VideoJob
from .worker_pool import (
    PermanentError,
    ProcessingMetrics,
    ProcessingResult,
    RecoverableError,
    StageProcessor,
    WorkerContext,
)

LOGGER = logging.getLogger(__name__)


class ClipExtractionProcessor(StageProcessor):
    """Create YOLO-annotated clips for arrivals/departures."""

    def __init__(self, config: Config) -> None:
        self.config = config
        self.stage_cfg = config.processing.stage3_clip_extraction
        self.model_path = Path(config.paths.detection_model)
        self.logger = logging.getLogger("stage3")

    def process(self, job: VideoJob, context: WorkerContext) -> Optional[ProcessingResult]:  # noqa: ARG002
        event_csv = get_event_csv_path(self.config, job)
        if not event_csv.exists():
            raise RecoverableError(f"Events CSV missing for {job.video_id}: {event_csv}")

        if not job.filepath.exists():
            raise PermanentError(f"Source video missing for {job.video_id}: {job.filepath}")

        events_df = pd.read_csv(event_csv)
        source_label = self._source_label(job)
        filtered_events = self.filter_events(events_df, source_label=source_label)
        return self.process_events_for_job(job, filtered_events, event_csv_path=event_csv)

    def process_events_for_job(
        self,
        job: VideoJob,
        events_df: pd.DataFrame,
        *,
        event_csv_path: Path,
    ) -> ProcessingResult:
        """Generate clips for the subset of events that belong to the provided job."""
        if events_df.empty:
            self.logger.info("[%s] No events to clip", job.video_id)
            return ProcessingResult(
                metadata={"clips": 0, "events_csv": str(event_csv_path)},
                metrics=ProcessingMetrics(clips_count=0),
            )

        video_start = self._extract_video_start(job.filepath)
        if video_start is None:
            raise PermanentError(f"Unable to infer start time from filename: {job.filepath.name}")
        video_duration = self._safe_video_duration(job.filepath)

        clips_created = 0
        clips_failed = 0
        for _, event in events_df.iterrows():
            success = self._process_event(job, event, video_start, video_duration)
            if success:
                clips_created += 1
            else:
                clips_failed += 1

        metadata = {
            "events_csv": str(event_csv_path),
            "clips_created": clips_created,
            "clips_failed": clips_failed,
        }
        return ProcessingResult(
            metadata=metadata,
            metrics=ProcessingMetrics(clips_count=clips_created),
        )

    def _filter_events(self, events_df: pd.DataFrame, *, source_label: Optional[str] = None) -> pd.DataFrame:
        df = events_df.copy()
        if "timestamp" not in df.columns:
            if "absolute_timestamp" in df.columns:
                df["timestamp"] = pd.to_datetime(df["absolute_timestamp"], errors="coerce")
            else:
                df["timestamp"] = pd.NaT
        else:
            df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

        if source_label:
            df = self._filter_events_for_source(df, source_label)

        if self.stage_cfg.event_types:
            df = df[df["type"].isin(self.stage_cfg.event_types)]
        if self.stage_cfg.fish_only:
            df = df[(df["type"] == "arrival") & (df.get("arrival_with_fish", False) == True)]
        df = df.dropna(subset=["timestamp"])
        return df

    def _filter_events_for_source(self, df: pd.DataFrame, source_label: str) -> pd.DataFrame:
        normalized_label = source_label.upper()
        if "file" in df.columns:
            matches = df["file"].astype(str).str.upper() == normalized_label
            subset = df[matches]
            if not subset.empty:
                return subset
        if "observation_period" in df.columns:
            cleaned = (
                df["observation_period"]
                .astype(str)
                .str.replace(".csv", "", regex=False)
                .str.replace("_raw", "", regex=False)
                .str.upper()
            )
            matches = cleaned == normalized_label
            subset = df[matches]
            if not subset.empty:
                return subset
        self.logger.debug("No events matched source label %s", source_label)
        return df.iloc[0:0]

    def _process_event(
        self,
        job: VideoJob,
        event: pd.Series,
        video_start: datetime,
        video_duration: Optional[float],
    ) -> bool:
        offset_seconds = self._resolve_event_offset(event, video_start)
        if offset_seconds is None:
            self.logger.warning("[%s] Unable to determine offset for event", job.video_id)
            return False
        if offset_seconds < 0:
            self.logger.warning("[%s] Event before video start, skipping", job.video_id)
            return False
        clip_start = max(0.0, offset_seconds - self.stage_cfg.clip_before)
        clip_end = offset_seconds + self.stage_cfg.clip_after
        if video_duration is not None:
            clip_end = min(video_duration, clip_end)
        if clip_end - clip_start <= 0:
            self.logger.warning("[%s] Invalid clip window, skipping", job.video_id)
            return False

        subfolder = self._determine_subfolder(event)
        video_out_dir = get_clips_output_dir(self.config, job, f"{subfolder}/video")
        csv_out_dir = get_clips_output_dir(self.config, job, f"{subfolder}/csv")
        video_out_dir.mkdir(parents=True, exist_ok=True)
        csv_out_dir.mkdir(parents=True, exist_ok=True)

        event_id = event.get("event_id")
        output_filename = format_event_filename(event_id)
        final_video_path = video_out_dir / output_filename
        detections_csv = csv_out_dir / output_filename.replace(".mp4", "_detections.csv")

        if final_video_path.exists() and detections_csv.exists():
            self.logger.debug("[%s] Clip already exists: %s", job.video_id, final_video_path)
            return True

        overlay_text = format_overlay_text(event)
        temp_clip = video_out_dir / f"temp_{output_filename}"
        success = extract_clip_with_overlay(
            job.filepath,
            clip_start,
            clip_end,
            temp_clip,
            overlay_text,
        )
        if not success:
            self.logger.error("[%s] Failed to extract base clip for %s", job.video_id, output_filename)
            if temp_clip.exists():
                temp_clip.unlink()
            return False

        if self.model_path.exists():
            yolo_temp = video_out_dir / f"yolo_{output_filename}"
            detections_success = run_yolo_on_clip(
                temp_clip,
                self.model_path,
                yolo_temp,
                detections_csv,
                overlay_text,
            )
            if temp_clip.exists():
                temp_clip.unlink()
            if not detections_success:
                if yolo_temp.exists():
                    yolo_temp.unlink()
                self.logger.error("[%s] YOLO overlay failed for %s", job.video_id, output_filename)
                return False

            if self.stage_cfg.compression.enabled:
                compressed = compress_video_h264(
                    yolo_temp,
                    final_video_path,
                    crf=self.stage_cfg.compression.crf,
                    preset=self.stage_cfg.compression.preset,
                )
                if yolo_temp.exists():
                    yolo_temp.unlink()
                if not compressed:
                    self.logger.error("[%s] Compression failed for %s", job.video_id, output_filename)
                    return False
            else:
                yolo_temp.rename(final_video_path)
        else:
            # No YOLO model available, keep base clip and emit empty detections CSV
            temp_clip.rename(final_video_path)
            if not detections_csv.exists():
                pd.DataFrame(
                    columns=["frame", "class", "confidence", "xmin", "ymin", "xmax", "ymax"]
                ).to_csv(detections_csv, index=False)

        return True

    def _resolve_event_offset(self, event: pd.Series, video_start: Optional[datetime]) -> Optional[float]:
        if "second" in event and pd.notna(event["second"]):
            try:
                return float(event["second"])
            except (TypeError, ValueError):
                self.logger.debug("Invalid second value for event %s", event.get("event_id"))
        timestamp = event.get("timestamp")
        if pd.isna(timestamp) or video_start is None:
            return None
        return float((timestamp - video_start).total_seconds())

    @staticmethod
    def _source_label(job: VideoJob) -> str:
        return job.filepath.stem.upper()

    def _determine_subfolder(self, event: pd.Series) -> str:
        if event["type"] == "arrival":
            has_fish = bool(event.get("arrival_with_fish", False))
            return "arrival_with_fish" if has_fish else "arrival_no_fish"
        return "departure"

    def _extract_video_start(self, video_path: Path) -> Optional[datetime]:
        timestamp = extract_timestamp_from_filename(str(video_path.name))
        if timestamp:
            return timestamp
        return None

    def _safe_video_duration(self, video_path: Path) -> Optional[float]:
        try:
            return float(get_video_duration(video_path))
        except Exception as exc:  # pragma: no cover - best effort
            self.logger.warning("Failed to read video duration for %s: %s", video_path, exc)
            return None
