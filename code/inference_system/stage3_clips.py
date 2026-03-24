"""Stage 3 processor: extract annotated video clips for detected events."""

from __future__ import annotations

import logging
import os
import re
import sys
from datetime import datetime, timedelta
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
from .event_database import EventDatabase
from .path_utils import get_clips_output_dir
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
        self.timestamp_fail_log = Path(config.paths.log_dir) / "failed_timestamp_videos.tsv"

    def process(self, job: VideoJob, context: WorkerContext) -> Optional[ProcessingResult]:  # noqa: ARG002
        event_db = EventDatabase.for_station(self.config, job.station)
        event_db.initialize()
        events_df = event_db.fetch_events_for_video(station=job.station, video_id=job.video_id)
        events_df = self._normalize_event_columns(events_df)

        if not job.filepath.exists():
            raise PermanentError(f"Source video missing for {job.video_id}: {job.filepath}")

        filtered_events = self._filter_events(events_df)
        return self.process_events_for_job(job, filtered_events)

    @staticmethod
    def _normalize_event_columns(events_df: pd.DataFrame) -> pd.DataFrame:
        if events_df.empty:
            return events_df
        df = events_df.copy()
        if "event_type" in df.columns and "type" not in df.columns:
            df["type"] = df["event_type"]
        if "type" not in df.columns:
            # Legacy / partially migrated rows may not have explicit type.
            # Infer when possible from event_id suffix, else keep empty.
            if "event_id" in df.columns:
                event_ids = df["event_id"].astype(str).str.lower()
                inferred = pd.Series(index=df.index, dtype=object)
                inferred[event_ids.str.endswith("_arr")] = "arrival"
                inferred[event_ids.str.endswith("_dep")] = "departure"
                df["type"] = inferred
            else:
                df["type"] = pd.Series(index=df.index, dtype=object)
        if "arrival_with_fish_stage2" in df.columns and "arrival_with_fish" not in df.columns:
            df["arrival_with_fish"] = df["arrival_with_fish_stage2"].fillna(0).astype(int)
        return df

    def process_events_for_job(
        self,
        job: VideoJob,
        events_df: pd.DataFrame,
    ) -> ProcessingResult:
        """Generate clips for the subset of events that belong to the provided job."""
        if events_df.empty:
            self.logger.info("[%s] No events to clip", job.video_id)
            return ProcessingResult(
                metadata={"clips": 0},
                metrics=ProcessingMetrics(clips_count=0),
            )

        event_db = EventDatabase.for_station(self.config, job.station)
        event_db.initialize()

        video_start = self._extract_video_start(job.filepath, job.video_id)
        if video_start is None:
            self._record_failed_timestamp(job)
            self.logger.warning(
                "[%s] Unable to infer video start from filename '%s'; "
                "falling back to per-event second offsets when available",
                job.video_id,
                job.filepath.name,
            )
        video_duration = self._safe_video_duration(job.filepath)

        clips_created = 0
        clips_failed = 0
        for _, event in events_df.iterrows():
            success = self._process_event(job, event, video_start, video_duration, event_db)
            if success:
                clips_created += 1
            else:
                clips_failed += 1

        metadata = {
            "clips_created": clips_created,
            "clips_failed": clips_failed,
        }
        return ProcessingResult(
            metadata=metadata,
            metrics=ProcessingMetrics(clips_count=clips_created),
        )

    def _filter_events(self, events_df: pd.DataFrame, *, source_label: Optional[str] = None) -> pd.DataFrame:
        df = events_df.copy()
        # Keep second as numeric when present; clip timing can be derived from
        # this offset even when absolute timestamps are unavailable.
        if "second" in df.columns:
            df["second"] = pd.to_numeric(df["second"], errors="coerce")
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
            if "type" in df.columns:
                df = df[df["type"].isin(self.stage_cfg.event_types)]
            else:
                self.logger.warning("Events data missing 'type' column; skipping event-type filter")
        if self.stage_cfg.fish_only:
            if "type" in df.columns:
                df = df[(df["type"] == "arrival") & (df.get("arrival_with_fish", False) == True)]
            else:
                self.logger.warning("Events data missing 'type' column; skipping fish_only filter")

        # Accept either absolute timestamp or per-video second offset.
        # Some event CSVs only contain `second`, and those events are still
        # fully clip-able because stage3 resolves offsets from `second` first.
        has_timestamp = df["timestamp"].notna()
        has_second = df.get("second", pd.Series(index=df.index, dtype=float)).notna()
        df = df[has_timestamp | has_second]
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
        event_db: EventDatabase,
    ) -> bool:
        event_id = str(event.get("event_id", ""))
        offset_seconds = self._resolve_event_offset(event, video_start)
        if offset_seconds is None:
            self.logger.warning("[%s] Unable to determine offset for event", job.video_id)
            if event_id:
                event_db.update_stage3_artifacts(
                    station=job.station,
                    event_id=event_id,
                    event_video_path=None,
                    detections_csv_path=None,
                    stage3_status="failed_offset",
                )
            return False
        if offset_seconds < 0:
            self.logger.warning("[%s] Event before video start, skipping", job.video_id)
            if event_id:
                event_db.update_stage3_artifacts(
                    station=job.station,
                    event_id=event_id,
                    event_video_path=None,
                    detections_csv_path=None,
                    stage3_status="failed_negative_offset",
                )
            return False
        clip_start = max(0.0, offset_seconds - self.stage_cfg.clip_before)
        clip_end = offset_seconds + self.stage_cfg.clip_after
        if video_duration is not None:
            clip_end = min(video_duration, clip_end)
        if clip_end - clip_start <= 0:
            self.logger.warning("[%s] Invalid clip window, skipping", job.video_id)
            if event_id:
                event_db.update_stage3_artifacts(
                    station=job.station,
                    event_id=event_id,
                    event_video_path=None,
                    detections_csv_path=None,
                    stage3_status="failed_window",
                )
            return False

        subfolder = self._determine_subfolder(event)
        video_out_dir = get_clips_output_dir(self.config, job, f"{subfolder}/video")
        csv_out_dir = get_clips_output_dir(self.config, job, f"{subfolder}/csv")
        video_out_dir.mkdir(parents=True, exist_ok=True)
        csv_out_dir.mkdir(parents=True, exist_ok=True)

        output_filename = format_event_filename(event_id)
        final_video_path = video_out_dir / output_filename
        detections_csv = csv_out_dir / output_filename.replace(".mp4", "_detections.csv")

        if final_video_path.exists() and detections_csv.exists():
            self.logger.debug("[%s] Clip already exists: %s", job.video_id, final_video_path)
            if event_id:
                event_db.update_stage3_artifacts(
                    station=job.station,
                    event_id=event_id,
                    event_video_path=str(final_video_path.resolve()),
                    detections_csv_path=str(detections_csv.resolve()),
                    stage3_status="completed",
                )
            return True

        # Ensure overlay timestamp is populated even when events CSV has NaT.
        # We derive it from canonical timing: video start + event offset.
        event_for_overlay = event.copy()
        if pd.isna(event_for_overlay.get("timestamp")) and video_start is not None:
            event_for_overlay["timestamp"] = video_start + pd.to_timedelta(offset_seconds, unit="s")
        overlay_text = format_overlay_text(event_for_overlay)
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
            if event_id:
                event_db.update_stage3_artifacts(
                    station=job.station,
                    event_id=event_id,
                    event_video_path=None,
                    detections_csv_path=None,
                    stage3_status="failed_extract",
                )
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
                if event_id:
                    event_db.update_stage3_artifacts(
                        station=job.station,
                        event_id=event_id,
                        event_video_path=None,
                        detections_csv_path=None,
                        stage3_status="failed_yolo",
                    )
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
                    if event_id:
                        event_db.update_stage3_artifacts(
                            station=job.station,
                            event_id=event_id,
                            event_video_path=None,
                            detections_csv_path=None,
                            stage3_status="failed_compress",
                        )
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

        if event_id:
            event_db.update_stage3_artifacts(
                station=job.station,
                event_id=event_id,
                event_video_path=str(final_video_path.resolve()),
                detections_csv_path=str(detections_csv.resolve()),
                stage3_status="completed",
            )

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

    def _extract_video_start(self, video_path: Path, video_id: Optional[str] = None) -> Optional[datetime]:
        def _parse_date_and_time_tokens(date_token: str, time_token: str) -> Optional[datetime]:
            """Parse YYYYMMDD + HHMMSS and tolerate overflow in MM/SS.

            Some legacy files contain malformed time tokens like 034186 where
            seconds overflow. We normalize via timedelta math instead of
            rejecting the timestamp.
            """
            if len(date_token) != 8 or len(time_token) != 6:
                return None
            try:
                base_date = datetime.strptime(date_token, "%Y%m%d")
                hours = int(time_token[:2])
                minutes = int(time_token[2:4])
                seconds = int(time_token[4:6])
                return base_date + timedelta(hours=hours, minutes=minutes, seconds=seconds)
            except ValueError:
                return None

        timestamp = extract_timestamp_from_filename(str(video_path.name))
        if timestamp:
            return timestamp

        # Fallback for names like Rost3_20200517_182700.avi
        stem = video_path.stem
        underscore_match = re.search(r"(\d{8})_(\d{6})", stem)
        if underscore_match:
            parsed = _parse_date_and_time_tokens(underscore_match.group(1), underscore_match.group(2))
            if parsed:
                return parsed

        # Fallback for compact tokens like STATION_20200517182700
        compact_match = re.search(r"(\d{14})", stem)
        if compact_match:
            compact = compact_match.group(1)
            parsed = _parse_date_and_time_tokens(compact[:8], compact[8:14])
            if parsed:
                return parsed

        # Last-resort: derive from normalized video_id (e.g. ROST3_20200517T182700)
        if video_id:
            id_match = re.search(r"(\d{8}T\d{6})", video_id)
            if id_match:
                try:
                    return datetime.strptime(id_match.group(1), "%Y%m%dT%H%M%S")
                except ValueError:
                    pass
        return None

    def _safe_video_duration(self, video_path: Path) -> Optional[float]:
        try:
            return float(get_video_duration(video_path))
        except Exception as exc:  # pragma: no cover - best effort
            self.logger.warning("Failed to read video duration for %s: %s", video_path, exc)
            return None

    def _record_failed_timestamp(self, job: VideoJob) -> None:
        """Append filename timestamp parse failures for later manual QA."""
        try:
            self.timestamp_fail_log.parent.mkdir(parents=True, exist_ok=True)
            is_new_file = not self.timestamp_fail_log.exists()
            header = "logged_at_utc\tvideo_id\tstation\tdate\tfilepath\tfilename\n"
            line = (
                f"{datetime.utcnow().isoformat()}\t{job.video_id}\t{job.station}\t"
                f"{job.date}\t{job.filepath}\t{job.filepath.name}\n"
            )
            fd = os.open(
                str(self.timestamp_fail_log),
                os.O_CREAT | os.O_WRONLY | os.O_APPEND,
                0o644,
            )
            try:
                if is_new_file:
                    os.write(fd, header.encode("utf-8", errors="replace"))
                os.write(fd, line.encode("utf-8", errors="replace"))
            finally:
                os.close(fd)
        except Exception as exc:  # pragma: no cover - best effort logging only
            self.logger.debug("Failed to write timestamp failure audit line for %s: %s", job.video_id, exc)
