"""Stage 1 processor: run YOLO inference on raw video files."""

from __future__ import annotations

import logging
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional

import av  # type: ignore
import pandas as pd
import torch
from ultralytics import YOLO  # type: ignore

from .config_manager import Config
from .path_utils import get_detection_csv_path, get_detection_output_dir, get_model_display_name
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


class VideoInferenceProcessor(StageProcessor):
    """Execute YOLO inference on individual videos using PyAV for decoding."""

    def __init__(self, config: Config) -> None:
        self.config = config
        self.stage_cfg = config.processing.stage1_video_inference
        self.model_path = Path(config.paths.detection_model)
        self.model_name = get_model_display_name(config)
        self._model_cache: Dict[str, YOLO] = {}
        self._model_lock = threading.Lock()
        self.logger = logging.getLogger("stage1")

    def process(self, job: VideoJob, context: WorkerContext) -> Optional[ProcessingResult]:
        video_path = job.filepath
        if not video_path.exists():
            raise PermanentError(f"Video file not found: {video_path}")

        output_dir = get_detection_output_dir(self.config, job)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_csv = get_detection_csv_path(self.config, job)

        if output_csv.exists():
            self.logger.info("[%s] detections already exist, skipping", job.video_id)
            return ProcessingResult(
                metadata={"detections_csv": str(output_csv), "skipped": True},
                metrics=ProcessingMetrics(detections_count=0),
                duration_seconds=0.0,
            )

        device_key = self._device_key(context.gpu_id)
        model = self._get_or_load_model(device_key)

        start_time = time.perf_counter()
        detections = self._run_inference(video_path, model)

        df = pd.DataFrame(detections["rows"], columns=[
            "frame",
            "class",
            "confidence",
            "xmin",
            "ymin",
            "xmax",
            "ymax",
        ])
        df.to_csv(output_csv, index=False)
        duration = time.perf_counter() - start_time

        detections_count = len(df)
        fps_processed = 0.0
        if duration > 0 and detections["frames_processed"] > 0:
            fps_processed = detections["frames_processed"] / duration

        metrics = ProcessingMetrics(
            video_duration_seconds=detections["video_duration"],
            processing_duration_seconds=duration,
            fps_processed=fps_processed,
            detections_count=detections_count,
        )

        metadata = {
            "detections_csv": str(output_csv),
            "frames_processed": detections["frames_processed"],
            "frame_skip": self.stage_cfg.frame_skip,
            "batch_size": self.stage_cfg.batch_size,
            "model": self.model_name,
        }

        self.logger.info(
            "[%s] saved %s detections to %s", job.video_id, detections_count, output_csv.name
        )
        return ProcessingResult(metadata=metadata, metrics=metrics, duration_seconds=duration)

    def _device_key(self, gpu_id: Optional[int]) -> str:
        if gpu_id is None or not torch.cuda.is_available():
            return "cpu"
        return f"cuda:{gpu_id}"

    def _get_or_load_model(self, device_key: str) -> YOLO:
        with self._model_lock:
            if device_key in self._model_cache:
                return self._model_cache[device_key]

            if not self.model_path.exists():
                raise PermanentError(f"Model file not found: {self.model_path}")

            model = YOLO(str(self.model_path))
            model.to(device_key)
            self._model_cache[device_key] = model
            self.logger.info("Loaded YOLO model %s onto %s", self.model_name, device_key)
            return model

    def _run_inference(self, video_path: Path, model: YOLO) -> Dict[str, float | int | List[List[float]]]:
        try:
            container = av.open(str(video_path))
        except av.error.FFmpegError as exc:
            message = str(exc)
            if (
                "Invalid data found when processing input" in message
                or "End of file" in message
                or "EOF" in message
            ):
                raise PermanentError(
                    f"Video file appears corrupt, truncated, or unreadable and will be skipped: {video_path}: {message}"
                ) from exc
            raise RecoverableError(f"Failed to open video {video_path}: {message}") from exc

        stream = container.streams.video[0]
        stream.thread_type = "AUTO"

        frame_skip = self.stage_cfg.frame_skip
        batch_size = self.stage_cfg.batch_size

        results_list: List[List[float | int | str]] = []
        frame_buffer: List = []
        frame_indices: List[int] = []
        frame_count = 0
        processed_frames = 0

        def flush_batch() -> None:
            nonlocal processed_frames
            if not frame_buffer:
                return
            try:
                with torch.no_grad():
                    predictions = model(frame_buffer, verbose=False)
            except RuntimeError as exc:
                raise RecoverableError(f"YOLO inference failed: {exc}") from exc

            for prediction, idx in zip(predictions, frame_indices):
                names = prediction.names
                for box in prediction.boxes:
                    cls_idx = int(box.cls)
                    results_list.append([
                        idx,
                        names.get(cls_idx, str(cls_idx)),
                        float(box.conf.cpu()),
                        float(box.xyxy[0][0].cpu()),
                        float(box.xyxy[0][1].cpu()),
                        float(box.xyxy[0][2].cpu()),
                        float(box.xyxy[0][3].cpu()),
                    ])
            processed_frames += len(frame_buffer)
            frame_buffer.clear()
            frame_indices.clear()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        try:
            for frame in container.decode(stream):
                if frame_count % frame_skip == 0:
                    frame_buffer.append(frame.to_ndarray(format="bgr24"))
                    frame_indices.append(frame_count)
                    if len(frame_buffer) >= batch_size:
                        flush_batch()
                frame_count += 1
            flush_batch()
        except av.error.FFmpegError as exc:
            raise RecoverableError(f"Decoding failed for {video_path}: {exc}") from exc
        finally:
            container.close()

        video_duration = None
        if stream.duration and stream.time_base:
            try:
                video_duration = float(stream.duration * stream.time_base)
            except Exception:  # pragma: no cover - best-effort metadata
                video_duration = None

        return {
            "rows": results_list,
            "frames_processed": processed_frames,
            "video_duration": video_duration,
        }
