"""Worker pool management for orchestrating multi-stage pipeline execution."""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Protocol, runtime_checkable

from .config_manager import Config
from .job_scheduler import JobScheduler
from .state_manager import ProcessingStage, StateManager, VideoJob

LOGGER = logging.getLogger(__name__)


class RecoverableError(RuntimeError):
    """Exception signaling a transient failure that can be retried."""


class PermanentError(RuntimeError):
    """Exception signaling a non-recoverable failure."""


@dataclass(slots=True)
class WorkerContext:
    """Runtime context supplied to processors."""

    stage: ProcessingStage
    worker_id: str
    gpu_id: Optional[int] = None


@dataclass(slots=True)
class ProcessingMetrics:
    """Performance numbers emitted by processors for monitoring/logging."""

    video_duration_seconds: Optional[float] = None
    processing_duration_seconds: Optional[float] = None
    fps_processed: Optional[float] = None
    detections_count: Optional[int] = None
    events_count: Optional[int] = None
    clips_count: Optional[int] = None


@dataclass(slots=True)
class ProcessingResult:
    """Structured response from a stage processor."""

    metadata: Optional[Dict[str, Any]] = None
    metrics: Optional[ProcessingMetrics] = None
    duration_seconds: Optional[float] = None


ProcessorCallable = Callable[[VideoJob, WorkerContext], Optional[ProcessingResult]]


@runtime_checkable
class StageProcessor(Protocol):
    """Protocol describing stage processor implementations."""

    def process(self, job: VideoJob, context: WorkerContext) -> Optional[ProcessingResult]:
        """Process a job and optionally return a result payload."""


@dataclass(slots=True)
class WorkerSpec:
    """Blueprint for spawning workers."""

    worker_id: str
    stage: ProcessingStage
    gpu_id: Optional[int] = None


class WorkerPoolManager:
    """Coordinates CPU and GPU workers for each pipeline stage."""

    def __init__(
        self,
        config: Config,
        state_manager: StateManager,
        scheduler: JobScheduler,
        processors: Dict[ProcessingStage, StageProcessor | ProcessorCallable],
        *,
        job_poll_interval: float = 2.0,
    ) -> None:
        self.config = config
        self.state_manager = state_manager
        self.scheduler = scheduler
        self.processors = processors
        self.job_poll_interval = job_poll_interval
        self.stop_event = threading.Event()
        self.worker_specs: Dict[str, WorkerSpec] = {}
        self.workers: Dict[str, BaseWorker] = {}
        self.logger = logging.getLogger("worker_pool")

    def start_workers(self) -> None:
        if self.workers:
            self.logger.warning("Worker pool already running")
            return
        self.stop_event.clear()
        specs = self._build_worker_specs()
        for spec in specs:
            worker = self._spawn_worker(spec)
            self.worker_specs[spec.worker_id] = spec
            self.workers[spec.worker_id] = worker
            worker.start()
            self.logger.info("Started worker %s for stage %s", spec.worker_id, spec.stage.name)

    def stop_workers(self, graceful: bool = True, timeout: float = 30.0) -> None:
        if not self.workers:
            return
        self.logger.info("Stopping %s worker(s)", len(self.workers))
        self.stop_event.set()
        for worker in list(self.workers.values()):
            worker.join(timeout if graceful else 0.0)
        self.workers.clear()

    def monitor_workers(self) -> Dict[str, bool]:
        return {worker_id: worker.is_alive() for worker_id, worker in self.workers.items()}

    def restart_failed_worker(self, worker_id: str) -> bool:
        spec = self.worker_specs.get(worker_id)
        if spec is None:
            return False
        worker = self.workers.get(worker_id)
        if worker is not None and worker.is_alive():
            return False
        new_worker = self._spawn_worker(spec)
        self.workers[worker_id] = new_worker
        new_worker.start()
        self.logger.info("Restarted worker %s", worker_id)
        return True

    def _build_worker_specs(self) -> List[WorkerSpec]:
        specs: List[WorkerSpec] = []
        gpu_ids = self.config.hardware.gpus.device_ids[: self.config.hardware.gpus.count]

        workers_per_gpu = self.config.hardware.gpus.workers_per_gpu
        if ProcessingStage.STAGE1 in self.processors:
            for gpu_id in gpu_ids:
                for worker_idx in range(workers_per_gpu):
                    specs.append(WorkerSpec(
                        worker_id=f"gpu{gpu_id}-stage1-w{worker_idx}",
                        stage=ProcessingStage.STAGE1,
                        gpu_id=gpu_id,
                    ))

        if ProcessingStage.STAGE2 in self.processors:
            for idx in range(self.config.hardware.cpus.worker_count):
                specs.append(WorkerSpec(worker_id=f"cpu{idx}-stage2", stage=ProcessingStage.STAGE2))

        if not specs:
            raise ValueError("No worker specs generated. Ensure processors are provided for at least one stage.")
        return specs

    def _spawn_worker(self, spec: WorkerSpec) -> "BaseWorker":
        processor = self.processors.get(spec.stage)
        if processor is None:
            raise ValueError(f"No processor registered for stage {spec.stage}")
        worker_cls: type[BaseWorker]
        if spec.gpu_id is not None:
            worker_cls = GPUWorker
        else:
            worker_cls = CPUWorker
        return worker_cls(
            worker_id=spec.worker_id,
            stage=spec.stage,
            processor=processor,
            scheduler=self.scheduler,
            state_manager=self.state_manager,
            stop_event=self.stop_event,
            poll_interval=self.job_poll_interval,
            gpu_id=spec.gpu_id,
        )


class BaseWorker(threading.Thread):
    """Base class shared by CPU and GPU worker implementations."""

    def __init__(
        self,
        *,
        worker_id: str,
        stage: ProcessingStage,
        processor: StageProcessor | ProcessorCallable,
        scheduler: JobScheduler,
        state_manager: StateManager,
        stop_event: threading.Event,
        poll_interval: float,
        gpu_id: Optional[int] = None,
    ) -> None:
        super().__init__(name=worker_id, daemon=True)
        self.worker_id = worker_id
        self.stage = stage
        self.processor = processor
        self.scheduler = scheduler
        self.state_manager = state_manager
        self.stop_event = stop_event
        self.poll_interval = poll_interval
        self.gpu_id = gpu_id
        self.logger = logging.getLogger(f"worker.{worker_id}")

    def run(self) -> None:
        self.logger.info("Worker for stage %s started", self.stage.name)
        while not self.stop_event.is_set():
            job = self.scheduler.get_next_job(self.stage)
            if job is None:
                self.stop_event.wait(self.poll_interval)
                continue
            if not self._try_start_job(job):
                continue
            start_time = time.perf_counter()
            try:
                context = WorkerContext(stage=self.stage, worker_id=self.worker_id, gpu_id=self.gpu_id)
                result = self._invoke_processor(job, context)
                duration = self._resolve_duration(result, start_time)
                metadata = result.metadata if result else None
                self.state_manager.mark_job_completed(
                    job.video_id,
                    self.stage,
                    metadata=metadata,
                    duration_seconds=duration,
                )
                if result and result.metrics:
                    metrics = result.metrics
                    if metrics.processing_duration_seconds is None:
                        metrics.processing_duration_seconds = duration
                    self._record_metrics(job.video_id, metrics)
                self.logger.info("Completed %s for stage %s", job.video_id, self.stage.name)
            except RecoverableError as exc:
                self.logger.warning("Recoverable failure on %s: %s", job.video_id, exc)
                self._handle_failure(job, retryable=True, error_message=str(exc))
            except PermanentError as exc:
                self.logger.error("Permanent failure on %s: %s", job.video_id, exc)
                self._handle_failure(job, retryable=False, error_message=str(exc))
            except Exception as exc:  # Unexpected
                self.logger.exception("Unexpected failure on %s", job.video_id)
                self._handle_failure(job, retryable=False, error_message=str(exc))
        self.logger.info("Worker for stage %s stopped", self.stage.name)

    def _invoke_processor(self, job: VideoJob, context: WorkerContext) -> Optional[ProcessingResult]:
        processor = self.processor
        if hasattr(processor, "process"):
            return processor.process(job, context)  # type: ignore[call-arg]
        return processor(job, context)  # type: ignore[misc]

    def _try_start_job(self, job: VideoJob) -> bool:
        try:
            self.state_manager.mark_job_started(job.video_id, self.stage, self.worker_id)
            return True
        except ValueError:
            self.logger.debug("Job %s no longer pending", job.video_id)
            return False

    def _handle_failure(self, job: VideoJob, *, retryable: bool, error_message: str) -> None:
        self.state_manager.mark_job_failed(
            job.video_id,
            self.stage,
            error_message=error_message,
            retryable=retryable,
        )
        if retryable:
            self.scheduler.return_job(job, self.stage)

    def _record_metrics(self, video_id: str, metrics: ProcessingMetrics) -> None:
        self.state_manager.record_performance_metrics(
            video_id,
            self.stage,
            video_duration_seconds=metrics.video_duration_seconds,
            processing_duration_seconds=metrics.processing_duration_seconds,
            fps_processed=metrics.fps_processed,
            detections_count=metrics.detections_count,
            events_count=metrics.events_count,
            clips_count=metrics.clips_count,
        )

    @staticmethod
    def _resolve_duration(result: Optional[ProcessingResult], start_time: float) -> Optional[float]:
        if result and result.duration_seconds is not None:
            return result.duration_seconds
        return time.perf_counter() - start_time


class GPUWorker(BaseWorker):
    """Worker tied to a specific GPU device."""


class CPUWorker(BaseWorker):
    """Worker executing CPU-bound stages."""
