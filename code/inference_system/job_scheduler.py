"""Video discovery and job scheduling utilities for the inference pipeline."""

from __future__ import annotations

import hashlib
import logging
import queue
import re
from dataclasses import dataclass
from datetime import date as date_cls, datetime
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Set, Tuple

from .config_manager import Config
from .state_manager import ProcessingStage, StateManager, VideoJob

LOGGER = logging.getLogger(__name__)
DEFAULT_VIDEO_EXTENSIONS = {".mkv", ".mp4", ".avi"}

_STATION_TIMESTAMP_RE = re.compile(r"(?P<station>[A-Za-z0-9]+)_(?P<stamp>\d{8}T\d{6})")
_STATION_DATE_RE = re.compile(r"(?P<station>[A-Za-z0-9]+)_(?P<date>\d{8})")


@dataclass(slots=True)
class DiscoveredVideo:
    """Internal representation of a candidate video file."""

    path: Path
    video_id: str
    station: str
    timestamp: datetime
    filename: str

    @property
    def year(self) -> int:
        return self.timestamp.year

    @property
    def date_str(self) -> str:
        return self.timestamp.strftime("%Y-%m-%d")


class JobScheduler:
    """Discovers video files, ranks them, and feeds work queues per stage."""

    def __init__(
        self,
        config: Config,
        state_manager: StateManager,
        *,
        allowed_stations: Optional[List[str]] = None,
        allowed_years: Optional[List[int]] = None,
    ) -> None:
        self.config = config
        self.state_manager = state_manager
        self.video_extensions = {ext.lower() for ext in config.processing.stage3_clip_extraction.video_extensions}
        if not self.video_extensions:
            self.video_extensions = DEFAULT_VIDEO_EXTENSIONS.copy()
        self.stage_queues: Dict[ProcessingStage, "queue.PriorityQueue[Tuple[float, VideoJob]]"] = {}
        self.allowed_stations = {station.upper() for station in allowed_stations} if allowed_stations else None
        self.allowed_years = set(allowed_years) if allowed_years else None

    def discover_videos(self) -> int:
        """Walk configured video roots and register unseen files."""
        discovered_jobs: List[VideoJob] = []
        seen_ids: Set[str] = set()

        for video_path in self._iter_video_files():
            metadata = self._parse_video_metadata(video_path)
            if metadata is None:
                continue
            if metadata.video_id in seen_ids:
                continue
            if self.allowed_stations and metadata.station.upper() not in self.allowed_stations:
                continue
            if self.allowed_years and metadata.year not in self.allowed_years:
                continue
            if not self._passes_filters(metadata.timestamp.date()):
                continue
            seen_ids.add(metadata.video_id)
            priority = self.calculate_priority(metadata.station, metadata.year, metadata.timestamp.date())
            job = VideoJob(
                video_id=metadata.video_id,
                station=metadata.station,
                year=metadata.year,
                date=metadata.date_str,
                filename=metadata.filename,
                filepath=metadata.path,
                priority_score=priority,
                stage=ProcessingStage.STAGE1,
                retry_count=0,
            )
            discovered_jobs.append(job)

        if not discovered_jobs:
            LOGGER.info("No new videos discovered")
            return 0

        inserted = self.state_manager.register_videos(discovered_jobs)
        LOGGER.info("Discovered %s video(s) across %s root(s)", inserted, len(self.config.video_sources))
        return inserted

    def calculate_priorities(self) -> None:
        """Invalidate cached queues so fresh priorities are pulled from the database."""
        self.stage_queues.clear()

    def build_job_queue(self, stage: ProcessingStage) -> "queue.PriorityQueue[Tuple[float, VideoJob]]":
        pending = self.state_manager.get_pending_jobs(stage)
        pq: "queue.PriorityQueue[Tuple[float, VideoJob]]" = queue.PriorityQueue()
        for job in pending:
            pq.put((-job.priority_score, job))
        self.stage_queues[stage] = pq
        return pq

    def get_next_job(self, stage: ProcessingStage) -> Optional[VideoJob]:
        pq = self.stage_queues.get(stage)
        if pq is None or pq.empty():
            pq = self.build_job_queue(stage)
            if pq.empty():
                return None
        try:
            _, job = pq.get_nowait()
        except queue.Empty:
            return None
        return job

    def return_job(self, job: VideoJob, stage: ProcessingStage) -> None:
        pq = self.stage_queues.setdefault(stage, queue.PriorityQueue())
        pq.put((-job.priority_score, job))

    def calculate_priority(self, station: str, year: int, job_date: date_cls) -> float:
        """Score videos using year/station priority lists and recency."""
        year_priorities = {
            value: len(self.config.year_priority) - idx for idx, value in enumerate(self.config.year_priority)
        }
        station_priorities = {
            value: len(self.config.station_priority) - idx for idx, value in enumerate(self.config.station_priority)
        }

        year_weight = year_priorities.get(year, 0)
        station_key = station.upper()
        station_weight = station_priorities.get(station_key, 0)
        days_from_year_start = (job_date - date_cls(year, 1, 1)).days
        date_weight = max(0, 365 - days_from_year_start) / 365

        return year_weight * 1000 + station_weight * 10 + date_weight

    def _passes_filters(self, video_date: date_cls) -> bool:
        if not self.config.filters or not self.config.filters.date_range:
            return True
        date_range = self.config.filters.date_range
        if date_range.start and video_date < date_range.start:
            return False
        if date_range.end and video_date > date_range.end:
            return False
        return True

    def _iter_video_files(self) -> Iterator[Path]:
        for base in self.config.video_sources:
            base_path = Path(base)
            if not base_path.exists():
                LOGGER.warning("Video root does not exist: %s", base_path)
                continue
            for path in base_path.rglob("*"):
                if not path.is_file():
                    continue
                if "@eaDir" in path.parts:
                    continue
                if path.suffix.lower() in self.video_extensions:
                    yield path

    def _parse_video_metadata(self, video_path: Path) -> Optional[DiscoveredVideo]:
        stem = video_path.stem
        name = video_path.name
        match = _STATION_TIMESTAMP_RE.search(stem)
        if match:
            station = match.group("station").upper()
            timestamp = datetime.strptime(match.group("stamp"), "%Y%m%dT%H%M%S")
            video_id = f"{station}_{match.group('stamp')}"
            return DiscoveredVideo(video_path, video_id, station, timestamp, name)

        match = _STATION_DATE_RE.search(stem)
        if match:
            station = match.group("station").upper()
            timestamp = datetime.strptime(match.group("date"), "%Y%m%d")
            token = match.group("date")
            video_id = f"{station}_{token}_{self._stable_suffix(video_path)}"
            return DiscoveredVideo(video_path, video_id, station, timestamp, name)

        station = self._infer_station(video_path)
        timestamp = datetime.utcfromtimestamp(video_path.stat().st_mtime)
        video_id = f"{station}_{timestamp.strftime('%Y%m%dT%H%M%S')}_{self._stable_suffix(video_path)}"
        return DiscoveredVideo(video_path, video_id, station, timestamp, name)

    @staticmethod
    def _infer_station(video_path: Path) -> str:
        for part in reversed(video_path.parts):
            lowered = part.lower()
            if not part or "@" in part:
                continue
            if any(lowered.endswith(ext) for ext in DEFAULT_VIDEO_EXTENSIONS):
                continue
            return part.upper()
        return video_path.stem.upper()

    @staticmethod
    def _stable_suffix(video_path: Path) -> str:
        digest = hashlib.md5(str(video_path).encode("utf-8")).hexdigest()
        return digest[:6].upper()
