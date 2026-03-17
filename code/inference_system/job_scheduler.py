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
_STATION_UNDERSCORE_TIMESTAMP_RE = re.compile(
    r"(?P<station>[A-Za-z0-9]+)_(?P<date>\d{8})_(?P<time>\d{6})"
)
_STATION_COMPACT_TIMESTAMP_RE = re.compile(r"(?P<station>[A-Za-z0-9]+)_(?P<stamp>\d{14})")
_STATION_XPROTECT_RE = re.compile(
    r"(?:(?P<camera>\d+)_)?"
    r"(?P<station>[A-Za-z0-9]+)"
    r"(?:_\([^)]*\))?"
    r"_(?P<date>\d{4}-\d{2}-\d{2})"
    r"_(?P<time>\d{2}\.\d{2}\.\d{2})"
    r"(?:_\d+)?$"
)
_STATION_HUMAN_TIMESTAMP_RE = re.compile(
    r"(?:(?P<prefix>[A-Za-z0-9]+)_)?(?P<station>[A-Za-z0-9]+(?:_[A-Za-z0-9]+)*)_(?P<date>\d{4}[-_]\d{2}[-_]\d{2})(?:_(?P<time>\d{2}[:.\-]\d{2}[:.\-]\d{2}))?"
)
_STATION_DATE_RE = re.compile(r"(?P<station>[A-Za-z0-9]+)_(?P<date>\d{8})")
_DATE_DIR_RE = re.compile(r"\d{4}-\d{2}-\d{2}")
_YEAR_DIR_RE = re.compile(r"\d{4}")


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
        # priority queues store (-priority, deterministic tie-breaker, job)
        self.stage_queues: Dict[ProcessingStage, "queue.PriorityQueue[Tuple[float, str, VideoJob]]"] = {}

        # Optional explicit station filters from CLI (--stations). These are
        # used as an allow-list when provided, with case-insensitive partial
        # matching.
        #
        # IMPORTANT: config.priorities.stations is NOT an allow-list for
        # discovery. It only influences priority scoring.
        if allowed_stations:
            self._explicit_station_filters: Optional[List[str]] = [station.upper() for station in allowed_stations]
        else:
            self._explicit_station_filters = None

        # Stations that should always be ignored when building the database,
        # regardless of priorities. Also matched by case-insensitive
        # substring.
        if config.filters and config.filters.ignored_stations:
            self._ignored_station_patterns: List[str] = [s.upper() for s in config.filters.ignored_stations]
        else:
            self._ignored_station_patterns = []
        self.allowed_years = set(allowed_years) if allowed_years else None

    # ------------------------------------------------------------------

    @staticmethod
    def _normalize_station_token(value: str) -> str:
        """Normalize station tokens for resilient matching.

        Removes separators and other non-alphanumeric characters so patterns
        like "BJORN3_TRI3_SCALE" and "BJORN3TRI3SCALE" match each other.
        """
        return re.sub(r"[^A-Z0-9]", "", value.upper())

    def _station_matches_pattern(self, station: str, pattern: str) -> bool:
        if not pattern:
            return False
        station_upper = station.upper()
        pattern_upper = pattern.upper()
        if pattern_upper in station_upper:
            return True

        station_norm = self._normalize_station_token(station)
        pattern_norm = self._normalize_station_token(pattern)
        return bool(pattern_norm) and pattern_norm in station_norm

    def _is_station_ignored(self, station: str) -> bool:
        return any(self._station_matches_pattern(station, pattern) for pattern in self._ignored_station_patterns)

    def _is_station_selected(self, station: str) -> bool:
        """Return True if station passes ignore + optional CLI selection.

        - ignored_stations always exclude stations from discovery/DB and queues.
        - CLI --stations acts as an optional allow-list with partial matching.
        - If --stations is not provided, all non-ignored stations are selected.
        """
        if self._is_station_ignored(station):
            return False
        if not self._explicit_station_filters:
            return True
        return any(self._station_matches_pattern(station, pattern) for pattern in self._explicit_station_filters)

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
            if not self._is_station_selected(metadata.station):
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
        """Recalculate persisted priority scores for all registered videos.

        This ensures priority changes in config are applied even when videos
        are already present in the database and discovery is skipped.
        """
        updates: Dict[str, float] = {}
        for job in self.state_manager.get_all_videos():
            try:
                job_date = date_cls.fromisoformat(job.date)
            except ValueError:
                LOGGER.debug("Skipping priority recalculation for invalid date '%s' (%s)", job.date, job.video_id)
                continue
            updates[job.video_id] = self.calculate_priority(job.station, job.year, job_date)

        updated_count = self.state_manager.update_video_priorities(updates)
        LOGGER.info("Recalculated priorities for %s video(s)", updated_count)

        # Invalidate cached queues so fresh priorities are pulled from the database.
        self.stage_queues.clear()

    def build_job_queue(self, stage: ProcessingStage) -> "queue.PriorityQueue[Tuple[float, str, VideoJob]]":
        pending = self.state_manager.get_pending_jobs(stage)
        pq: "queue.PriorityQueue[Tuple[float, str, VideoJob]]" = queue.PriorityQueue()
        for job in pending:
            # Apply station/year filters at queue-build time as well, so
            # previously discovered jobs for other stations/years are not
            # processed when a subset is requested on the CLI.
            if not self._is_station_selected(job.station):
                continue
            if self.allowed_years and job.year not in self.allowed_years:
                continue
            pq.put((-job.priority_score, job.video_id, job))
        self.stage_queues[stage] = pq
        return pq

    def get_next_job(self, stage: ProcessingStage) -> Optional[VideoJob]:
        pq = self.stage_queues.get(stage)
        if pq is None or pq.empty():
            pq = self.build_job_queue(stage)
            if pq.empty():
                return None
        try:
            _, _, job = pq.get_nowait()
        except queue.Empty:
            return None
        return job

    def return_job(self, job: VideoJob, stage: ProcessingStage) -> None:
        pq = self.stage_queues.setdefault(stage, queue.PriorityQueue())
        pq.put((-job.priority_score, job.video_id, job))

    def calculate_priority(self, station: str, year: int, job_date: date_cls) -> float:
        """Score videos using year/station priority lists and recency."""
        year_priorities = {
            value: len(self.config.year_priority) - idx for idx, value in enumerate(self.config.year_priority)
        }

        year_weight = year_priorities.get(year, 0)
        station_weight = 0
        for idx, pattern in enumerate(self.config.station_priority):
            if self._station_matches_pattern(station, pattern):
                station_weight = len(self.config.station_priority) - idx
                break
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

        def _safe_parse(stamp: str, fmt: str, context: str) -> Optional[datetime]:
            try:
                return datetime.strptime(stamp, fmt)
            except ValueError as exc:
                LOGGER.debug("Skipping %s token '%s' in %s: %s", context, stamp, video_path, exc)
                return None

        match = _STATION_TIMESTAMP_RE.search(stem)
        if match:
            station = match.group("station").upper()
            stamp = match.group("stamp")
            timestamp = _safe_parse(stamp, "%Y%m%dT%H%M%S", "timestamp")
            if timestamp:
                video_id = f"{station}_{stamp}"
                return DiscoveredVideo(video_path, video_id, station, timestamp, name)

        # Support names like: Rost3_20200430_000001.avi
        match = _STATION_UNDERSCORE_TIMESTAMP_RE.search(stem)
        if match:
            station = match.group("station").upper()
            date_token = match.group("date")
            time_token = match.group("time")
            stamp = f"{date_token}{time_token}"
            timestamp = _safe_parse(stamp, "%Y%m%d%H%M%S", "underscore timestamp")
            if timestamp:
                formatted = timestamp.strftime("%Y%m%dT%H%M%S")
                video_id = f"{station}_{formatted}"
                return DiscoveredVideo(video_path, video_id, station, timestamp, name)

        match = _STATION_COMPACT_TIMESTAMP_RE.search(stem)
        if match:
            station = match.group("station").upper()
            stamp = match.group("stamp")
            timestamp = _safe_parse(stamp, "%Y%m%d%H%M%S", "compact timestamp")
            if timestamp:
                iso_stamp = f"{stamp[:8]}T{stamp[8:]}"
                video_id = f"{station}_{iso_stamp}"
                return DiscoveredVideo(video_path, video_id, station, timestamp, name)

        # Support XProtect-like names such as:
        #   55_ROST3_(192.168.1.130)_2025-05-23_21.00.00_22111.mp4
        match = _STATION_XPROTECT_RE.search(stem)
        if match:
            station = match.group("station").upper()
            date_token = re.sub(r"[^0-9]", "", match.group("date") or "")
            time_token = re.sub(r"[^0-9]", "", match.group("time") or "")
            stamp = f"{date_token}{time_token}"
            timestamp = _safe_parse(stamp, "%Y%m%d%H%M%S", "xprotect timestamp")
            if timestamp:
                formatted = timestamp.strftime("%Y%m%dT%H%M%S")
                video_id = f"{station}_{formatted}"
                return DiscoveredVideo(video_path, video_id, station, timestamp, name)

        match = _STATION_HUMAN_TIMESTAMP_RE.search(stem)
        if match:
            station = match.group("station").upper()
            date_token = re.sub(r"[^0-9]", "", match.group("date") or "")
            time_token = match.group("time")
            time_digits = re.sub(r"[^0-9]", "", time_token) if time_token else ""
            if len(date_token) == 8:
                if time_digits and len(time_digits) == 6:
                    stamp = f"{date_token}{time_digits}"
                    timestamp = _safe_parse(stamp, "%Y%m%d%H%M%S", "human timestamp")
                    if timestamp:
                        formatted = timestamp.strftime("%Y%m%dT%H%M%S")
                        video_id = f"{station}_{formatted}"
                        return DiscoveredVideo(video_path, video_id, station, timestamp, name)
                else:
                    timestamp = _safe_parse(date_token, "%Y%m%d", "human date")
                    if timestamp:
                        video_id = f"{station}_{timestamp.strftime('%Y%m%d')}_{self._stable_suffix(video_path)}"
                        return DiscoveredVideo(video_path, video_id, station, timestamp, name)

        match = _STATION_DATE_RE.search(stem)
        if match:
            station = match.group("station").upper()
            token = match.group("date")
            year = int(token[:4])
            if 2000 <= year <= 2100:
                timestamp = _safe_parse(token, "%Y%m%d", "date token")
                if timestamp:
                    video_id = f"{station}_{token}_{self._stable_suffix(video_path)}"
                    return DiscoveredVideo(video_path, video_id, station, timestamp, name)
            else:
                LOGGER.debug("Skipping unrealistic date token '%s' in %s", token, video_path)

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
            if _DATE_DIR_RE.fullmatch(part):
                continue
            if _YEAR_DIR_RE.fullmatch(part):
                continue
            return part.upper()
        return video_path.stem.upper()

    @staticmethod
    def _stable_suffix(video_path: Path) -> str:
        digest = hashlib.md5(str(video_path).encode("utf-8")).hexdigest()
        return digest[:6].upper()
