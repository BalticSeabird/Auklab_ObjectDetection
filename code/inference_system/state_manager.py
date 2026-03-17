"""Persistent state tracking for the multi-stage inference pipeline."""

from __future__ import annotations

import json
import sqlite3
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum, IntEnum
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional

ISO_TIMESTAMP_FORMAT = "%Y-%m-%d %H:%M:%S"


class ProcessingStage(IntEnum):
    """Enumeration of pipeline stages handled by the orchestrator."""

    STAGE1 = 1  # Video inference
    STAGE2 = 2  # Event detection


class StageStatus(str, Enum):
    """Lifecycle states for a stage row."""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass(slots=True)
class VideoJob:
    """Represents a unit of work tied to a specific video file."""

    video_id: str
    station: str
    year: int
    date: str
    filename: str
    filepath: Path
    priority_score: float
    stage: ProcessingStage = ProcessingStage.STAGE1
    retry_count: int = 0


@dataclass(slots=True)
class ProgressSummary:
    """Aggregate view of system progress."""

    total_videos: int
    completed_videos: int
    failed_videos: int
    in_progress_videos: int
    pending_videos: int
    stage_breakdown: Dict[ProcessingStage, Dict[StageStatus, int]]


class StateManager:
    """SQLite-backed persistence engine for pipeline bookkeeping."""

    def __init__(self, db_path: Path | str, *, connection_timeout: float = 30.0) -> None:
        self.db_path = Path(db_path)
        self.connection_timeout = connection_timeout
        self._init_lock = threading.Lock()

    def initialize_db(self) -> None:
        """Create schema and indexes if they do not exist."""
        with self._init_lock:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            with self._connect() as conn:
                cursor = conn.cursor()
                cursor.execute("PRAGMA journal_mode=WAL;")
                cursor.execute("PRAGMA foreign_keys=ON;")
                cursor.executescript(
                    """
                    CREATE TABLE IF NOT EXISTS videos (
                        video_id TEXT PRIMARY KEY,
                        station TEXT NOT NULL,
                        year INTEGER NOT NULL,
                        date TEXT NOT NULL,
                        filename TEXT NOT NULL,
                        filepath TEXT NOT NULL,
                        priority_score REAL NOT NULL,
                        discovered_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    );

                    CREATE TABLE IF NOT EXISTS processing_stages (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        video_id TEXT NOT NULL,
                        stage INTEGER NOT NULL,
                        status TEXT NOT NULL,
                        worker_id TEXT,
                        started_at TIMESTAMP,
                        completed_at TIMESTAMP,
                        duration_seconds REAL,
                        error_message TEXT,
                        retry_count INTEGER DEFAULT 0,
                        metadata TEXT,
                        FOREIGN KEY (video_id) REFERENCES videos(video_id),
                        UNIQUE(video_id, stage)
                    );

                    CREATE TABLE IF NOT EXISTS performance_metrics (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        video_id TEXT NOT NULL,
                        stage INTEGER NOT NULL,
                        video_duration_seconds REAL,
                        processing_duration_seconds REAL,
                        fps_processed REAL,
                        detections_count INTEGER,
                        events_count INTEGER,
                        clips_count INTEGER,
                        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        FOREIGN KEY (video_id) REFERENCES videos(video_id)
                    );

                    CREATE TABLE IF NOT EXISTS system_state (
                        key TEXT PRIMARY KEY,
                        value TEXT,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    );

                    CREATE TABLE IF NOT EXISTS corrupt_videos (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        video_id TEXT NOT NULL,
                        stage INTEGER NOT NULL,
                        error_message TEXT,
                        filepath TEXT,
                        station TEXT,
                        year INTEGER,
                        date TEXT,
                        recorded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        FOREIGN KEY (video_id) REFERENCES videos(video_id)
                    );

                    CREATE INDEX IF NOT EXISTS idx_processing_stage_status
                    ON processing_stages(status);

                    CREATE INDEX IF NOT EXISTS idx_processing_stage_stage_status
                    ON processing_stages(stage, status);

                    CREATE INDEX IF NOT EXISTS idx_videos_station_year
                    ON videos(station, year);
                    """
                )
                conn.commit()

    def discover_videos(self, videos: Iterable[VideoJob]) -> int:
        """Register discovered videos and ensure stage placeholders exist."""
        return self.register_videos(videos)

    def register_videos(self, videos: Iterable[VideoJob]) -> int:
        count = 0
        with self._connect() as conn:
            cursor = conn.cursor()
            for video in videos:
                payload = (
                    video.video_id,
                    video.station,
                    video.year,
                    video.date,
                    video.filename,
                    str(video.filepath),
                    float(video.priority_score),
                )
                cursor.execute(
                    """
                    INSERT INTO videos (video_id, station, year, date, filename, filepath, priority_score)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(video_id) DO UPDATE SET
                        station = excluded.station,
                        year = excluded.year,
                        date = excluded.date,
                        filename = excluded.filename,
                        filepath = excluded.filepath,
                        priority_score = excluded.priority_score
                    """,
                    payload,
                )
                for stage in ProcessingStage:
                    cursor.execute(
                        """
                        INSERT INTO processing_stages (video_id, stage, status)
                        VALUES (?, ?, ?)
                        ON CONFLICT(video_id, stage) DO NOTHING
                        """,
                        (video.video_id, int(stage), StageStatus.PENDING.value),
                    )
                count += 1
            conn.commit()
        return count

    def get_pending_jobs(self, stage: ProcessingStage, limit: Optional[int] = None) -> List[VideoJob]:
        dependency_join = ""
        params: List[Any] = [int(stage), StageStatus.PENDING.value]

        if stage != ProcessingStage.STAGE1:
            prev_stage = ProcessingStage(stage - 1)
            dependency_join = (
                "JOIN processing_stages prev ON prev.video_id = v.video_id "
                "AND prev.stage = ? AND prev.status = ?"
            )
            params.extend([int(prev_stage), StageStatus.COMPLETED.value])

        query = (
            f"""
            SELECT v.video_id, v.station, v.year, v.date, v.filename, v.filepath,
                   v.priority_score, ps.retry_count
            FROM videos v
            JOIN processing_stages ps ON v.video_id = ps.video_id
            {dependency_join}
            WHERE ps.stage = ? AND ps.status = ?
            ORDER BY v.priority_score DESC, v.discovered_at ASC
            """
        )
        if limit:
            query += " LIMIT ?"
            params.append(limit)

        with self._connect() as conn:
            cursor = conn.cursor()
            cursor.execute(query, params)
            rows = cursor.fetchall()
        return [
            VideoJob(
                video_id=row["video_id"],
                station=row["station"],
                year=row["year"],
                date=row["date"],
                filename=row["filename"],
                filepath=Path(row["filepath"]),
                priority_score=row["priority_score"],
                stage=stage,
                retry_count=row["retry_count"],
            )
            for row in rows
        ]

    def mark_job_started(self, video_id: str, stage: ProcessingStage, worker_id: str) -> None:
        with self._connect() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                UPDATE processing_stages
                SET status = ?, worker_id = ?, started_at = CURRENT_TIMESTAMP, completed_at = NULL,
                    duration_seconds = NULL, error_message = NULL
                WHERE video_id = ? AND stage = ? AND status = ?
                """,
                (StageStatus.IN_PROGRESS.value, worker_id, video_id, int(stage), StageStatus.PENDING.value),
            )
            if cursor.rowcount == 0:
                raise ValueError(f"Job {video_id} stage {stage} is not pending")
            conn.commit()

    def mark_job_completed(
        self,
        video_id: str,
        stage: ProcessingStage,
        *,
        metadata: Optional[Dict[str, Any]] = None,
        duration_seconds: Optional[float] = None,
    ) -> None:
        with self._connect() as conn:
            cursor = conn.cursor()
            computed_duration = duration_seconds or self._calculate_duration(conn, video_id, stage)
            cursor.execute(
                """
                UPDATE processing_stages
                SET status = ?, completed_at = CURRENT_TIMESTAMP, duration_seconds = ?,
                    metadata = ?
                WHERE video_id = ? AND stage = ?
                """,
                (
                    StageStatus.COMPLETED.value,
                    computed_duration,
                    self._serialize_json(metadata),
                    video_id,
                    int(stage),
                ),
            )
            if cursor.rowcount == 0:
                raise ValueError(f"No job found for video {video_id} stage {stage}")
            conn.commit()

    def mark_job_failed(
        self,
        video_id: str,
        stage: ProcessingStage,
        *,
        error_message: str,
        retryable: bool,
    ) -> None:
        target_status = StageStatus.PENDING if retryable else StageStatus.FAILED
        with self._connect() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                UPDATE processing_stages
                SET status = ?, error_message = ?, completed_at = CURRENT_TIMESTAMP,
                    retry_count = retry_count + 1
                WHERE video_id = ? AND stage = ?
                """,
                (target_status.value, error_message, video_id, int(stage)),
            )
            if cursor.rowcount == 0:
                raise ValueError(f"No job found for video {video_id} stage {stage}")

            # For non-retryable failures that indicate corrupt or unreadable
            # video files, persist a row in the corrupt_videos table so they
            # can be analyzed separately.
            if not retryable and error_message:
                if (
                    "Invalid data found when processing input" in error_message
                    or "appears corrupt or unreadable" in error_message
                ):
                    cursor.execute(
                        """
                        INSERT INTO corrupt_videos (video_id, stage, error_message, filepath, station, year, date)
                        SELECT v.video_id, ?, ?, v.filepath, v.station, v.year, v.date
                        FROM videos v
                        WHERE v.video_id = ?
                        """,
                        (int(stage), error_message, video_id),
                    )
            conn.commit()

    def return_job(self, video_id: str, stage: ProcessingStage) -> None:
        with self._connect() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                UPDATE processing_stages
                SET status = ?, worker_id = NULL, started_at = NULL, completed_at = NULL
                WHERE video_id = ? AND stage = ?
                """,
                (StageStatus.PENDING.value, video_id, int(stage)),
            )
            conn.commit()

    def record_performance_metrics(
        self,
        video_id: str,
        stage: ProcessingStage,
        *,
        video_duration_seconds: Optional[float] = None,
        processing_duration_seconds: Optional[float] = None,
        fps_processed: Optional[float] = None,
        detections_count: Optional[int] = None,
        events_count: Optional[int] = None,
        clips_count: Optional[int] = None,
    ) -> None:
        with self._connect() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO performance_metrics (
                    video_id, stage, video_duration_seconds, processing_duration_seconds,
                    fps_processed, detections_count, events_count, clips_count
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    video_id,
                    int(stage),
                    video_duration_seconds,
                    processing_duration_seconds,
                    fps_processed,
                    detections_count,
                    events_count,
                    clips_count,
                ),
            )
            conn.commit()

    def get_progress_summary(self) -> ProgressSummary:
        with self._connect() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM videos")
            total_videos = cursor.fetchone()[0]

            stage_breakdown: Dict[ProcessingStage, Dict[StageStatus, int]] = {
                stage: {status: 0 for status in StageStatus}
                for stage in ProcessingStage
            }
            cursor.execute(
                """
                SELECT stage, status, COUNT(*) as cnt
                FROM processing_stages
                GROUP BY stage, status
                """
            )
            for row in cursor.fetchall():
                stage_breakdown[ProcessingStage(row["stage"])][StageStatus(row["status"])] = row["cnt"]

            final_stage = max(ProcessingStage)
            final_stage_counts = stage_breakdown[final_stage]
            summary = ProgressSummary(
                total_videos=total_videos,
                completed_videos=final_stage_counts[StageStatus.COMPLETED],
                failed_videos=final_stage_counts[StageStatus.FAILED],
                in_progress_videos=final_stage_counts[StageStatus.IN_PROGRESS],
                pending_videos=final_stage_counts[StageStatus.PENDING],
                stage_breakdown=stage_breakdown,
            )
            return summary

    def is_all_complete(self) -> bool:
        with self._connect() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT COUNT(*) AS remaining
                FROM processing_stages
                WHERE stage = ? AND status NOT IN (?, ?)
                """,
                (
                    int(max(ProcessingStage)),
                    StageStatus.COMPLETED.value,
                    StageStatus.SKIPPED.value,
                ),
            )
            remaining = cursor.fetchone()[0]
        return remaining == 0

    def get_videos_for_station_date(self, station: str, date: str) -> List[VideoJob]:
        """Return all registered videos for a given station/date pair."""
        with self._connect() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT video_id, station, year, date, filename, filepath, priority_score
                FROM videos
                WHERE station = ? AND date = ?
                ORDER BY discovered_at ASC
                """,
                (station, date),
            )
            rows = cursor.fetchall()
        return [
            VideoJob(
                video_id=row["video_id"],
                station=row["station"],
                year=row["year"],
                date=row["date"],
                filename=row["filename"],
                filepath=Path(row["filepath"]),
                priority_score=row["priority_score"],
            )
            for row in rows
        ]

    def get_all_videos(self) -> List[VideoJob]:
        """Return all registered videos.

        Used for maintenance tasks like priority recalculation.
        """
        with self._connect() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT video_id, station, year, date, filename, filepath, priority_score
                FROM videos
                """
            )
            rows = cursor.fetchall()
        return [
            VideoJob(
                video_id=row["video_id"],
                station=row["station"],
                year=row["year"],
                date=row["date"],
                filename=row["filename"],
                filepath=Path(row["filepath"]),
                priority_score=row["priority_score"],
            )
            for row in rows
        ]

    def update_video_priorities(self, updates: Dict[str, float]) -> int:
        """Update priority scores for existing videos.

        Args:
            updates: Mapping of video_id -> priority_score.

        Returns:
            Number of rows updated.
        """
        if not updates:
            return 0
        with self._connect() as conn:
            cursor = conn.cursor()
            cursor.executemany(
                """
                UPDATE videos
                SET priority_score = ?
                WHERE video_id = ?
                """,
                [(float(priority), video_id) for video_id, priority in updates.items()],
            )
            conn.commit()
            return cursor.rowcount

    def reset_stuck_jobs(self, timeout_seconds: int) -> int:
        """Requeue jobs stuck in progress for longer than the threshold."""
        threshold = datetime.utcnow() - timedelta(seconds=timeout_seconds)
        cutoff = threshold.strftime(ISO_TIMESTAMP_FORMAT)
        with self._connect() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                UPDATE processing_stages
                SET status = ?, worker_id = NULL, started_at = NULL
                WHERE status = ? AND started_at IS NOT NULL AND started_at < ?
                """,
                (StageStatus.PENDING.value, StageStatus.IN_PROGRESS.value, cutoff),
            )
            conn.commit()
            return cursor.rowcount

    def reset_failed_jobs(self, *, stage: Optional[ProcessingStage] = None) -> int:
        """Requeue failed jobs back to pending state.

        Useful for troubleshooting after code/config fixes where previously
        failed jobs should be retried without rebuilding discovery state.
        """
        with self._connect() as conn:
            cursor = conn.cursor()
            if stage is None:
                cursor.execute(
                    """
                    UPDATE processing_stages
                    SET status = ?, worker_id = NULL, started_at = NULL, completed_at = NULL
                    WHERE status = ?
                    """,
                    (StageStatus.PENDING.value, StageStatus.FAILED.value),
                )
            else:
                cursor.execute(
                    """
                    UPDATE processing_stages
                    SET status = ?, worker_id = NULL, started_at = NULL, completed_at = NULL
                    WHERE status = ? AND stage = ?
                    """,
                    (StageStatus.PENDING.value, StageStatus.FAILED.value, int(stage)),
                )
            conn.commit()
            return cursor.rowcount

    def set_system_state(self, key: str, value: Any) -> None:
        with self._connect() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO system_state(key, value, updated_at)
                VALUES (?, ?, CURRENT_TIMESTAMP)
                ON CONFLICT(key) DO UPDATE SET value = excluded.value, updated_at = CURRENT_TIMESTAMP
                """,
                (key, json.dumps(value, default=str)),
            )
            conn.commit()

    def get_system_state(self, key: str) -> Optional[Any]:
        with self._connect() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT value FROM system_state WHERE key = ?", (key,))
            row = cursor.fetchone()
        if not row:
            return None
        try:
            return json.loads(row[0])
        except json.JSONDecodeError:
            return row[0]

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.db_path, timeout=self.connection_timeout)
        conn.row_factory = sqlite3.Row
        try:
            conn.execute("PRAGMA foreign_keys=ON;")
            yield conn
        finally:
            conn.close()

    def _calculate_duration(
        self,
        conn: sqlite3.Connection,
        video_id: str,
        stage: ProcessingStage,
    ) -> Optional[float]:
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT started_at FROM processing_stages
            WHERE video_id = ? AND stage = ?
            """,
            (video_id, int(stage)),
        )
        row = cursor.fetchone()
        if not row or row["started_at"] is None:
            return None
        started = datetime.strptime(row["started_at"], ISO_TIMESTAMP_FORMAT)
        return (datetime.utcnow() - started).total_seconds()

    @staticmethod
    def _serialize_json(metadata: Optional[Dict[str, Any]]) -> Optional[str]:
        if metadata is None:
            return None
        return json.dumps(metadata, default=str)
