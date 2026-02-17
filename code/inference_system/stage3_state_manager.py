"""Persistence utilities for the standalone stage3 clip extraction workflow."""

from __future__ import annotations

import sqlite3
import threading
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Iterable, List, Optional, Sequence


class Stage3BatchStatus(str, Enum):
    """Lifecycle states tracked for stage3 batches."""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass(slots=True)
class Stage3BatchRegistration:
    """Description of a station/date pair discovered from summarized events."""

    station: str
    date: str  # YYYY-MM-DD string
    year: int
    events_csv: Path


@dataclass(slots=True)
class Stage3BatchRecord:
    """Full record fetched from the stage3 state database."""

    station: str
    date: str
    year: int
    events_csv: Path
    status: Stage3BatchStatus
    clips_created: int
    clips_failed: int
    error_message: Optional[str]


class Stage3StateManager:
    """Lightweight SQLite wrapper for coordinating per-day clip extraction batches."""

    def __init__(self, db_path: Path | str, *, connection_timeout: float = 30.0) -> None:
        self.db_path = Path(db_path)
        self.connection_timeout = connection_timeout
        self._init_lock = threading.Lock()

    def initialize_db(self) -> None:
        with self._init_lock:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            with self._connect() as conn:
                cursor = conn.cursor()
                cursor.executescript(
                    """
                    CREATE TABLE IF NOT EXISTS stage3_batches (
                        station TEXT NOT NULL,
                        date TEXT NOT NULL,
                        year INTEGER NOT NULL,
                        events_csv TEXT NOT NULL,
                        status TEXT NOT NULL DEFAULT 'pending',
                        started_at TIMESTAMP,
                        completed_at TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        clips_created INTEGER DEFAULT 0,
                        clips_failed INTEGER DEFAULT 0,
                        error_message TEXT,
                        PRIMARY KEY (station, date)
                    );

                    CREATE INDEX IF NOT EXISTS idx_stage3_status ON stage3_batches(status);
                    CREATE INDEX IF NOT EXISTS idx_stage3_station_date ON stage3_batches(station, date);
                    """
                )
                conn.commit()

    def register_batches(self, batches: Iterable[Stage3BatchRegistration]) -> int:
        count = 0
        with self._connect() as conn:
            cursor = conn.cursor()
            for batch in batches:
                cursor.execute(
                    """
                    INSERT INTO stage3_batches (station, date, year, events_csv, status)
                    VALUES (?, ?, ?, ?, ?)
                    ON CONFLICT(station, date) DO UPDATE SET
                        year = excluded.year,
                        events_csv = excluded.events_csv
                    """,
                    (
                        batch.station,
                        batch.date,
                        batch.year,
                        str(batch.events_csv),
                        Stage3BatchStatus.PENDING.value,
                    ),
                )
                count += 1
            conn.commit()
        return count

    def fetch_batches(
        self,
        *,
        statuses: Optional[Sequence[Stage3BatchStatus]] = None,
        stations: Optional[Sequence[str]] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> List[Stage3BatchRecord]:
        clauses: List[str] = []
        params: List[object] = []

        if statuses:
            placeholders = ",".join("?" for _ in statuses)
            clauses.append(f"status IN ({placeholders})")
            params.extend(status.value for status in statuses)
        if stations:
            station_set = [station.upper() for station in stations]
            placeholders = ",".join("?" for _ in station_set)
            clauses.append(f"station IN ({placeholders})")
            params.extend(station_set)
        if start_date:
            clauses.append("date >= ?")
            params.append(start_date)
        if end_date:
            clauses.append("date <= ?")
            params.append(end_date)

        query = (
            "SELECT station, date, year, events_csv, status, clips_created, clips_failed, error_message "
            "FROM stage3_batches"
        )
        if clauses:
            query += " WHERE " + " AND ".join(clauses)
        query += " ORDER BY date ASC, station ASC"
        if limit:
            query += " LIMIT ?"
            params.append(limit)

        with self._connect() as conn:
            cursor = conn.cursor()
            cursor.execute(query, params)
            rows = cursor.fetchall()
        return [
            Stage3BatchRecord(
                station=row["station"],
                date=row["date"],
                year=row["year"],
                events_csv=Path(row["events_csv"]),
                status=Stage3BatchStatus(row["status"]),
                clips_created=row["clips_created"] or 0,
                clips_failed=row["clips_failed"] or 0,
                error_message=row["error_message"],
            )
            for row in rows
        ]

    def mark_batch_started(self, station: str, date: str) -> None:
        self._update_status(
            station,
            date,
            Stage3BatchStatus.IN_PROGRESS,
            extra_set="started_at = CURRENT_TIMESTAMP, completed_at = NULL, error_message = NULL",
        )

    def mark_batch_completed(self, station: str, date: str, *, clips_created: int, clips_failed: int) -> None:
        with self._connect() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                UPDATE stage3_batches
                SET status = ?, completed_at = CURRENT_TIMESTAMP, clips_created = ?,
                    clips_failed = ?, error_message = NULL
                WHERE station = ? AND date = ?
                """,
                (
                    Stage3BatchStatus.COMPLETED.value,
                    clips_created,
                    clips_failed,
                    station,
                    date,
                ),
            )
            conn.commit()

    def mark_batch_failed(self, station: str, date: str, *, error_message: str) -> None:
        with self._connect() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                UPDATE stage3_batches
                SET status = ?, completed_at = CURRENT_TIMESTAMP, error_message = ?
                WHERE station = ? AND date = ?
                """,
                (
                    Stage3BatchStatus.FAILED.value,
                    error_message,
                    station,
                    date,
                ),
            )
            conn.commit()

    def reset_batch(self, station: str, date: str) -> None:
        self._update_status(
            station,
            date,
            Stage3BatchStatus.PENDING,
            extra_set="started_at = NULL, completed_at = NULL, error_message = NULL",
        )

    def _update_status(
        self,
        station: str,
        date: str,
        status: Stage3BatchStatus,
        *,
        extra_set: Optional[str] = None,
    ) -> None:
        set_clause = "status = ?"
        if extra_set:
            set_clause = f"{set_clause}, {extra_set}"
        with self._connect() as conn:
            cursor = conn.cursor()
            cursor.execute(
                f"UPDATE stage3_batches SET {set_clause} WHERE station = ? AND date = ?",
                (status.value, station, date),
            )
            conn.commit()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=self.connection_timeout)
        conn.row_factory = sqlite3.Row
        return conn
