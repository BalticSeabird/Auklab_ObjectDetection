"""Per-station SQLite persistence for pipeline events."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import pandas as pd

from .path_utils import get_station_event_db_path


class EventDatabase:
    """Simple upsert/read API around per-station event SQLite DB files."""

    def __init__(self, db_path: Path) -> None:
        self.db_path = Path(db_path)

    @classmethod
    def for_station(cls, config: Any, station: str) -> "EventDatabase":
        return cls(get_station_event_db_path(config, station))

    def initialize(self) -> None:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.execute("PRAGMA journal_mode=WAL;")
            conn.execute("PRAGMA foreign_keys=ON;")
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS schema_migrations (
                    version INTEGER PRIMARY KEY,
                    applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );

                CREATE TABLE IF NOT EXISTS events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    station TEXT NOT NULL,
                    video_id TEXT NOT NULL,
                    year INTEGER NOT NULL,
                    date TEXT NOT NULL,
                    filename TEXT NOT NULL,
                    event_id TEXT NOT NULL,
                    event_type TEXT,
                    second REAL,
                    before_mean REAL,
                    after_mean REAL,
                    arrival_with_fish_stage2 INTEGER,
                    fish_count REAL,
                    fish_mean_area REAL,
                    fish_max_area REAL,
                    absolute_timestamp TEXT,
                    original_video_path TEXT,
                    event_video_path TEXT,
                    detections_csv_path TEXT,
                    stage3_status TEXT,
                    is_actual_arrival INTEGER,
                    is_new_fish_arrival INTEGER,
                    fish_detections_stage4 INTEGER,
                    fish_avg_confidence_stage4 REAL,
                    stage4_rule_version TEXT,
                    stage4_rule_hits TEXT,
                    stage4_features TEXT,
                    stage4_model_score REAL,
                    stage4_decision_source TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(station, event_id)
                );

                CREATE INDEX IF NOT EXISTS idx_events_video_id
                ON events(video_id);

                CREATE INDEX IF NOT EXISTS idx_events_abs_ts
                ON events(absolute_timestamp);

                CREATE INDEX IF NOT EXISTS idx_events_type
                ON events(event_type);
                """
            )
            self._ensure_events_columns(conn)
            conn.commit()

    def _ensure_events_columns(self, conn: sqlite3.Connection) -> None:
        """Apply lightweight schema migrations for existing event DBs."""
        rows = conn.execute("PRAGMA table_info(events)").fetchall()
        existing = {row[1] for row in rows}

        if "fish_detections_stage4" not in existing:
            conn.execute("ALTER TABLE events ADD COLUMN fish_detections_stage4 INTEGER")
        if "fish_avg_confidence_stage4" not in existing:
            conn.execute("ALTER TABLE events ADD COLUMN fish_avg_confidence_stage4 REAL")
        if "stage4_model_score" not in existing:
            conn.execute("ALTER TABLE events ADD COLUMN stage4_model_score REAL")
        if "stage4_decision_source" not in existing:
            conn.execute("ALTER TABLE events ADD COLUMN stage4_decision_source TEXT")

    def upsert_events(self, *, station: str, job: Any, events_df: pd.DataFrame) -> int:
        """Insert or update stage2 event rows for a single source video."""
        if events_df.empty:
            return 0

        payloads: List[tuple] = []
        for _, row in events_df.iterrows():
            payloads.append(
                (
                    station,
                    job.video_id,
                    int(job.year),
                    job.date,
                    job.filename,
                    str(row.get("event_id", "")),
                    self._as_text(row.get("type")),
                    self._as_float(row.get("second")),
                    self._as_float(row.get("before_mean")),
                    self._as_float(row.get("after_mean")),
                    self._as_int_bool(row.get("arrival_with_fish")),
                    self._as_float(row.get("fish_count")),
                    self._as_float(row.get("fish_mean_area")),
                    self._as_float(row.get("fish_max_area")),
                    self._as_timestamp_text(row.get("absolute_timestamp")),
                    self._as_text(row.get("original_video_path")),
                    self._as_text(row.get("event_video_path")),
                )
            )

        with self._connect() as conn:
            conn.executemany(
                """
                INSERT INTO events (
                    station, video_id, year, date, filename, event_id,
                    event_type, second, before_mean, after_mean,
                    arrival_with_fish_stage2, fish_count, fish_mean_area, fish_max_area,
                    absolute_timestamp, original_video_path, event_video_path,
                    updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                ON CONFLICT(station, event_id) DO UPDATE SET
                    video_id = excluded.video_id,
                    year = excluded.year,
                    date = excluded.date,
                    filename = excluded.filename,
                    event_type = excluded.event_type,
                    second = excluded.second,
                    before_mean = excluded.before_mean,
                    after_mean = excluded.after_mean,
                    arrival_with_fish_stage2 = excluded.arrival_with_fish_stage2,
                    fish_count = excluded.fish_count,
                    fish_mean_area = excluded.fish_mean_area,
                    fish_max_area = excluded.fish_max_area,
                    absolute_timestamp = excluded.absolute_timestamp,
                    original_video_path = excluded.original_video_path,
                    event_video_path = excluded.event_video_path,
                    updated_at = CURRENT_TIMESTAMP
                """,
                payloads,
            )
            conn.commit()
        return len(payloads)

    def fetch_events_for_video(self, *, station: str, video_id: str) -> pd.DataFrame:
        with self._connect() as conn:
            return pd.read_sql_query(
                """
                SELECT *
                FROM events
                WHERE station = ? AND video_id = ?
                ORDER BY COALESCE(second, 0) ASC, event_id ASC
                """,
                conn,
                params=(station, video_id),
            )

    def update_stage3_artifacts(
        self,
        *,
        station: str,
        event_id: str,
        event_video_path: Optional[str],
        detections_csv_path: Optional[str],
        stage3_status: str,
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE events
                SET event_video_path = COALESCE(?, event_video_path),
                    detections_csv_path = COALESCE(?, detections_csv_path),
                    stage3_status = ?,
                    updated_at = CURRENT_TIMESTAMP
                WHERE station = ? AND event_id = ?
                """,
                (event_video_path, detections_csv_path, stage3_status, station, event_id),
            )
            conn.commit()

    def update_stage4_labels(
        self,
        *,
        station: str,
        event_id: str,
        is_actual_arrival: int,
        is_new_fish_arrival: int,
        fish_detections_stage4: int,
        fish_avg_confidence_stage4: float,
        rule_version: str,
        rule_hits: Iterable[str],
        features: Dict[str, Any],
        model_score: Optional[float] = None,
        decision_source: Optional[str] = None,
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE events
                SET is_actual_arrival = ?,
                    is_new_fish_arrival = ?,
                    fish_detections_stage4 = ?,
                    fish_avg_confidence_stage4 = ?,
                    stage4_rule_version = ?,
                    stage4_rule_hits = ?,
                    stage4_features = ?,
                    stage4_model_score = ?,
                    stage4_decision_source = ?,
                    updated_at = CURRENT_TIMESTAMP
                WHERE station = ? AND event_id = ?
                """,
                (
                    int(is_actual_arrival),
                    int(is_new_fish_arrival),
                    int(fish_detections_stage4),
                    float(fish_avg_confidence_stage4),
                    rule_version,
                    json.dumps(list(rule_hits)),
                    json.dumps(features, default=str),
                    self._as_float(model_score),
                    self._as_text(decision_source),
                    station,
                    event_id,
                ),
            )
            conn.commit()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    @staticmethod
    def _as_text(value: Any) -> Optional[str]:
        if value is None:
            return None
        if pd.isna(value):
            return None
        return str(value)

    @staticmethod
    def _as_float(value: Any) -> Optional[float]:
        if value is None:
            return None
        try:
            if pd.isna(value):
                return None
        except TypeError:
            pass
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _as_int_bool(value: Any) -> Optional[int]:
        if value is None:
            return None
        try:
            if pd.isna(value):
                return None
        except TypeError:
            pass
        return 1 if bool(value) else 0

    @staticmethod
    def _as_timestamp_text(value: Any) -> Optional[str]:
        if value is None:
            return None
        try:
            if pd.isna(value):
                return None
        except TypeError:
            pass
        try:
            return pd.to_datetime(value).isoformat()
        except Exception:
            return str(value)
