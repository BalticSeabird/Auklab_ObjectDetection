#!/usr/bin/env python3
"""
Batch reassess all events in event databases using the trained Stage4 model.

This script:
1. Loads the Stage4 model
2. Queries all events from event databases
3. Extracts Stage4 features from detection CSVs
4. Runs model prediction on each event
5. Updates the database with new predictions

Usage:
    python code/inference_system/stage4_batch_runner.py \
        --events-db-root data/events_db \
        --model-path models/stage4/tri3_fish_arrival_model.json \
        --stations TRI3,TRI6,ROST2,ROST3
"""

import argparse
import logging
import sqlite3
import sys
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime

import pandas as pd

PROJECT_CODE_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_CODE_DIR))

from inference_system.stage4_modeling import Stage4ModelArtifact, extract_stage4_features

LOGGER = logging.getLogger(__name__)


def setup_logging(log_dir: Path) -> None:
    """Configure logging."""
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"stage4_batch_reassessment_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout),
        ],
    )
    LOGGER.info(f"Logging to {log_file}")


class Stage4BatchReassessor:
    """Batch reassess events using Stage4 model."""

    def __init__(
        self,
        events_db_root: Path,
        model_path: Path,
        stations: List[str],
    ) -> None:
        """Initialize the reassessor."""
        self.events_db_root = Path(events_db_root)
        self.model_path = Path(model_path)
        self.stations = stations
        self.model_artifact: Optional[Stage4ModelArtifact] = None
        self._load_model()

    def _load_model(self) -> None:
        """Load the Stage4 model artifact."""
        if not self.model_path.exists():
            LOGGER.error(f"Model file not found: {self.model_path}")
            return

        try:
            self.model_artifact = Stage4ModelArtifact.from_path(self.model_path)
            LOGGER.info(f"Loaded Stage4 model from {self.model_path}")
        except Exception as e:
            LOGGER.error(f"Failed to load model: {e}")
            self.model_artifact = None

    def _load_detections(self, csv_path: str) -> pd.DataFrame:
        """Safely load detections CSV."""
        try:
            if not csv_path or not Path(csv_path).exists():
                return pd.DataFrame()
            return pd.read_csv(str(csv_path))
        except Exception as e:
            LOGGER.warning(f"Failed to load detections from {csv_path}: {e}")
            return pd.DataFrame()

    def _classify_event(self, event_row: dict) -> Dict:
        """Classify a single event using the model."""
        detections_csv = event_row.get("detections_csv_path")
        if not detections_csv:
            return {"is_new_fish_arrival": 0, "model_score": None, "decision_source": "rules"}

        detections = self._load_detections(detections_csv)
        if detections.empty:
            return {"is_new_fish_arrival": 0, "model_score": None, "decision_source": "rules"}

        stage2_flag = int(event_row.get("arrival_with_fish_stage2", 0))
        features = extract_stage4_features(detections, stage2_flag=stage2_flag)

        if self.model_artifact is None:
            return {"is_new_fish_arrival": 0, "model_score": None, "decision_source": "no_model"}

        try:
            model_score = float(self.model_artifact.predict_proba(features))
            is_new_fish_arrival = 1 if model_score >= self.model_artifact.threshold else 0
            return {"is_new_fish_arrival": is_new_fish_arrival, "model_score": model_score, "decision_source": "model"}
        except Exception as e:
            LOGGER.warning(f"Error predicting: {e}")
            return {"is_new_fish_arrival": 0, "model_score": None, "decision_source": "error"}

    def process_station(self, station: str) -> bool:
        """Process all events in a station database."""
        db_path = self.events_db_root / f"{station}_events.db"

        if not db_path.exists():
            LOGGER.warning(f"Database not found: {db_path}")
            return False

        try:
            LOGGER.info(f"Processing station: {station}")
            conn = sqlite3.connect(db_path)
            conn.row_factory = sqlite3.Row

            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM events
                WHERE event_type = 'arrival'
                AND is_actual_arrival = 1
                AND stage3_status = 'completed'
                AND detections_csv_path IS NOT NULL
            """)

            events = cursor.fetchall()
            LOGGER.info(f"Found {len(events)} events to reassess")

            updated = 0
            for idx, event in enumerate(events, 1):
                event_id = event["event_id"]
                event_dict = dict(event)
                result = self._classify_event(event_dict)

                try:
                    conn.execute("""
                        UPDATE events
                        SET is_new_fish_arrival = ?, stage4_model_score = ?, stage4_decision_source = ?, updated_at = CURRENT_TIMESTAMP
                        WHERE event_id = ?
                    """, (result["is_new_fish_arrival"], result["model_score"], result["decision_source"], event_id))
                    updated += 1

                    if idx % 50 == 0:
                        LOGGER.info(f"  Processed {idx}/{len(events)}")
                except Exception as e:
                    LOGGER.warning(f"Failed to update {event_id}: {e}")

            conn.commit()
            conn.close()
            LOGGER.info(f"Station {station} complete: {updated} updated")
            return True

        except Exception as e:
            LOGGER.error(f"Error processing {station}: {e}", exc_info=True)
            return False

    def run(self) -> bool:
        """Run batch reassessment."""
        if self.model_artifact is None:
            LOGGER.error("Model not loaded")
            return False

        LOGGER.info(f"Starting batch reassessment for: {self.stations}")
        success_count = sum(1 for station in self.stations if self.process_station(station))
        LOGGER.info(f"Complete: {success_count}/{len(self.stations)} stations")
        return success_count == len(self.stations)


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Batch reassess events using Stage4 model")
    parser.add_argument("--events-db-root", type=Path, default=Path("data/events_db"))
    parser.add_argument("--model-path", type=Path, default=Path("models/stage4/tri3_fish_arrival_model.json"))
    parser.add_argument("--stations", type=str, default="TRI3,TRI6,ROST2,ROST3,ROST4,ROST5,ROST6,FAR3,FAR6,BONDEN5,BONDEN6")
    parser.add_argument("--log-dir", type=Path, default=Path("logs"))

    args = parser.parse_args()
    setup_logging(args.log_dir)

    stations = [s.strip() for s in args.stations.split(",") if s.strip()]
    reassessor = Stage4BatchReassessor(events_db_root=args.events_db_root, model_path=args.model_path, stations=stations)
    success = reassessor.run()
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
