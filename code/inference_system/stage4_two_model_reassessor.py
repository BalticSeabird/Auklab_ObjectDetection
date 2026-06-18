#!/usr/bin/env python3
"""
Two-Model Inference Pipeline

Stage4 uses two separate models:
  Model 1: Detects if there's a TRUE bird arrival
  Model 2: Detects if there's fish at the arrival (only run if Model 1 = YES)

Final output: is_new_fish_arrival = (Model1 > threshold) AND (Model2 > threshold)

Usage:
    python code/inference_system/stage4_two_model_reassessor.py \
        --events-db-root data/events_db \
        --model1-path models/stage4/model1_bird_arrival.json \
        --model2-path models/stage4/model2_fish_arrival.json \
        --stations TRI3,TRI6
"""

import argparse
import json
import logging
import sqlite3
import sys
from pathlib import Path
from typing import Dict, Optional
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
    log_file = log_dir / f"stage4_two_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout),
        ],
    )
    LOGGER.info(f"Logging to {log_file}")


class TwoModelStage4:
    """Two-model Stage4 classifier."""

    def __init__(self, model1_path: Path, model2_path: Path, stations: list):
        """Initialize with both models."""
        self.model1_path = Path(model1_path)
        self.model2_path = Path(model2_path)
        self.stations = stations
        self.model1: Optional[Stage4ModelArtifact] = None
        self.model2: Optional[Stage4ModelArtifact] = None
        self._load_models()

    def _load_models(self) -> None:
        """Load both model artifacts."""
        try:
            self.model1 = Stage4ModelArtifact.from_path(self.model1_path)
            LOGGER.info(f"Loaded Model 1 (bird arrival) from {self.model1_path}")
        except Exception as e:
            LOGGER.error(f"Failed to load Model 1: {e}")

        try:
            self.model2 = Stage4ModelArtifact.from_path(self.model2_path)
            LOGGER.info(f"Loaded Model 2 (fish arrival) from {self.model2_path}")
        except Exception as e:
            LOGGER.error(f"Failed to load Model 2: {e}")

    def _load_detections(self, csv_path: str) -> pd.DataFrame:
        """Safely load detections CSV."""
        try:
            if not csv_path or not Path(csv_path).exists():
                return pd.DataFrame()
            return pd.read_csv(str(csv_path))
        except Exception as e:
            LOGGER.warning(f"Failed to load detections: {e}")
            return pd.DataFrame()

    def classify_event(self, event_row: dict) -> Dict:
        """Classify using two-model pipeline."""
        detections_csv = event_row.get("detections_csv_path")
        if not detections_csv:
            return {
                "is_actual_arrival": 0,
                "is_new_fish_arrival": 0,
                "model1_score": None,
                "model2_score": None,
                "decision_source": "no_detections",
            }

        detections = self._load_detections(detections_csv)
        if detections.empty:
            return {
                "is_actual_arrival": 0,
                "is_new_fish_arrival": 0,
                "model1_score": None,
                "model2_score": None,
                "decision_source": "empty_detections",
            }

        stage2_flag = int(event_row.get("arrival_with_fish_stage2", 0))
        features = extract_stage4_features(detections, stage2_flag=stage2_flag)

        # Model 1: Is there a true bird arrival?
        if self.model1 is None:
            return {
                "is_actual_arrival": 0,
                "is_new_fish_arrival": 0,
                "model1_score": None,
                "model2_score": None,
                "decision_source": "no_model1",
            }

        try:
            model1_score = float(self.model1.predict_proba(features))
            is_actual_arrival = 1 if model1_score >= self.model1.threshold else 0
        except Exception as e:
            LOGGER.warning(f"Model 1 prediction error: {e}")
            return {
                "is_actual_arrival": 0,
                "is_new_fish_arrival": 0,
                "model1_score": None,
                "model2_score": None,
                "decision_source": "model1_error",
            }

        # If no actual arrival, skip model 2
        if is_actual_arrival == 0:
            return {
                "is_actual_arrival": 0,
                "is_new_fish_arrival": 0,
                "model1_score": model1_score,
                "model2_score": None,
                "decision_source": "no_arrival",
            }

        # Model 2: Is there fish at the arrival?
        if self.model2 is None:
            return {
                "is_actual_arrival": 1,
                "is_new_fish_arrival": 0,
                "model1_score": model1_score,
                "model2_score": None,
                "decision_source": "no_model2",
            }

        try:
            model2_score = float(self.model2.predict_proba(features))
            is_new_fish_arrival = 1 if model2_score >= self.model2.threshold else 0
        except Exception as e:
            LOGGER.warning(f"Model 2 prediction error: {e}")
            return {
                "is_actual_arrival": 1,
                "is_new_fish_arrival": 0,
                "model1_score": model1_score,
                "model2_score": None,
                "decision_source": "model2_error",
            }

        return {
            "is_actual_arrival": is_actual_arrival,
            "is_new_fish_arrival": is_new_fish_arrival,
            "model1_score": model1_score,
            "model2_score": model2_score,
            "decision_source": "two_models",
        }

    def process_station(self, station: str, db_root: Path) -> bool:
        """Process all events in a station database."""
        db_path = db_root / f"{station}_events.db"

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
                AND stage3_status = 'completed'
                AND detections_csv_path IS NOT NULL
            """)

            events = cursor.fetchall()
            LOGGER.info(f"Found {len(events)} events to reassess")

            updated = 0
            for idx, event in enumerate(events, 1):
                event_id = event["event_id"]
                event_dict = dict(event)
                result = self.classify_event(event_dict)

                try:
                    conn.execute("""
                        UPDATE events
                        SET is_actual_arrival = ?,
                            is_new_fish_arrival = ?,
                            stage4_model_score = ?,
                            stage4_decision_source = ?,
                            updated_at = CURRENT_TIMESTAMP
                        WHERE event_id = ?
                    """, (
                        result["is_actual_arrival"],
                        result["is_new_fish_arrival"],
                        result["model2_score"],  # Store fish model score
                        result["decision_source"],
                        event_id,
                    ))
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

    def run(self, db_root: Path) -> bool:
        """Run two-model classification on all stations."""
        if self.model1 is None or self.model2 is None:
            LOGGER.error("One or both models failed to load")
            return False

        LOGGER.info(f"Starting two-model reassessment for: {self.stations}")
        success_count = sum(1 for station in self.stations if self.process_station(station, db_root))
        LOGGER.info(f"Complete: {success_count}/{len(self.stations)} stations processed")
        return success_count == len(self.stations)


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Two-model Stage4 batch reassessment")
    parser.add_argument("--events-db-root", type=Path, default=Path("data/events_db"))
    parser.add_argument("--model1-path", type=Path, default=Path("models/stage4/model1_bird_arrival.json"))
    parser.add_argument("--model2-path", type=Path, default=Path("models/stage4/model2_fish_arrival.json"))
    parser.add_argument(
        "--stations",
        type=str,
        default="TRI3,TRI6,ROST2,ROST3,ROST4,ROST5,ROST6,FAR3,FAR6,BONDEN5,BONDEN6",
    )
    parser.add_argument("--log-dir", type=Path, default=Path("logs"))

    args = parser.parse_args()
    setup_logging(args.log_dir)

    stations = [s.strip() for s in args.stations.split(",") if s.strip()]
    classifier = TwoModelStage4(args.model1_path, args.model2_path, stations)
    success = classifier.run(args.events_db_root)
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
