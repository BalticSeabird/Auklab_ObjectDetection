"""Stage 4 processor: post-classify arrivals using stage3 clip detections."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

from .config_manager import Config
from .event_database import EventDatabase
from .stage4_modeling import Stage4ModelArtifact, extract_stage4_features, parse_optional_int
from .state_manager import VideoJob
from .worker_pool import ProcessingMetrics, ProcessingResult, StageProcessor, WorkerContext

LOGGER = logging.getLogger(__name__)


class Stage4PostClassificationProcessor(StageProcessor):
    """Assign final arrival labels from stage3 clip-level detections."""

    def __init__(self, config: Config) -> None:
        self.config = config
        self.stage_cfg = config.processing.stage4_post_classification
        self.logger = logging.getLogger("stage4")
        self.model_artifact: Optional[Stage4ModelArtifact] = None
        self._load_model_if_configured()

    def _load_model_if_configured(self) -> None:
        if not self.stage_cfg.use_model:
            return
        model_path = Path(self.stage_cfg.model_path)
        if not model_path.exists():
            self.logger.warning("Stage4 model enabled but missing file: %s", model_path)
            return
        try:
            self.model_artifact = Stage4ModelArtifact.from_path(model_path)
            self.logger.info("Loaded Stage4 model artifact from %s", model_path)
        except Exception as exc:
            self.logger.exception("Failed to load Stage4 model artifact from %s: %s", model_path, exc)
            self.model_artifact = None

    def process(self, job: VideoJob, context: WorkerContext) -> Optional[ProcessingResult]:  # noqa: ARG002
        event_db = EventDatabase.for_station(self.config, job.station)
        event_db.initialize()
        events_df = event_db.fetch_events_for_video(station=job.station, video_id=job.video_id)

        if events_df.empty:
            self.logger.info("[%s] No events for stage4", job.video_id)
            return ProcessingResult(metadata={"classified_events": 0}, metrics=ProcessingMetrics(events_count=0))

        classified = 0
        for _, event in events_df.iterrows():
            event_id = str(event.get("event_id", ""))
            if not event_id:
                continue

            labels = self._classify_event(event)
            event_db.update_stage4_labels(
                station=job.station,
                event_id=event_id,
                is_actual_arrival=labels["is_actual_arrival"],
                is_new_fish_arrival=labels["is_new_fish_arrival"],
                fish_detections_stage4=labels["fish_detections_stage4"],
                fish_avg_confidence_stage4=labels["fish_avg_confidence_stage4"],
                rule_version=self.stage_cfg.rule_version,
                rule_hits=labels["rule_hits"],
                features=labels["features"],
                model_score=labels.get("model_score"),
                decision_source=str(labels.get("decision_source", "rules")),
            )
            classified += 1

        return ProcessingResult(
            metadata={"classified_events": classified, "rule_version": self.stage_cfg.rule_version},
            metrics=ProcessingMetrics(events_count=classified),
        )

    def _classify_event(self, event_row: pd.Series) -> Dict[str, object]:
        event_type = str(event_row.get("event_type", ""))
        detections_csv = event_row.get("detections_csv_path")
        stage3_status = str(event_row.get("stage3_status") or "")

        # Stage4 semantics currently focus on arrival refinement only.
        # For other event types, avoid treating missing clip detections as an error.
        if event_type != "arrival":
            return {
                "is_actual_arrival": 0,
                "is_new_fish_arrival": 0,
                "fish_detections_stage4": 0,
                "fish_avg_confidence_stage4": 0.0,
                "rule_hits": ["non_arrival_event"],
                "features": {"event_type": event_type, "stage3_status": stage3_status},
                "model_score": None,
                "decision_source": "rules",
            }

        if stage3_status != "completed":
            return {
                "is_actual_arrival": 0,
                "is_new_fish_arrival": 0,
                "fish_detections_stage4": 0,
                "fish_avg_confidence_stage4": 0.0,
                "rule_hits": ["stage3_not_completed"],
                "features": {"event_type": event_type, "stage3_status": stage3_status},
                "model_score": None,
                "decision_source": "rules",
            }

        if not detections_csv or not Path(str(detections_csv)).exists():
            return {
                "is_actual_arrival": 0,
                "is_new_fish_arrival": 0,
                "fish_detections_stage4": 0,
                "fish_avg_confidence_stage4": 0.0,
                "rule_hits": ["missing_detections_csv"],
                "features": {"event_type": event_type, "stage3_status": stage3_status},
                "model_score": None,
                "decision_source": "rules",
            }

        detections = pd.read_csv(str(detections_csv))
        stage2_flag = parse_optional_int(event_row.get("arrival_with_fish_stage2"), default=0)
        features = extract_stage4_features(detections, stage2_flag=stage2_flag)
        fish_detections_stage4 = int(features.get("fish_detection_count", 0))
        fish_avg_confidence_stage4 = float(features.get("fish_avg_confidence", 0.0))

        is_actual_arrival, arrival_hits = self._rule_actual_arrival(features)
        is_new_fish_arrival, fish_hits = self._rule_new_fish_arrival(features, event_row, is_actual_arrival)

        model_score: Optional[float] = None
        decision_source = "rules"
        if is_actual_arrival and self.model_artifact is not None:
            model_score = float(self.model_artifact.predict_proba(features))
            threshold = self.stage_cfg.model_threshold
            if model_score >= threshold:
                is_new_fish_arrival = True
                fish_hits.append("model_positive")
            else:
                is_new_fish_arrival = False
                fish_hits.append("model_negative")
            fish_hits.append(f"model_threshold_{threshold:.3f}")
            decision_source = "model"

        return {
            "is_actual_arrival": int(is_actual_arrival),
            "is_new_fish_arrival": int(is_new_fish_arrival),
            "fish_detections_stage4": fish_detections_stage4,
            "fish_avg_confidence_stage4": fish_avg_confidence_stage4,
            "rule_hits": arrival_hits + fish_hits,
            "features": features,
            "model_score": model_score,
            "decision_source": decision_source,
        }

    def _rule_actual_arrival(self, features: Dict[str, object]) -> Tuple[bool, List[str]]:
        hits: List[str] = []

        bird_frames = int(features.get("bird_frames", 0))
        displacement = float(features.get("bird_displacement", 0.0))
        mean_motion = float(features.get("bird_mean_motion", 0.0))

        if bird_frames >= self.stage_cfg.min_bird_frames:
            hits.append("bird_frames_ok")
        if displacement >= self.stage_cfg.min_centroid_displacement:
            hits.append("bird_displacement_ok")
        if mean_motion >= self.stage_cfg.min_mean_motion:
            hits.append("bird_motion_ok")

        is_actual = (
            bird_frames >= self.stage_cfg.min_bird_frames
            and (
                displacement >= self.stage_cfg.min_centroid_displacement
                or mean_motion >= self.stage_cfg.min_mean_motion
            )
        )
        if not is_actual:
            hits.append("actual_arrival_rejected")
        return is_actual, hits

    def _rule_new_fish_arrival(
        self,
        features: Dict[str, object],
        event_row: pd.Series,
        is_actual_arrival: bool,
    ) -> Tuple[bool, List[str]]:
        hits: List[str] = []
        if not is_actual_arrival:
            hits.append("not_actual_arrival")
            return False, hits

        fish_frames = int(features.get("fish_frames", 0))
        fish_first_ratio = float(features.get("fish_first_frame_ratio", 1.0))
        stage2_flag = parse_optional_int(event_row.get("arrival_with_fish_stage2"), default=0)

        if stage2_flag == 1:
            hits.append("stage2_arrival_with_fish")
        if fish_frames >= self.stage_cfg.min_fish_frames:
            hits.append("fish_frames_ok")
        if fish_first_ratio > self.stage_cfg.fish_early_presence_ratio:
            hits.append("fish_not_early")

        is_new_fish = (
            stage2_flag == 1
            and fish_frames >= self.stage_cfg.min_fish_frames
            and fish_first_ratio > self.stage_cfg.fish_early_presence_ratio
        )
        if not is_new_fish:
            hits.append("new_fish_arrival_rejected")
        return is_new_fish, hits
