"""Versioned Pydantic contracts exchanged between backend modules."""

from datetime import datetime
from enum import Enum

from pydantic import BaseModel, ConfigDict, Field


class ContractModel(BaseModel):
    """Strict base model for internal and API contracts."""

    model_config = ConfigDict(extra="forbid")


class ObservationStatus(str, Enum):
    VALID = "valid"
    UNCERTAIN = "uncertain"
    INSUFFICIENT_VISIBILITY = "insufficient_visibility"
    NO_HANDS_DETECTED = "no_hands_detected"


class BehaviorLabel(str, Enum):
    FINGER_COUNTING = "finger_counting"
    OTHER_HAND_MOVEMENT = "other_hand_movement"
    UNCERTAIN = "uncertain"


class ScreeningRiskLevel(str, Enum):
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"


class CubePosition(ContractModel):
    """A cube centre in an implementation-defined coordinate space."""

    x: float
    y: float
    z: float | None = None


class AssessmentEvent(ContractModel):
    event_id: str = Field(min_length=1)
    pseudonymous_child_id: str = Field(min_length=1)
    session_id: str = Field(min_length=1)
    task_id: str = Field(min_length=1)
    task_type: str = Field(min_length=1)
    difficulty_level: str = Field(min_length=1)
    started_at: datetime
    completed_at: datetime | None = None
    model_version: str = Field(min_length=1)


class CubeAssessmentResult(ContractModel):
    detected_cube_count: int = Field(ge=0)
    cube_colours: list[str]
    cube_positions: list[CubePosition]
    detected_answer: float | None
    expected_answer: float
    is_correct: bool
    error_value: float
    response_time_ms: int = Field(ge=0)
    attempt_count: int = Field(ge=1)
    hint_count: int = Field(ge=0)
    detection_confidence: float = Field(ge=0.0, le=1.0)
    validation_status: str = Field(min_length=1)


class QuestionRecommendation(ContractModel):
    recommended_task_type: str = Field(min_length=1)
    recommended_difficulty: str = Field(min_length=1)
    predicted_success_probability: float = Field(ge=0.0, le=1.0)
    recommendation_reason: str = Field(min_length=1)
    history_window_size: int = Field(ge=0)
    model_version: str = Field(min_length=1)


class BehaviorObservation(ContractModel):
    behavior_label: BehaviorLabel
    behavior_confidence: float = Field(ge=0.0, le=1.0)
    behavior_duration_ms: int = Field(ge=0)
    finger_count_estimate: int | None = Field(default=None, ge=0)
    hand_tracking_quality: float = Field(ge=0.0, le=1.0)
    head_yaw: float | None = None
    head_pitch: float | None = None
    coarse_gaze_direction: str | None = None
    attention_indicator: str | None = None
    observation_status: ObservationStatus
    model_version: str = Field(min_length=1)


class ScreeningProfile(ContractModel):
    topic_mastery_scores: dict[str, float]
    persistent_error_patterns: list[str]
    response_time_pattern: str
    finger_counting_pattern: str
    attention_pattern: str
    screening_risk_level: ScreeningRiskLevel
    screening_confidence: float = Field(ge=0.0, le=1.0)
    important_features: list[str]
    recommended_next_action: str
    model_version: str = Field(min_length=1)
    generated_at: datetime
