"""C2 API schemas; core-engine dictionaries remain framework independent."""

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from backend.app.shared.contracts import AssessmentEvent, CubeAssessmentResult, QuestionRecommendation


class C2ApiModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class LearnerRegistration(C2ApiModel):
    pseudonymous_child_id: str = Field(min_length=1)
    grade: int = Field(ge=1, le=5)


class ValidatedResponseRequest(C2ApiModel):
    event: AssessmentEvent
    result: CubeAssessmentResult


class RobotTaskSpecification(C2ApiModel):
    question_id: str
    game_type: str
    tier: str
    prompt: str
    numbered_cubes: list[int]
    robot_action: str
    metadata: dict[str, Any]
    voice_instruction_text: str
    placement_zones: list[str]
    hardware_profile_version: str


class StatefulRecommendationResponse(C2ApiModel):
    recommendation_id: str | None
    session_id: str
    task_id: str | None
    recommendation: QuestionRecommendation | None
    robot_task: RobotTaskSpecification | None
    assessment_status: dict[str, Any]
    provenance: dict[str, Any]
