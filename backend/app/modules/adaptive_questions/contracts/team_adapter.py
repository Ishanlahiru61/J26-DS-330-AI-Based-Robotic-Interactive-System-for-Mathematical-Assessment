"""Explicit mappings between team contracts and the portable C2 core."""

from __future__ import annotations

from backend.app.shared.contracts import (
    AssessmentEvent,
    CubeAssessmentResult,
    QuestionRecommendation,
    RecommendationDecisionStatus,
)

from ..core_service import Component2Service
from ..infrastructure.errors import IntegrationError


def _normalise_difficulty(value: str) -> str:
    compact = value.strip().lower().replace("_", " ").replace("-", " ")
    return {"a": "tier a", "b": "tier b"}.get(compact, compact)


class TeamContractAdapter:
    """Keep team-specific Pydantic types outside the portable core engine."""

    @staticmethod
    def recommendation(task: dict) -> QuestionRecommendation:
        recommendation = task["recommendation"]
        probability = recommendation.get("predicted_probability")
        if probability is None:
            decision_status = RecommendationDecisionStatus.ABSTAINED
        elif str(recommendation.get("recommendation_mode", "")).startswith("TRAINED_ML"):
            decision_status = RecommendationDecisionStatus.MODEL_SCORED
        else:
            decision_status = RecommendationDecisionStatus.FALLBACK_SCORED
        return QuestionRecommendation(
            recommended_task_type=recommendation["game_type"],
            recommended_difficulty=recommendation["tier"],
            predicted_success_probability=probability,
            decision_status=decision_status,
            recommendation_reason=recommendation["reason_code"],
            history_window_size=recommendation["history_count"],
            model_version=recommendation.get("model_version"),
        )

    @staticmethod
    def ingest(
        core: Component2Service,
        event: AssessmentEvent,
        result: CubeAssessmentResult,
    ) -> dict:
        """Ingest one C1 result, scoring only a valid first-attempt response."""

        session = core.db.session(event.session_id)
        if session["learner_id"] != event.pseudonymous_child_id:
            raise IntegrationError("C1_CONTEXT_MISMATCH", "learner and session do not match")
        question = core.db.question(event.session_id, event.task_id)
        if question is None:
            raise IntegrationError("INVALID_QUESTION", "task_id is not an issued C2 question")
        if event.task_type.strip().upper() != question["game_type"]:
            raise IntegrationError("C1_CONTEXT_MISMATCH", "task_type does not match the issued C2 game")
        if _normalise_difficulty(event.difficulty_level) != _normalise_difficulty(question["tier"]):
            raise IntegrationError("C1_CONTEXT_MISMATCH", "difficulty does not match the issued C2 tier")
        if float(result.expected_answer) != float(question["expected_answer"]):
            raise IntegrationError("C1_CONTEXT_MISMATCH", "expected answer does not match the issued question")

        context = {
            "assessment_event": event.model_dump(mode="json"),
            "cube_result": result.model_dump(mode="json"),
            "learner_id": event.pseudonymous_child_id,
            "activity_id": question["activity_id"],
            "attempt_count": result.attempt_count,
            "response_time_ms": result.response_time_ms,
        }
        existing = core.db.interaction_payload(event.event_id)
        if existing:
            if (
                existing.get("session_id") == event.session_id
                and existing.get("question_id") == event.task_id
                and existing.get("c1_context") == context
            ):
                return {"status": "duplicate_ignored", "event_id": event.event_id}
            raise IntegrationError("DUPLICATE_EVENT", "event_id already exists with a different payload")
        if question["status"] != "ACTIVE":
            raise IntegrationError("INVALID_QUESTION", "task_id is not the active issued C2 question")
        status = result.validation_status.strip().lower()
        timestamp = (event.completed_at or event.started_at).isoformat()
        if status != "valid":
            non_scoring_status = {
                "uncertain": "UNCERTAIN",
                "insufficient_visibility": "INSUFFICIENT_VISIBILITY",
                "no_hands_detected": "NO_HANDS_DETECTED",
            }.get(status, "INVALID")
            return core.db.record_non_scoring(
                event.session_id,
                event.task_id,
                non_scoring_status,
                event.event_id,
                timestamp,
                context,
            )
        if event.completed_at is None:
            raise IntegrationError("INCOMPLETE_EVENT", "a valid scored response requires completed_at")
        if result.attempt_count != 1:
            return core.db.record_non_scoring(
                event.session_id,
                event.task_id,
                "RETRY_NOT_SCORED",
                event.event_id,
                timestamp,
                context,
            )
        if result.detected_answer is None:
            raise IntegrationError("INVALID_VALIDATED_RESULT", "valid C1 result requires detected_answer")
        return core.db.record_answer(
            event.session_id,
            event.task_id,
            None,
            event.event_id,
            result.response_time_ms,
            timestamp,
            validated_correctness=int(result.is_correct),
            c1_context=context,
        )
