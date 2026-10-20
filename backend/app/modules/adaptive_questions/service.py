"""Team-facing async services over the portable, stateful Version 10 core."""

import asyncio
from typing import Protocol, Sequence

from backend.app.shared.contracts import (
    AssessmentEvent,
    CubeAssessmentResult,
    QuestionRecommendation,
)

from .contracts.team_adapter import TeamContractAdapter
from .core_service import Component2Service
from .schemas import StatefulRecommendationResponse


class QuestionRecommender(Protocol):
    """Port implemented by a validated recommendation adapter."""

    async def recommend(
        self,
        history: Sequence[tuple[AssessmentEvent, CubeAssessmentResult]],
        age_group: str,
        curriculum: str,
    ) -> QuestionRecommendation:
        """Recommend a constrained next task from chronological history."""


class AdaptiveQuestionService:
    """Legacy stateless protocol retained for existing consumers.

    This protocol cannot start a cold session because it has no learner/session
    identifiers. New orchestration must use StatefulAdaptiveQuestionService.
    """
    def __init__(self, recommender: QuestionRecommender) -> None:
        self._recommender = recommender

    async def recommend_next(
        self,
        history: Sequence[tuple[AssessmentEvent, CubeAssessmentResult]],
        age_group: str,
        curriculum: str,
    ) -> QuestionRecommendation:
        return await self._recommender.recommend(history, age_group, curriculum)


class StatefulAdaptiveQuestionService:
    """Async boundary that preserves C2 sessions, idempotency and stopping rules."""

    def __init__(self, core: Component2Service | None = None) -> None:
        self.core = core or Component2Service()

    async def register_learner(self, pseudonymous_child_id: str, grade: int) -> dict:
        return await asyncio.to_thread(
            self.core.register_learner,
            pseudonymous_child_id,
            pseudonymous_child_id,
            grade,
        )

    async def start_or_resume(self, pseudonymous_child_id: str) -> dict:
        return await asyncio.to_thread(self.core.start_or_resume, pseudonymous_child_id)

    async def recommend_for_session(self, session_id: str) -> StatefulRecommendationResponse:
        task = await asyncio.to_thread(self.core.next_task, session_id)
        recommendation = task["recommendation"]
        status = recommendation["assessment_status"]
        if task.get("robot_task") is None:
            shared = None
            provenance = recommendation.get("decision_trace", {})
        else:
            shared = TeamContractAdapter.recommendation(task)
            provenance = {
                "recommendation_mode": recommendation.get("recommendation_mode"),
                "model_version": recommendation.get("model_version"),
                "research_model_version": recommendation.get("research_model_version"),
                "model_support_status": recommendation.get("model_support_status"),
                "abstention_reason": recommendation.get("abstention_reason"),
                "decision_trace": recommendation.get("decision_trace"),
            }
        return StatefulRecommendationResponse(
            recommendation_id=task.get("recommendation_id"),
            session_id=session_id,
            task_id=task.get("question_id"),
            recommendation=shared,
            robot_task=task.get("robot_task"),
            assessment_status=status,
            provenance=provenance,
        )

    async def ingest_validated_response(
        self,
        event: AssessmentEvent,
        result: CubeAssessmentResult,
    ) -> dict:
        return await asyncio.to_thread(TeamContractAdapter.ingest, self.core, event, result)

    async def export_mathematical_evidence(
        self,
        pseudonymous_child_id: str,
        session_id: str | None = None,
    ) -> dict:
        return await asyncio.to_thread(self.core.export_c4, pseudonymous_child_id, session_id)
