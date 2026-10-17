"""Adaptive-question application use cases."""

from typing import Protocol, Sequence

from backend.app.shared.contracts import (
    AssessmentEvent,
    CubeAssessmentResult,
    QuestionRecommendation,
)


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
    def __init__(self, recommender: QuestionRecommender) -> None:
        self._recommender = recommender

    async def recommend_next(
        self,
        history: Sequence[tuple[AssessmentEvent, CubeAssessmentResult]],
        age_group: str,
        curriculum: str,
    ) -> QuestionRecommendation:
        return await self._recommender.recommend(history, age_group, curriculum)
