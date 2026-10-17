"""Risk-profiling application use cases."""

from typing import Protocol, Sequence

from backend.app.shared.contracts import (
    AssessmentEvent,
    BehaviorObservation,
    CubeAssessmentResult,
    QuestionRecommendation,
    ScreeningProfile,
)


class ProfileEstimator(Protocol):
    """Port implemented only after thresholds and methods are validated."""

    async def estimate(
        self,
        event: AssessmentEvent,
        cube_result: CubeAssessmentResult,
        behavior: BehaviorObservation,
        recommendation: QuestionRecommendation,
        history: Sequence[AssessmentEvent],
    ) -> ScreeningProfile:
        """Build an explainable, non-diagnostic screening profile."""


class RiskProfilingService:
    def __init__(self, estimator: ProfileEstimator) -> None:
        self._estimator = estimator

    async def update_profile(
        self,
        event: AssessmentEvent,
        cube_result: CubeAssessmentResult,
        behavior: BehaviorObservation,
        recommendation: QuestionRecommendation,
        history: Sequence[AssessmentEvent],
    ) -> ScreeningProfile:
        return await self._estimator.estimate(
            event, cube_result, behavior, recommendation, history
        )
