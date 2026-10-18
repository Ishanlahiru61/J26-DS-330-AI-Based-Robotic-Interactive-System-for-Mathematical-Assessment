"""Application-level coordination of a completed assessment task."""

from dataclasses import dataclass
from typing import Protocol, Sequence

from backend.app.modules.risk_profiling.service import RiskProfilingService
from backend.app.shared.contracts import (
    AssessmentEvent,
    BehaviorObservation,
    CubeAssessmentResult,
    QuestionRecommendation,
    ScreeningProfile,
)


@dataclass(frozen=True)
class CorrelationKey:
    """Identifiers used to join outputs from concurrent edge processing."""

    pseudonymous_child_id: str
    session_id: str
    task_id: str

    @classmethod
    def from_event(cls, event: AssessmentEvent) -> "CorrelationKey":
        return cls(
            pseudonymous_child_id=event.pseudonymous_child_id,
            session_id=event.session_id,
            task_id=event.task_id,
        )


@dataclass(frozen=True)
class CompletedTaskInputs:
    event: AssessmentEvent
    recommendation: QuestionRecommendation
    cube_result: CubeAssessmentResult
    behavior_observation: BehaviorObservation

    @property
    def correlation_key(self) -> CorrelationKey:
        return CorrelationKey.from_event(self.event)


@dataclass(frozen=True)
class ProfileUpdateRequested:
    """Internal event for durable or background profile processing."""

    inputs: CompletedTaskInputs
    history: Sequence[AssessmentEvent]


class ProfileUpdatePublisher(Protocol):
    """Port for an in-process background task or durable event adapter."""

    async def publish(self, event: ProfileUpdateRequested) -> None:
        """Schedule a profile update without coupling to queue technology."""


class AssessmentWorkflow:
    """Combines validated module outputs and requests a profile update."""

    def __init__(
        self,
        risk_service: RiskProfilingService,
        profile_update_publisher: ProfileUpdatePublisher,
    ) -> None:
        self._risk_service = risk_service
        self._profile_update_publisher = profile_update_publisher

    async def queue_profile_update(
        self,
        inputs: CompletedTaskInputs,
        history: Sequence[AssessmentEvent],
    ) -> CorrelationKey:
        """Publish long-running profile work and return its correlation key."""

        await self._profile_update_publisher.publish(
            ProfileUpdateRequested(inputs=inputs, history=history)
        )
        return inputs.correlation_key

    async def handle_profile_update(
        self,
        inputs: CompletedTaskInputs,
        history: Sequence[AssessmentEvent],
    ) -> ScreeningProfile:
        """Handle the published event in a background worker adapter."""

        return await self._risk_service.update_profile(
            event=inputs.event,
            cube_result=inputs.cube_result,
            behavior=inputs.behavior_observation,
            recommendation=inputs.recommendation,
            history=history,
        )
