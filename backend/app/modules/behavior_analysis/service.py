"""Behaviour-analysis application use cases."""

from typing import Protocol

from backend.app.shared.contracts import BehaviorObservation


class BehaviorAnalysisEngine(Protocol):
    """Port implemented by the edge-local tracking pipeline."""

    async def observe(self, task_id: str) -> BehaviorObservation:
        """Return a derived observation without uploading a raw stream."""


class BehaviorAnalysisService:
    def __init__(self, engine: BehaviorAnalysisEngine) -> None:
        self._engine = engine

    async def observe_task(self, task_id: str) -> BehaviorObservation:
        return await self._engine.observe(task_id)
