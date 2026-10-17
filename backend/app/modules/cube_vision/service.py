"""Cube-vision application use cases."""

from typing import Protocol

from backend.app.shared.contracts import CubeAssessmentResult


class CubeAnalysisEngine(Protocol):
    """Port implemented by an edge-local cube analysis adapter."""

    async def analyse(self, task_id: str, expected_answer: float) -> CubeAssessmentResult:
        """Analyse the locally available task frame or stream."""


class CubeVisionService:
    def __init__(self, engine: CubeAnalysisEngine) -> None:
        self._engine = engine

    async def assess(
        self, task_id: str, expected_answer: float
    ) -> CubeAssessmentResult:
        return await self._engine.analyse(task_id, expected_answer)
