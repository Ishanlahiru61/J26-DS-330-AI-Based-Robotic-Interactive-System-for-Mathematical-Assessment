"""FastAPI routes for the stateful C2 application boundary."""

from functools import lru_cache

from fastapi import APIRouter, Depends

from .schemas import LearnerRegistration, StatefulRecommendationResponse, ValidatedResponseRequest
from .service import StatefulAdaptiveQuestionService

router = APIRouter(prefix="/adaptive-questions", tags=["adaptive questions"])


@lru_cache(maxsize=1)
def get_service() -> StatefulAdaptiveQuestionService:
    """Construct the SQLite-backed service lazily, not during app import."""

    return StatefulAdaptiveQuestionService()


@router.post("/learners")
async def register_learner(
    request: LearnerRegistration,
    service: StatefulAdaptiveQuestionService = Depends(get_service),
) -> dict:
    return await service.register_learner(request.pseudonymous_child_id, request.grade)


@router.post("/learners/{pseudonymous_child_id}/sessions")
async def start_or_resume_session(
    pseudonymous_child_id: str,
    service: StatefulAdaptiveQuestionService = Depends(get_service),
) -> dict:
    return await service.start_or_resume(pseudonymous_child_id)


@router.get("/sessions/{session_id}/next", response_model=StatefulRecommendationResponse)
async def recommend_next(
    session_id: str,
    service: StatefulAdaptiveQuestionService = Depends(get_service),
) -> StatefulRecommendationResponse:
    return await service.recommend_for_session(session_id)


@router.post("/validated-responses")
async def ingest_validated_response(
    request: ValidatedResponseRequest,
    service: StatefulAdaptiveQuestionService = Depends(get_service),
) -> dict:
    return await service.ingest_validated_response(request.event, request.result)


@router.get("/learners/{pseudonymous_child_id}/evidence")
async def export_mathematical_evidence(
    pseudonymous_child_id: str,
    session_id: str | None = None,
    service: StatefulAdaptiveQuestionService = Depends(get_service),
) -> dict:
    return await service.export_mathematical_evidence(pseudonymous_child_id, session_id)
