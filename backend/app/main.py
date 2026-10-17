"""FastAPI application entry point."""

from fastapi import FastAPI

from backend.app.modules.adaptive_questions.api import router as adaptive_router
from backend.app.modules.assessment.api import router as assessment_router
from backend.app.modules.behavior_analysis.api import router as behavior_router
from backend.app.modules.cube_vision.api import router as cube_vision_router
from backend.app.modules.risk_profiling.api import router as risk_router

app = FastAPI(
    title="J26-DS-330 Screening Support API",
    description="Educational screening support; not a clinical diagnosis.",
    version="0.1.0",
)

app.include_router(assessment_router)
app.include_router(cube_vision_router)
app.include_router(adaptive_router)
app.include_router(behavior_router)
app.include_router(risk_router)


@app.get("/health", tags=["operations"])
def health() -> dict[str, str]:
    """Report whether the API process is available."""

    return {"status": "ok"}
