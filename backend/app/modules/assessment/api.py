"""Assessment HTTP route declarations."""

from fastapi import APIRouter

router = APIRouter(prefix="/assessments", tags=["assessments"])
