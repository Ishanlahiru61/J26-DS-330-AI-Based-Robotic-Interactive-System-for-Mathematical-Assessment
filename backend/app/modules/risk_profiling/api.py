"""Risk-profiling HTTP route declarations."""

from fastapi import APIRouter

router = APIRouter(prefix="/risk-profiles", tags=["risk profiles"])
