from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any


@dataclass(frozen=True)
class LearnerState:
    learner_id: str
    current_grade: int
    allowed_grade_band: tuple[int, int]
    prior_interaction_count: int = 0
    prior_correct_count: int = 0
    current_skill_context: str | None = None
    recent_outcomes: tuple[int, ...] = ()
    recent_skill_outcomes: tuple[int, ...] = ()
    recent_skill_outcomes_by_skill: dict[str, tuple[int, ...]] = field(default_factory=dict)
    prior_skill_exposures: dict[str, int] = field(default_factory=dict)
    prior_skill_correct: dict[str, int] = field(default_factory=dict)
    tier_a_exposures: dict[str, int] = field(default_factory=dict)
    tier_a_correct: dict[str, int] = field(default_factory=dict)
    recent_streak: int = 0
    history_availability: bool = False
    skill_history_availability: bool = False
    cold_start_status: str = "first"
    last_activity: str | None = None
    last_tier: str | None = None
    successfully_attempted_activities: tuple[str, ...] = ()
    timestamp_valid: bool = True
    validation_warnings: tuple[str, ...] = ()
    ignored_event_count: int = 0
    interactions_since_last_correct: int | None = None

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["allowed_grade_band"] = list(self.allowed_grade_band)
        return value


def cold_start_bucket(prior_count: int) -> str:
    if prior_count <= 0:
        return "first"
    if prior_count <= 4:
        return "2-4"
    if prior_count <= 10:
        return "5-10"
    return "11+"


def make_learner_state(**kwargs: Any) -> LearnerState:
    """Construct a leakage-safe pre-event state; no current-event outcome is accepted."""
    grade = int(kwargs["current_grade"])
    prior = int(kwargs.get("prior_interaction_count", 0))
    kwargs.setdefault("allowed_grade_band", (grade, grade))
    kwargs.setdefault("cold_start_status", cold_start_bucket(prior))
    kwargs.setdefault("history_availability", prior > 0)
    kwargs.setdefault("skill_history_availability", bool(kwargs.get("prior_skill_exposures")))
    for key in ("recent_outcomes", "recent_skill_outcomes", "successfully_attempted_activities", "validation_warnings"):
        if key in kwargs:
            kwargs[key] = tuple(kwargs[key])
    if "recent_skill_outcomes_by_skill" in kwargs:
        kwargs["recent_skill_outcomes_by_skill"] = {
            str(skill): tuple(outcomes)
            for skill, outcomes in kwargs["recent_skill_outcomes_by_skill"].items()
        }
    return LearnerState(**kwargs)
