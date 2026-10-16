from __future__ import annotations

from .policy_config import load_policy_settings


def recent_outcomes_for_skill(state, skill: str) -> tuple[int, ...]:
    """Return only outcomes belonging to the candidate skill."""
    by_skill = getattr(state, "recent_skill_outcomes_by_skill", {}) or {}
    if skill in by_skill:
        return tuple(by_skill[skill])
    if skill == state.current_skill_context:
        return tuple(state.recent_skill_outcomes)
    return ()


def tier_b_evidence(state, skill: str) -> bool:
    """Conservative project heuristic: exposure plus repeated evidence, never one response."""
    settings = load_policy_settings()
    exposures = int(state.tier_a_exposures.get(skill, 0))
    correct = int(state.tier_a_correct.get(skill, 0))
    window = int(settings["tier_b_recent_window"])
    recent = recent_outcomes_for_skill(state, skill)[-window:]
    return (
        exposures >= int(settings["tier_b_min_exposures"])
        and correct / max(exposures, 1) >= float(settings["tier_b_min_accumulated_correct_rate"])
        and len(recent) == window
        and sum(recent) >= int(settings["tier_b_min_recent_correct"])
    )


def transition_reason(state, skill: str) -> str:
    settings = load_policy_settings()
    window = int(settings["tier_b_recent_window"])
    recent = recent_outcomes_for_skill(state, skill)[-window:]
    if len(recent) == window and sum(recent) < int(settings["tier_b_min_recent_correct"]):
        return "REINFORCE_FOUNDATION"
    if tier_b_evidence(state, skill):
        return "PROGRESS_WITHIN_SKILL"
    return "MAINTAIN_CURRENT_LEVEL"
