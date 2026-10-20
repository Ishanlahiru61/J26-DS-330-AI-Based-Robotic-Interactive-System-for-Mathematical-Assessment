from __future__ import annotations

from .policy_config import load_policy_settings


def abstention_decision(state, eligible: list[dict], support: dict[str, str]) -> tuple[bool, str | None]:
    if not state.timestamp_valid:
        return True, "TIMESTAMP_HISTORY_ANOMALY"
    if not eligible:
        return True, "NO_ELIGIBLE_CANDIDATE"
    if state.prior_interaction_count > 0 and not state.skill_history_availability:
        return True, "SKILL_HISTORY_MISSING"
    minimum_history = int(load_policy_settings()["full_model_min_prior_interactions"])
    if state.prior_interaction_count < minimum_history:
        return True, "COLD_START_INSUFFICIENT_HISTORY"
    if not any(support.get(c["activity_id"]) == "supported" for c in eligible):
        return True, "NO_ELIGIBLE_MODEL_CANDIDATE"
    return False, None
