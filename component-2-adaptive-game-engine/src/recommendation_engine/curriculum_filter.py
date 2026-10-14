from __future__ import annotations

from .progression_rules import tier_b_evidence


def curriculum_eligible(candidate: dict, state) -> tuple[bool, str]:
    if not 1 <= state.current_grade <= 5:
        return False, "GRADE_OUT_OF_SCOPE"
    if not candidate["active"]:
        return False, "INACTIVE"
    if not candidate["curriculum_trace_id"].strip():
        return False, "MISSING_CURRICULUM_TRACE"
    if not candidate["grade_min"] <= state.current_grade <= candidate["grade_max"]:
        return False, "GRADE_NOT_SUPPORTED"
    if candidate["robot_suitability"].lower() not in {"yes", "high", "very good", "excellent"}:
        return False, "ROBOT_UNSUITABLE"
    if candidate["physical_cube_suitability"].lower() not in {"yes", "high", "very good", "excellent"}:
        return False, "CUBE_UNSUITABLE"
    if candidate["tier"] == "Tier B" and not tier_b_evidence(state, candidate["project_skill"]):
        return False, "TIER_B_EVIDENCE_INSUFFICIENT"
    prereq = candidate["prerequisite_skill"].strip()
    required = candidate["prerequisite_level"].strip().lower()
    if prereq and required == "exposure>=2" and state.prior_skill_exposures.get(prereq, 0) < 2:
        return False, "PREREQUISITE_REQUIRED"
    return True, "ELIGIBLE"


def generate_candidates(catalog: list[dict], state) -> tuple[list[dict], list[dict]]:
    eligible, filtered = [], []
    for candidate in catalog:
        ok, reason = curriculum_eligible(candidate, state)
        (eligible if ok else filtered).append(candidate if ok else {**candidate, "filter_reason": reason})
    return eligible, filtered
