from __future__ import annotations

from .progression_rules import tier_b_evidence, transition_reason


def select_fallback(eligible: list[dict], state) -> tuple[dict | None, str]:
    if not eligible:
        return None, "HUMAN_SELECTION_REQUIRED"
    same_skill_b = [c for c in eligible if c["project_skill"] == state.current_skill_context and c["tier"] == "Tier B"]
    if tier_b_evidence(state, state.current_skill_context or "") and same_skill_b:
        nonrepeat_b = [c for c in same_skill_b if c["activity_id"] != state.last_activity]
        choices = nonrepeat_b or same_skill_b
        return sorted(choices, key=lambda c: c["activity_id"])[0], "SAME_SKILL_TIER_B_PROGRESS"
    same_skill_a = [c for c in eligible if c["project_skill"] == state.current_skill_context and c["tier"] == "Tier A"]
    nonrepeat = [c for c in same_skill_a if c["activity_id"] != state.last_activity]
    if transition_reason(state, state.current_skill_context or "") == "REINFORCE_FOUNDATION" and same_skill_a:
        choices = nonrepeat or same_skill_a
        return sorted(choices, key=lambda c: c["activity_id"])[0], "SAME_SKILL_TIER_A_REINFORCEMENT"
    if nonrepeat:
        return sorted(nonrepeat, key=lambda c: c["activity_id"])[0], "SAME_SKILL_TIER_A"
    if same_skill_a:
        return sorted(same_skill_a, key=lambda c: c['activity_id'])[0], 'SAME_SKILL_TIER_A'
    previous = [c for c in eligible if c["activity_id"] in state.successfully_attempted_activities and c["activity_id"] != state.last_activity]
    if previous:
        return sorted(previous, key=lambda c: c["activity_id"])[0], "PRIOR_SUCCESS_FOUNDATION"
    prereq = [c for c in eligible if c["tier"] == "Tier A" and any(x["prerequisite_skill"] == c["project_skill"] for x in eligible)]
    if prereq:
        return sorted(prereq, key=lambda c: c["activity_id"])[0], "PREREQUISITE_ACTIVITY"
    neutral = [c for c in eligible if c["tier"] == "Tier A" and c["project_skill"] in {"Counting / subitizing", "Quantity comparison"}]
    if neutral:
        return sorted(neutral, key=lambda c: c["activity_id"])[0], "GRADE_APPROPRIATE_BASELINE"
    tier_a = [c for c in eligible if c["tier"] == "Tier A"]
    if tier_a:
        return sorted(tier_a, key=lambda c: c["activity_id"])[0], "CURRICULUM_SAFE_TIER_A"
    return None, "HUMAN_SELECTION_REQUIRED"
