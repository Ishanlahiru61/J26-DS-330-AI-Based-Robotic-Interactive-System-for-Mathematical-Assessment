from __future__ import annotations
from datetime import datetime, timezone
from pathlib import Path
from .abstention import abstention_decision
from .candidate_catalog import load_catalog
from .curriculum_filter import generate_candidates
from .fallback import select_fallback
from .feature_adapter import candidate_model_support
from .progression_rules import transition_reason

VERSIONS = {
    "model_version": "H2_GB_RT_OFF_v1",
    "preprocessing_version": "step3_h2_preprocessor_v1",
    "calibration_version": "H2_GB_RT_OFF_platt_v1",
    "curriculum_version": "NIE_G1-G5_2026-09-28",
    "activity_catalog_version": "c2_eight_game_catalog_v2",
    "recommendation_policy_version": "C2_COVERAGE_PROGRESS_v2",
}

class RecommendationEngine:
    def __init__(self, catalog_path: str | Path):
        self.catalog_path = Path(catalog_path)
        self.catalog = load_catalog(self.catalog_path)

    def recommend_next_activity(self, state, timestamp: str | None = None) -> dict:
        eligible, filtered = generate_candidates(self.catalog, state)
        support = {c["activity_id"]: candidate_model_support(c, state)[0] for c in eligible}
        abstained, reason = abstention_decision(state, eligible, support)
        selected = None
        fallback_rule = None
        predicted = None
        mode = "MODEL"
        reason_code = transition_reason(state, state.current_skill_context or "")
        if not abstained:
            # Local activities have no approved ASSISTments feature adapter/ranker yet.
            # A support flag alone must never produce a model-mode response with no selection.
            abstained = True
            reason = "LEARNED_RANKING_UNAVAILABLE"
        if abstained:
            selected, fallback_rule = select_fallback(eligible, state)
            mode = "RULE_BASED_FALLBACK" if selected else "HUMAN_REVIEW_REQUIRED"
            if fallback_rule == "SAME_SKILL_TIER_B_PROGRESS":
                reason_code = "PROGRESS_WITHIN_SKILL"
            elif fallback_rule == "SAME_SKILL_TIER_A_REINFORCEMENT":
                reason_code = "REINFORCE_FOUNDATION"
            else:
                reason_code = {
                    "COLD_START_INSUFFICIENT_HISTORY": "COLD_START_FALLBACK",
                    "SKILL_HISTORY_MISSING": "SKILL_HISTORY_MISSING",
                    "NO_ELIGIBLE_MODEL_CANDIDATE": "MODEL_UNSUPPORTED",
                    "NO_ELIGIBLE_CANDIDATE": "NO_ELIGIBLE_MODEL_CANDIDATE",
                    "TIMESTAMP_HISTORY_ANOMALY": "MODEL_UNSUPPORTED",
                    "LEARNED_RANKING_UNAVAILABLE": "MODEL_RANKING_UNAVAILABLE",
                }.get(reason or "", "MODEL_UNSUPPORTED")
        selected_support = support.get(selected["activity_id"], "unknown") if selected else "unknown"
        return {
            "learner_id": state.learner_id,
            "recommendation_timestamp": timestamp or datetime.now(timezone.utc).isoformat(),
            "candidate_activity_id": selected["activity_id"] if selected else None,
            "project_skill": selected["project_skill"] if selected else None,
            "tier": selected["tier"] if selected else None,
            "predicted_probability": predicted,
            "model_support_status": selected_support,
            "curriculum_eligible": bool(selected),
            "recommendation_mode": mode,
            "reason_code": reason_code,
            "fallback_used": bool(selected and abstained),
            "fallback_rule": fallback_rule,
            "abstained": abstained,
            "abstention_reason": reason,
            "previous_activity_id": state.last_activity,
            "history_count": state.prior_interaction_count,
            "cold_start_status": state.cold_start_status,
            "eligible_candidate_ids": [c["activity_id"] for c in eligible],
            "filtered_candidates": [{"activity_id": c["activity_id"], "reason": c["filter_reason"]} for c in filtered],
            **VERSIONS,
        }


def recommend_next_activity(learner_state, catalog_path: str | Path | None = None) -> dict:
    path = Path(catalog_path) if catalog_path else Path(__file__).resolve().parents[2] / "config" / "activity_catalog.csv"
    return RecommendationEngine(path).recommend_next_activity(learner_state)
