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

    def recommend_next_activity(self, state, timestamp: str | None = None, *,
                                eligible_candidates: list[dict] | None = None,
                                filtered_candidates: list[dict] | None = None,
                                adaptive_decision: dict | None = None) -> dict:
        eligible, filtered = (generate_candidates(self.catalog, state)
                              if eligible_candidates is None
                              else (list(eligible_candidates), list(filtered_candidates or [])))
        if adaptive_decision is not None:
            return self._adaptive_result(state, eligible, filtered, adaptive_decision, timestamp)
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

    def _adaptive_result(self, state, eligible: list[dict], filtered: list[dict], decision: dict,
                         timestamp: str | None) -> dict:
        action = decision["decision_type"]
        preferred_tier = "Tier B" if action == "ADVANCE" else "Tier A"
        choices = [c for c in eligible if c["tier"] == preferred_tier]
        if not choices:
            choices = [c for c in eligible if c["tier"] == "Tier A"] or eligible
        selected = sorted(choices, key=lambda c: c["activity_id"])[0] if choices else None
        model_output = decision.get("model_output") or {}
        probability = model_output.get("success_probability")
        model_used = probability is not None
        mode = "EXPERIMENTAL_MODEL_ASSISTED" if model_used else ("RULE_FALLBACK" if selected else "NO_ELIGIBLE_ACTIVITY")
        fallback_reason = None if model_used else (model_output.get("abstention_reason") or "LOCAL_MODEL_EVIDENCE_UNAVAILABLE")
        candidate_trace = [{
            "activity_id": c["activity_id"], "game_type": c["game_type"], "tier": c["tier"],
            "eligible": True,
            "state_success_probability": probability,
            "score_scope": "SAME_OPERATIONAL_SKILL" if probability is not None else None,
        } for c in eligible]
        return {
            "learner_id": state.learner_id,
            "recommendation_timestamp": timestamp or datetime.now(timezone.utc).isoformat(),
            "candidate_activity_id": selected["activity_id"] if selected else None,
            "project_skill": selected["project_skill"] if selected else None,
            "tier": selected["tier"] if selected else None,
            "predicted_probability": probability,
            "model_support_status": "experimental_local_state" if model_used else "abstained",
            "model_provenance": model_output if model_output else None,
            "curriculum_eligible": bool(selected),
            "recommendation_mode": mode,
            "decision_type": action,
            "reason_code": decision["reason_code"],
            "fallback_used": bool(selected and not model_used),
            "fallback_rule": action if selected and not model_used else None,
            "abstained": not model_used,
            "abstention_reason": fallback_reason,
            "previous_activity_id": state.last_activity,
            "history_count": state.prior_interaction_count,
            "cold_start_status": state.cold_start_status,
            "eligible_candidate_ids": [c["activity_id"] for c in eligible],
            "filtered_candidates": [{"activity_id": c["activity_id"], "reason": c["filter_reason"]} for c in filtered],
            "decision_trace": {
                "history_summary": {"prior_interaction_count": state.prior_interaction_count,
                                    "skill_exposures": state.prior_skill_exposures.get(state.current_skill_context or "", 0),
                                    "recent_skill_outcomes": list(state.recent_skill_outcomes)},
                "eligible_candidates": candidate_trace,
                "excluded_candidates": [{"activity_id": c["activity_id"], "reason": c["filter_reason"]} for c in filtered],
                "decision_type": action, "pathway": mode,
                "selected_activity_id": selected["activity_id"] if selected else None,
                "fallback_reason": fallback_reason,
            },
            **{**VERSIONS,
               "model_version": model_output.get("model_version") if model_used else None,
               "research_model_version": VERSIONS["model_version"],
               "recommendation_policy_version": "C2_ADAPTIVE_ASSESSMENT_v1"},
        }


def recommend_next_activity(learner_state, catalog_path: str | Path | None = None) -> dict:
    path = Path(catalog_path) if catalog_path else Path(__file__).resolve().parents[2] / "config" / "activity_catalog.csv"
    return RecommendationEngine(path).recommend_next_activity(learner_state)
