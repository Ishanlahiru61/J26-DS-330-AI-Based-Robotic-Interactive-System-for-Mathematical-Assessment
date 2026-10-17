"""Transparent online learner-state estimator for local PP1 activities.

This is not the frozen ASSISTments XGBoost model. It is an experimental
Beta-Bernoulli model updated from one learner's strictly prior valid responses.
"""
from __future__ import annotations

import json
import math
from pathlib import Path


SETTINGS_PATH = Path(__file__).resolve().parents[2] / "config" / "adaptive_assessment_settings.json"


def load_adaptive_settings() -> dict:
    settings = json.loads(SETTINGS_PATH.read_text(encoding="utf-8"))
    required = {
        "policy_version", "local_state_model_version", "minimum_skill_observations",
        "minimum_model_observations", "beta_prior_alpha", "beta_prior_beta",
        "advance_probability_threshold", "reinforce_probability_threshold",
        "maximum_consecutive_same_game", "maximum_total_interactions",
    }
    if not required.issubset(settings):
        raise ValueError("adaptive assessment settings are incomplete")
    if not 0 < settings["reinforce_probability_threshold"] < settings["advance_probability_threshold"] < 1:
        raise ValueError("adaptive probability thresholds are invalid")
    return settings


def estimate_skill_success(state, skill: str, settings: dict | None = None) -> dict:
    """Return a strict-prior posterior estimate or explicit abstention."""
    settings = settings or load_adaptive_settings()
    attempts = int(state.prior_skill_exposures.get(skill, 0))
    correct = int(state.prior_skill_correct.get(skill, 0))
    if attempts < 0 or correct < 0 or correct > attempts:
        raise ValueError("invalid learner-state counts")
    result = {
        "model_version": settings["local_state_model_version"],
        "target": "next local response correctness for the same operational skill",
        "skill": skill,
        "evidence_count": attempts,
        "correct_count": correct,
        "uses_strict_prior_history": True,
        "validation_status": "EXPERIMENTAL_UNVALIDATED_FOR_REAL_CHILDREN",
        "success_probability": None,
        "posterior_standard_deviation": None,
        "abstention_reason": None,
    }
    if attempts < int(settings["minimum_model_observations"]):
        result["abstention_reason"] = "INSUFFICIENT_LOCAL_SKILL_EVIDENCE"
        return result
    alpha = float(settings["beta_prior_alpha"]) + correct
    beta = float(settings["beta_prior_beta"]) + attempts - correct
    probability = alpha / (alpha + beta)
    variance = alpha * beta / ((alpha + beta) ** 2 * (alpha + beta + 1))
    if not math.isfinite(probability) or not 0 <= probability <= 1:
        raise ValueError("invalid posterior estimate")
    result["success_probability"] = probability
    result["posterior_standard_deviation"] = math.sqrt(variance)
    return result
