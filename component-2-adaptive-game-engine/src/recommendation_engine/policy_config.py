from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path


DEFAULT_SETTINGS_PATH = Path(__file__).resolve().parents[2] / "config" / "recommendation_settings.json"
REQUIRED_SETTINGS = {
    "policy_version": str,
    "curriculum_grade_min": int,
    "curriculum_grade_max": int,
    "full_model_min_prior_interactions": int,
    "tier_b_min_exposures": int,
    "tier_b_min_accumulated_correct_rate": (int, float),
    "tier_b_recent_window": int,
    "tier_b_min_recent_correct": int,
    "repetition_cooldown_activities": int,
}


@lru_cache(maxsize=8)
def _load(path_text: str) -> dict:
    path = Path(path_text)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid recommendation settings: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError("Recommendation settings must be a JSON object")
    missing = sorted(set(REQUIRED_SETTINGS) - set(value))
    if missing:
        raise ValueError(f"Recommendation settings missing required fields: {missing}")
    for key, expected_type in REQUIRED_SETTINGS.items():
        if isinstance(value[key], bool) or not isinstance(value[key], expected_type):
            raise ValueError(f"Invalid recommendation setting type: {key}")
    if not 1 <= value["curriculum_grade_min"] <= value["curriculum_grade_max"]:
        raise ValueError("Invalid curriculum grade range")
    if value["full_model_min_prior_interactions"] < 1:
        raise ValueError("full_model_min_prior_interactions must be positive")
    if value["tier_b_min_exposures"] < 1 or value["tier_b_recent_window"] < 1:
        raise ValueError("Tier-B exposure and window thresholds must be positive")
    if not 0 <= float(value["tier_b_min_accumulated_correct_rate"]) <= 1:
        raise ValueError("tier_b_min_accumulated_correct_rate must be in [0,1]")
    if not 0 <= value["tier_b_min_recent_correct"] <= value["tier_b_recent_window"]:
        raise ValueError("tier_b_min_recent_correct must fit within tier_b_recent_window")
    return value


def load_policy_settings(path: str | Path | None = None) -> dict:
    """Return a validated copy of the authoritative Step-4 runtime policy."""
    resolved = Path(path or DEFAULT_SETTINGS_PATH).resolve()
    return dict(_load(str(resolved)))
