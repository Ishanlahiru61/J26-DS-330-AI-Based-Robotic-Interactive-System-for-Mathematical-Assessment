from __future__ import annotations

GAME_ORDER = ("COUNTING", "COMPARISON", "ORDERING", "PLACE_VALUE", "ADDITION", "SUBTRACTION", "MULTIPLICATION", "DIVISION")
GAME_TO_AREA = {"COUNTING":"Number sense","COMPARISON":"Number sense","ORDERING":"Number sense","PLACE_VALUE":"Number relationships","ADDITION":"Arithmetic calculation","SUBTRACTION":"Arithmetic calculation","MULTIPLICATION":"Multiplicative reasoning","DIVISION":"Multiplicative reasoning"}
AREA_IDS = {'Number sense': 'A1', 'Number relationships': 'A2',
            'Arithmetic calculation': 'A3', 'Multiplicative reasoning': 'A4'}
GAME_TO_SKILL = {"COUNTING":"Counting / subitizing","COMPARISON":"Quantity comparison","ORDERING":"Ordering / sequencing","PLACE_VALUE":"Place value / building","ADDITION":"Addition","SUBTRACTION":"Subtraction","MULTIPLICATION":"Multiplication / grouping","DIVISION":"Division / sharing"}

from ..prediction.local_state_model import estimate_skill_success, load_adaptive_settings
from .progression_rules import tier_b_evidence


def choose_game(profiles: list[dict], eligible_games: set[str], recent: list[dict]) -> tuple[str, str]:
    """Coverage before personalization; eligibility is never bypassed."""
    by_game = {p["game_type"]: p for p in profiles}
    candidates = [g for g in GAME_ORDER if g in eligible_games]
    if not candidates:
        raise ValueError("NO_ELIGIBLE_GAME")
    under = [g for g in candidates if by_game.get(g, {}).get("valid_interactions", 0) < 3]
    if under:
        min_count = min(by_game.get(g, {}).get("valid_interactions", 0) for g in under)
        tied = [g for g in under if by_game.get(g, {}).get("valid_interactions", 0) == min_count]
        area_last = {a: -1 for a in set(GAME_TO_AREA.values())}
        for i, event in enumerate(recent): area_last[GAME_TO_AREA[event["game_type"]]] = i
        chosen = min(tied, key=lambda g: (area_last[GAME_TO_AREA[g]], GAME_ORDER.index(g)))
        return chosen, "COVERAGE_UNDERASSESSED"
    error_counts = {g: 0 for g in candidates}
    for event in recent[-12:]:
        if event["game_type"] in error_counts and not event["correct"]: error_counts[event["game_type"]] += 1
    max_errors = max(error_counts.values())
    if max_errors >= 2:
        # Bound reinforcement to two consecutive questions so other areas remain sampled.
        repeated = recent[-2:]
        choices = [g for g in candidates if not (len(repeated) == 2 and all(e['game_type'] == g for e in repeated))]
        if choices:
            worst = max(error_counts[g] for g in choices)
            if worst >= 2:
                return min((g for g in choices if error_counts[g] == worst), key=GAME_ORDER.index), 'REPEATED_ERROR_REINFORCEMENT'
            return min(choices, key=lambda g: (by_game[g].get('last_assessed_at') or '', GAME_ORDER.index(g))), 'COVERAGE_REINFORCEMENT_BREAK'
    return min(candidates, key=lambda g: (by_game[g].get("last_assessed_at") or "", GAME_ORDER.index(g))), "LEAST_RECENTLY_ASSESSED"


def _consecutive_game_count(recent: list[dict], game: str) -> int:
    count = 0
    for event in reversed(recent):
        if event["game_type"] != game:
            break
        count += 1
    return count


def evidence_requirements(profiles: list[dict], games: list[str], settings: dict) -> dict:
    """Return transparent game- and area-level evidence state for the 16--24 policy."""
    by_game = {p["game_type"]: p for p in profiles}
    game_minimum = int(settings["minimum_game_observations"])
    inconsistent_minimum = int(settings["inconsistent_game_observations"])
    area_minimum = int(settings["minimum_area_observations"])
    game_rows = []
    for game in games:
        profile = by_game.get(game, {})
        attempts = int(profile.get("valid_interactions", 0))
        correct = int(profile.get("correct_interactions", 0))
        mixed = attempts >= 2 and 0 < correct < attempts
        required = inconsistent_minimum if mixed else game_minimum
        game_rows.append({
            "game_type": game,
            "area": GAME_TO_AREA[game],
            "valid_interactions": attempts,
            "correct_interactions": correct,
            "mixed_outcomes": mixed,
            "minimum_required": required,
            "evidence_status": "SUFFICIENT" if attempts >= required else "INSUFFICIENT",
        })
    areas = []
    for area in dict.fromkeys(GAME_TO_AREA[g] for g in games):
        included = [row for row in game_rows if row["area"] == area]
        attempts = sum(row["valid_interactions"] for row in included)
        areas.append({
            "area": area,
            "area_id": AREA_IDS[area],
            "valid_interactions": attempts,
            "minimum_required": area_minimum,
            "evidence_status": "SUFFICIENT" if attempts >= area_minimum else "INSUFFICIENT",
            "applicable_games": [row["game_type"] for row in included],
            "unobserved_games": [row["game_type"] for row in included if row["valid_interactions"] == 0],
        })
    return {"games": game_rows, "areas": areas}


def choose_adaptive_focus(profiles: list[dict], eligible_candidates: list[dict], recent: list[dict], state,
                          predictions: dict[str, dict] | None = None) -> dict:
    """Choose a budget-aware focus without letting personalization bypass coverage."""
    settings = load_adaptive_settings()
    predictions = predictions or {}

    def prediction_for(game: str) -> dict:
        skill = GAME_TO_SKILL[game]
        return predictions.get(skill) or estimate_skill_success(state, skill, settings)
    games = [g for g in GAME_ORDER if any(c["game_type"] == g for c in eligible_candidates)]
    if not games:
        return {"game_type": None, "decision_type": "NO_ELIGIBLE_ACTIVITY", "reason_code": "NO_ELIGIBLE_CANDIDATE", "model_output": None}
    by_game = {p["game_type"]: p for p in profiles}
    total = sum(int(by_game.get(g, {}).get("valid_interactions", 0)) for g in games)
    maximum = int(settings["maximum_total_interactions"])
    minimum_total = int(settings["minimum_total_valid_responses"])
    if total >= maximum:
        return {"game_type": None, "decision_type": "STOP", "reason_code": "MAXIMUM_INTERACTIONS_REACHED", "model_output": None}
    evidence = evidence_requirements(profiles, games, settings)
    game_evidence = {row["game_type"]: row for row in evidence["games"]}
    last_game = recent[-1]["game_type"] if recent and recent[-1]["game_type"] in games else None
    if "COUNTING" in games and total < int(settings["maximum_consecutive_same_game"]):
        estimate = prediction_for("COUNTING")
        probability = estimate.get("success_probability")
        action = "START" if total == 0 else "CONTINUE"
        reason = "FOUNDATIONAL_GRADE_APPROPRIATE_START" if total == 0 else "FOUNDATIONAL_EVIDENCE_BLOCK"
        tier_b_available = any(c["game_type"] == "COUNTING" and c["tier"] == "Tier B" for c in eligible_candidates)
        if (tier_b_available and tier_b_evidence(state, GAME_TO_SKILL["COUNTING"])
                and probability is not None and probability >= settings["advance_probability_threshold"]):
            action, reason = "ADVANCE", "MODEL_SUPPORTED_TIER_ADVANCE"
        elif total >= 2 and probability is not None and probability <= settings["reinforce_probability_threshold"]:
            action, reason = "REINFORCE", "MODEL_SUPPORTED_FOUNDATION_REINFORCEMENT"
        return {"game_type": "COUNTING", "decision_type": action,
                "reason_code": reason, "model_output": estimate}
    under = [g for g in games if game_evidence[g]["evidence_status"] == "INSUFFICIENT"]
    if under:
        counts = {g: int(by_game.get(g, {}).get("valid_interactions", 0)) for g in under}
        minimum_count = min(counts.values())
        tied = [g for g in under if counts[g] == minimum_count]
        area_last = {area: -1 for area in set(GAME_TO_AREA.values())}
        for index, event in enumerate(recent):
            area_last[GAME_TO_AREA[event["game_type"]]] = index
        # Coverage defines the admissible tied set.  Within it, valid model
        # evidence chooses the activity nearest the explicit 0.65 challenge
        # target; deterministic curriculum ordering breaks exact ties.
        valid = {g: prediction_for(g) for g in tied}
        scored = [g for g in tied if valid[g].get("success_probability") is not None]
        if scored:
            selected = min(scored, key=lambda g: (
                abs(float(valid[g]["success_probability"]) - 0.65),
                area_last[GAME_TO_AREA[g]], GAME_ORDER.index(g)))
            reason = "MODEL_RANKED_COVERAGE_CANDIDATE"
        else:
            selected = min(tied, key=lambda g: (area_last[GAME_TO_AREA[g]], GAME_ORDER.index(g)))
            reason = "COVERAGE_UNDERASSESSED"
        output = dict(valid[selected])
        output["candidate_skill_predictions"] = {
            GAME_TO_SKILL[g]: {
                "success_probability": valid[g].get("success_probability"),
                "source": valid[g].get("source", "LOCAL_BETA_BERNOULLI"),
                "valid": valid[g].get("valid", valid[g].get("success_probability") is not None),
            } for g in tied
        }
        decision = "START" if not last_game else ("CONTINUE" if selected == last_game else "SWITCH")
        probability = output.get("success_probability")
        tier_b_available = any(c["game_type"] == selected and c["tier"] == "Tier B" for c in eligible_candidates)
        if (tier_b_available and tier_b_evidence(state, GAME_TO_SKILL[selected])
                and probability is not None and probability >= settings["advance_probability_threshold"]):
            decision = "ADVANCE"
        return {"game_type": selected, "decision_type": decision,
                "reason_code": reason, "model_output": output}

    under_areas = [row for row in evidence["areas"] if row["evidence_status"] == "INSUFFICIENT"]
    if under_areas:
        target_areas = {row["area"] for row in under_areas}
        choices = [g for g in games if GAME_TO_AREA[g] in target_areas]
    elif total < minimum_total:
        minimum_count = min(game_evidence[g]["valid_interactions"] for g in games)
        choices = [g for g in games if game_evidence[g]["valid_interactions"] == minimum_count]
    else:
        return {"game_type": None, "decision_type": "COMPLETE", "reason_code": "MINIMUM_TOTAL_AND_REQUIRED_EVIDENCE_REACHED", "model_output": prediction_for(last_game) if last_game else None}

    # After mandatory coverage, the model chooses among equally admissible games.
    # This preserves genuine model influence without allowing high probability to
    # skip a required game or area.
    estimates = {g: prediction_for(g) for g in choices}
    scored = [g for g in choices if estimates[g].get("success_probability") is not None]
    selected = min(scored, key=lambda g: (abs(float(estimates[g]["success_probability"]) - 0.65), GAME_ORDER.index(g))) if scored else min(choices, key=GAME_ORDER.index)
    estimate = estimates[selected]
    probability = estimate.get("success_probability")
    action = "SWITCH" if selected != last_game else "CONTINUE"
    tier_b_available = any(c["game_type"] == selected and c["tier"] == "Tier B" for c in eligible_candidates)
    if (selected == last_game and state.last_tier == "Tier A" and tier_b_available
            and tier_b_evidence(state, GAME_TO_SKILL[selected]) and probability is not None
            and probability >= settings["advance_probability_threshold"]):
        action = "ADVANCE"
    elif selected == last_game and probability is not None and probability <= settings["reinforce_probability_threshold"]:
        action = "REINFORCE"
    return {"game_type": selected, "decision_type": action,
            "reason_code": "MINIMUM_TOTAL_BUDGET_ALLOCATION" if not under_areas else "AREA_EVIDENCE_REQUIRED",
            "model_output": estimate}
