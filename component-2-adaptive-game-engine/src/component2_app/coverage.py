from __future__ import annotations

GAME_ORDER = ("COUNTING", "COMPARISON", "ORDERING", "PLACE_VALUE", "ADDITION", "SUBTRACTION", "MULTIPLICATION", "DIVISION")
GAME_TO_AREA = {"COUNTING":"Number sense","COMPARISON":"Number sense","ORDERING":"Number sense","PLACE_VALUE":"Number relationships","ADDITION":"Arithmetic calculation","SUBTRACTION":"Arithmetic calculation","MULTIPLICATION":"Multiplicative reasoning","DIVISION":"Multiplicative reasoning"}
AREA_IDS = {'Number sense': 'A1', 'Number relationships': 'A2',
            'Arithmetic calculation': 'A3', 'Multiplicative reasoning': 'A4'}
GAME_TO_SKILL = {"COUNTING":"Counting / subitizing","COMPARISON":"Quantity comparison","ORDERING":"Ordering / sequencing","PLACE_VALUE":"Place value / building","ADDITION":"Addition","SUBTRACTION":"Subtraction","MULTIPLICATION":"Multiplication / grouping","DIVISION":"Division / sharing"}

from recommendation_engine.local_state_model import estimate_skill_success, load_adaptive_settings
from recommendation_engine.progression_rules import tier_b_evidence


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


def choose_adaptive_focus(profiles: list[dict], eligible_candidates: list[dict], recent: list[dict], state) -> dict:
    """Choose continue/advance/reinforce/switch without bypassing hard eligibility."""
    settings = load_adaptive_settings()
    games = [g for g in GAME_ORDER if any(c["game_type"] == g for c in eligible_candidates)]
    if not games:
        return {"game_type": None, "decision_type": "NO_ELIGIBLE_ACTIVITY", "reason_code": "NO_ELIGIBLE_CANDIDATE", "model_output": None}
    by_game = {p["game_type"]: p for p in profiles}
    total = sum(int(by_game.get(g, {}).get("valid_interactions", 0)) for g in games)
    if total >= int(settings["maximum_total_interactions"]):
        return {"game_type": None, "decision_type": "STOP", "reason_code": "MAXIMUM_INTERACTIONS_REACHED", "model_output": None}
    minimum = int(settings["minimum_skill_observations"])
    last_game = recent[-1]["game_type"] if recent and recent[-1]["game_type"] in games else None
    if last_game:
        skill = GAME_TO_SKILL[last_game]
        estimate = estimate_skill_success(state, skill, settings)
        attempts = int(by_game[last_game]["valid_interactions"])
        consecutive = _consecutive_game_count(recent, last_game)
        probability = estimate["success_probability"]
        recent_same = [e["correct"] for e in recent if e["game_type"] == last_game][-2:]
        if attempts < minimum:
            decision = "REINFORCE" if probability is not None and probability <= settings["reinforce_probability_threshold"] and recent_same == [0, 0] else "CONTINUE"
            return {"game_type": last_game, "decision_type": decision, "reason_code": "MINIMUM_EVIDENCE_REQUIRED", "model_output": estimate}
        tier_b_available = any(c["game_type"] == last_game and c["tier"] == "Tier B" for c in eligible_candidates)
        if (state.last_tier == "Tier A" and tier_b_available and tier_b_evidence(state, skill)
                and probability is not None and probability >= settings["advance_probability_threshold"]
                and consecutive < settings["maximum_consecutive_same_game"]):
            return {"game_type": last_game, "decision_type": "ADVANCE", "reason_code": "MODEL_SUPPORTED_TIER_ADVANCE", "model_output": estimate}
        if (probability is not None and probability <= settings["reinforce_probability_threshold"]
                and consecutive < settings["maximum_consecutive_same_game"]):
            return {"game_type": last_game, "decision_type": "REINFORCE", "reason_code": "MODEL_SUPPORTED_FOUNDATION_REINFORCEMENT", "model_output": estimate}
    under = [g for g in games if int(by_game.get(g, {}).get("valid_interactions", 0)) < minimum]
    if under:
        counts = {g: int(by_game.get(g, {}).get("valid_interactions", 0)) for g in under}
        minimum_count = min(counts.values())
        tied = [g for g in under if counts[g] == minimum_count]
        area_last = {area: -1 for area in set(GAME_TO_AREA.values())}
        for index, event in enumerate(recent):
            area_last[GAME_TO_AREA[event["game_type"]]] = index
        selected = min(tied, key=lambda g: (area_last[GAME_TO_AREA[g]], GAME_ORDER.index(g)))
        return {"game_type": selected, "decision_type": "SWITCH" if last_game else "START", "reason_code": "COVERAGE_UNDERASSESSED", "model_output": estimate_skill_success(state, GAME_TO_SKILL[last_game], settings) if last_game else None}
    return {"game_type": None, "decision_type": "COMPLETE", "reason_code": "MINIMUM_GRADE_APPLICABLE_COVERAGE_REACHED", "model_output": estimate_skill_success(state, GAME_TO_SKILL[last_game], settings) if last_game else None}
