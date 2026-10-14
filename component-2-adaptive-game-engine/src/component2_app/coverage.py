from __future__ import annotations

GAME_ORDER = ("COUNTING", "COMPARISON", "ORDERING", "PLACE_VALUE", "ADDITION", "SUBTRACTION", "MULTIPLICATION", "DIVISION")
GAME_TO_AREA = {"COUNTING":"Number sense","COMPARISON":"Number sense","ORDERING":"Number relationships","PLACE_VALUE":"Number relationships","ADDITION":"Arithmetic calculation","SUBTRACTION":"Arithmetic calculation","MULTIPLICATION":"Multiplicative reasoning","DIVISION":"Multiplicative reasoning"}
GAME_TO_SKILL = {"COUNTING":"Counting / subitizing","COMPARISON":"Quantity comparison","ORDERING":"Ordering / sequencing","PLACE_VALUE":"Place value / building","ADDITION":"Addition","SUBTRACTION":"Subtraction","MULTIPLICATION":"Multiplication / grouping","DIVISION":"Division / sharing"}


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
        return min((g for g in candidates if error_counts[g] == max_errors), key=GAME_ORDER.index), "REPEATED_ERROR_REINFORCEMENT"
    return min(candidates, key=lambda g: (by_game[g].get("last_assessed_at") or "", GAME_ORDER.index(g))), "LEAST_RECENTLY_ASSESSED"
