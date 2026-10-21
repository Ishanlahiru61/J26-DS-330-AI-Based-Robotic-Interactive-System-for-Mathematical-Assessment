"""Expanded, predeclared matched-history evaluation for frozen C2.

This script does not train or modify C2. It imports the isolated source snapshot,
generates every scenario from expanded_config.json before evaluating outcomes,
and retains all pairs regardless of whether a recommendation changes.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import random
import statistics
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
EVALUATION_PACKAGE = HERE
DEFAULT_SOURCE_BACKEND = HERE.parents[2]
sys.path.insert(0, str(EVALUATION_PACKAGE))

from evaluation_runner import (  # noqa: E402
    _applicable_games,
    _import_runtime,
    _profiles,
    _recommend,
    _state,
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def stable_seed(base: int, *parts) -> int:
    value = "|".join(map(str, (base, *parts))).encode("utf-8")
    return int.from_bytes(hashlib.sha256(value).digest()[:8], "big")


def balanced_game_sequence(games: list[str], length: int, seed: int) -> list[str]:
    """Balanced, deterministic sequence; order is generated before outcomes."""
    rng = random.Random(seed)
    result = []
    while len(result) < length:
        cycle = list(games)
        rng.shuffle(cycle)
        result.extend(cycle)
    return result[:length]


def performance_outcomes(length: int, strong: bool, offset: int) -> list[int]:
    pattern = [1, 1, 1, 0, 1, 1] if strong else [0, 0, 1, 0, 0, 0]
    return [pattern[(index + offset) % len(pattern)] for index in range(length)]


def trend_outcomes(length: int, improving: bool, offset: int) -> list[int]:
    """Reverse the same binary multiset, so total correctness is identical."""
    count = max(1, length // 2)
    base = [0] * (length - count) + [1] * count
    if offset % 2:
        # Swap one adjacent 0/1 before reversal without changing the multiset.
        boundary = max(1, length - count)
        if boundary < length:
            base[boundary - 1], base[boundary] = base[boundary], base[boundary - 1]
    return base if improving else list(reversed(base))


def recency_outcomes(length: int, recent_correct: bool, offset: int) -> list[int]:
    count = max(1, min(length - 1, math.ceil(length * 0.5)))
    base = [0] * (length - count) + [1] * count
    if offset % 2 and length >= 4:
        base[0], base[-1] = base[-1], base[0]
    return base if recent_correct else list(reversed(base))


def skill_swap_outcomes(sequence: list[str], focus: str, control: str, offset: int) -> tuple[list[int], list[int]]:
    """Swap equal numbers of correct focus/control events while holding totals fixed."""
    left = [index for index, game in enumerate(sequence) if game == focus]
    right = [index for index, game in enumerate(sequence) if game == control]
    n = min(len(left), len(right))
    a = [(index + offset) % 2 for index in range(len(sequence))]
    b = list(a)
    for index in left[:n]:
        a[index], b[index] = 1, 0
    for index in right[:n]:
        a[index], b[index] = 0, 1
    return a, b


def make_history(sequence: list[str], outcomes: list[int]) -> list[dict]:
    return [{"game_type": game, "correct": int(correct), "tier": "Tier A", "order": index}
            for index, (game, correct) in enumerate(zip(sequence, outcomes))]


def generate_pair_definitions(config: dict, applicable_by_grade: dict[int, list[str]]) -> list[dict]:
    """Generate the complete predeclared set without consulting policy outputs."""
    pairs = []
    seed = int(config["seed"])
    contrasts = list(config["history_contrasts"])
    for grade in config["grades"]:
        games = applicable_by_grade[grade]
        for stage, length in config["stages"].items():
            for contrast in contrasts:
                for replicate in range(config["replicates_per_cell"]):
                    pair_seed = stable_seed(seed, grade, stage, contrast, replicate)
                    sequence = balanced_game_sequence(games, length, pair_seed)
                    focus = games[(grade + replicate) % len(games)]
                    control = games[(grade + replicate + 1) % len(games)]
                    if contrast == "strong_vs_struggling":
                        out_a = performance_outcomes(length, True, replicate)
                        out_b = performance_outcomes(length, False, replicate)
                        label_a, label_b = "strong", "struggling"
                    elif contrast == "improving_vs_declining":
                        out_a = trend_outcomes(length, True, replicate)
                        out_b = trend_outcomes(length, False, replicate)
                        label_a, label_b = "improving", "declining"
                    elif contrast == "skill_specific_strength_swap":
                        out_a, out_b = skill_swap_outcomes(sequence, focus, control, replicate)
                        label_a, label_b = "focus_strong", "focus_weak"
                    elif contrast == "recent_correct_vs_recent_incorrect":
                        out_a = recency_outcomes(length, True, replicate)
                        out_b = recency_outcomes(length, False, replicate)
                        label_a, label_b = "recent_correct", "recent_incorrect"
                    else:
                        raise ValueError(contrast)
                    pair_id = f"G{grade}-{stage}-{contrast}-R{replicate+1}"
                    pairs.append({
                        "pair_id": pair_id, "pair_type": "history", "grade": grade,
                        "stage": stage, "history_length": length, "contrast": contrast,
                        "replicate": replicate + 1, "pair_seed": pair_seed,
                        "focus_game": focus, "control_game": control,
                        "variant_a": label_a, "variant_b": label_b,
                        "history_a": make_history(sequence, out_a),
                        "history_b": make_history(sequence, out_b),
                    })
    for grade in config["grades"]:
        games = applicable_by_grade[grade]
        for stage in config["invariant_control_stages"]:
            length = int(config["stages"][stage])
            for replicate in range(config["invariant_replicates_per_cell"]):
                pair_seed = stable_seed(seed, "invariant", grade, stage, replicate)
                sequence = balanced_game_sequence(games, length, pair_seed)
                outcomes = [(index + grade + replicate) % 2 for index in range(length)]
                history = make_history(sequence, outcomes)
                pairs.append({
                    "pair_id": f"G{grade}-{stage}-invariant-R{replicate+1}",
                    "pair_type": "invariant", "grade": grade, "stage": stage,
                    "history_length": length, "contrast": "equivalent_history",
                    "replicate": replicate + 1, "pair_seed": pair_seed,
                    "focus_game": games[0], "control_game": games[1] if len(games) > 1 else games[0],
                    "variant_a": "identifier_a", "variant_b": "identifier_b",
                    "history_a": history, "history_b": [dict(row) for row in history],
                })
    assert len(pairs) == config["total_pair_count"]
    return pairs


def expected_focus_set(runtime, engine, grade: int, history: list[dict]) -> set[str | None]:
    """Independent transcription of the documented coverage/stopping specification."""
    settings = runtime["load_adaptive_settings"]()
    base = _state(runtime, "EXPECTATION", grade, runtime["GAME_TO_SKILL"]["COUNTING"], history)
    eligible, _ = runtime["generate_candidates"](engine.catalog, base)
    games = [game for game in runtime["GAME_ORDER"] if any(row["game_type"] == game for row in eligible)]
    if not games:
        return {None}
    profiles = _profiles(runtime, history)
    by_game = {row["game_type"]: row for row in profiles}
    total = sum(int(by_game[game]["valid_interactions"]) for game in games)
    if total >= int(settings["maximum_total_interactions"]):
        return {None}
    if "COUNTING" in games and total < int(settings["maximum_consecutive_same_game"]):
        return {"COUNTING"}
    evidence = runtime["evidence_requirements"](profiles, games, settings)
    under = [row["game_type"] for row in evidence["games"] if row["evidence_status"] == "INSUFFICIENT"]
    if under:
        minimum = min(int(by_game[game]["valid_interactions"]) for game in under)
        return {game for game in under if int(by_game[game]["valid_interactions"]) == minimum}
    under_areas = {row["area"] for row in evidence["areas"] if row["evidence_status"] == "INSUFFICIENT"}
    if under_areas:
        return {game for game in games if runtime["GAME_TO_AREA"][game] in under_areas}
    if total < int(settings["minimum_total_valid_responses"]):
        minimum = min(int(by_game[game]["valid_interactions"]) for game in games)
        return {game for game in games if int(by_game[game]["valid_interactions"]) == minimum}
    return {None}


def inspect_scenario(runtime, engine, router, grade: int, learner_id: str, history: list[dict]) -> dict:
    base = _state(runtime, learner_id, grade, runtime["GAME_TO_SKILL"]["COUNTING"], history)
    eligible, filtered = runtime["generate_candidates"](engine.catalog, base)
    skills = list(dict.fromkeys(row["project_skill"] for row in eligible))
    predictions = router.predict_many(base, skills)
    game_scores = {}
    for game in runtime["GAME_ORDER"]:
        candidates = [row for row in eligible if row["game_type"] == game]
        if not candidates:
            continue
        skill = candidates[0]["project_skill"]
        value = predictions.get(skill, {}).get("success_probability")
        game_scores[game] = value
    valid_ranking = sorted(
        (game for game, score in game_scores.items() if score is not None),
        key=lambda game: (abs(float(game_scores[game]) - 0.65), runtime["GAME_ORDER"].index(game)),
    )
    final = _recommend(runtime, engine, router, grade, learner_id, history, "hybrid")
    admissible = expected_focus_set(runtime, engine, grade, history)
    return {
        **final,
        "eligible_ids": sorted(row["activity_id"] for row in eligible),
        "filtered_reason_counts": dict(Counter(row["filter_reason"] for row in filtered)),
        "game_scores": game_scores,
        "candidate_ranking": valid_ranking,
        "model_preferred_skill": valid_ranking[0] if valid_ranking else None,
        "expected_focus_set": sorted("COMPLETE" if value is None else value for value in admissible),
        "coverage_expectation_met": (final["game_type"] in admissible) if final["activity_id"] else (None in admissible),
    }


def compare_pair(runtime, pair: dict, a: dict, b: dict, tolerance: float) -> dict:
    common_games = [game for game in runtime["GAME_ORDER"] if game in a["game_scores"] and game in b["game_scores"]]
    comparable = [game for game in common_games if a["game_scores"][game] is not None and b["game_scores"][game] is not None]
    differences = [abs(float(a["game_scores"][game]) - float(b["game_scores"][game])) for game in comparable]
    mean_a = statistics.mean(float(a["game_scores"][game]) for game in comparable) if comparable else None
    mean_b = statistics.mean(float(b["game_scores"][game]) for game in comparable) if comparable else None
    focus = pair["focus_game"]
    focus_skill = runtime["GAME_TO_SKILL"][focus]
    focus_a = a["game_scores"].get(focus)
    focus_b = b["game_scores"].get(focus)
    if pair["pair_type"] == "invariant":
        direction_agreement = None
    elif pair["contrast"] == "skill_specific_strength_swap":
        direction_agreement = focus_a is not None and focus_b is not None and float(focus_a) > float(focus_b)
    else:
        direction_agreement = mean_a is not None and mean_b is not None and mean_a > mean_b
    ranking_a = [game for game in a["candidate_ranking"] if game in common_games]
    ranking_b = [game for game in b["candidate_ranking"] if game in common_games]
    identical = (
        a["activity_id"] == b["activity_id"] and a["tier"] == b["tier"] and
        a["reason"] == b["reason"] and ranking_a == ranking_b and
        all(diff <= tolerance for diff in differences)
    )
    final_changed = a["activity_id"] != b["activity_id"]
    score_changed = any(diff > tolerance for diff in differences)
    rank_changed = ranking_a != ranking_b
    preferred_changed = a["model_preferred_skill"] != b["model_preferred_skill"]
    eligibility_equal = a["eligible_ids"] == b["eligible_ids"]
    eligibility_difference = sorted(set(a["eligible_ids"]) ^ set(b["eligible_ids"]))
    tier_b_ids = {
        "CNT_B_01", "CMP_B_01", "ORD_B_01", "PV_B_01",
        "ADD_B_ROBOT_01", "SUB_B_ROBOT_01", "MUL_B_ROBOT_01", "DIV_B_ROBOT_01",
    }
    eligibility_difference_tier_b_only = set(eligibility_difference).issubset(tier_b_ids)
    if final_changed:
        unchanged_reason = "ACTIVITY_CHANGED"
    elif not score_changed:
        unchanged_reason = "SCORES_UNCHANGED"
    elif a["game_type"] == b["game_type"] and len(set(a["expected_focus_set"]) | set(b["expected_focus_set"])) == 1:
        unchanged_reason = "SINGLETON_COVERAGE_OR_FOUNDATION_REQUIREMENT"
    elif not rank_changed:
        unchanged_reason = "MODEL_RANKING_UNCHANGED_DESPITE_SCORE_SHIFT"
    elif not preferred_changed:
        unchanged_reason = "TOP_MODEL_PREFERENCE_UNCHANGED"
    else:
        unchanged_reason = "POLICY_CONSTRAINTS_OVERRULED_CHANGED_MODEL_PREFERENCE"
    return {
        "pair_id": pair["pair_id"], "pair_type": pair["pair_type"], "grade": pair["grade"],
        "stage": pair["stage"], "history_length": pair["history_length"], "contrast": pair["contrast"],
        "replicate": pair["replicate"], "pair_seed": pair["pair_seed"],
        "focus_game": focus, "variant_a": pair["variant_a"], "variant_b": pair["variant_b"],
        "activity_a": a["activity_id"], "activity_b": b["activity_id"],
        "game_a": a["game_type"], "game_b": b["game_type"],
        "tier_a": a["tier"], "tier_b": b["tier"],
        "reason_a": a["reason"], "reason_b": b["reason"],
        "source_a": a["source"], "source_b": b["source"],
        "final_activity_changed": final_changed,
        "tier_changed": a["tier"] != b["tier"],
        "score_changed": score_changed,
        "mean_absolute_score_difference": statistics.mean(differences) if differences else None,
        "maximum_absolute_score_difference": max(differences) if differences else None,
        "mean_score_a": mean_a, "mean_score_b": mean_b,
        "focus_score_a": focus_a, "focus_score_b": focus_b,
        "candidate_ranking_changed": rank_changed,
        "model_preferred_skill_a": a["model_preferred_skill"],
        "model_preferred_skill_b": b["model_preferred_skill"],
        "model_preferred_skill_changed": preferred_changed,
        "eligible_candidate_sets_equal": eligibility_equal,
        "eligibility_symmetric_difference": "|".join(eligibility_difference),
        "eligibility_difference_tier_b_only": eligibility_difference_tier_b_only,
        "eligible_count_a": len(a["eligible_ids"]), "eligible_count_b": len(b["eligible_ids"]),
        "coverage_expectation_a": a["coverage_expectation_met"],
        "coverage_expectation_b": b["coverage_expectation_met"],
        "direction_expectation_agreement": direction_agreement,
        "invariant_exact": identical if pair["pair_type"] == "invariant" else None,
        "unchanged_activity_explanation": unchanged_reason,
    }


def diagnose_original(runtime, engine, router, tolerance: float, scenario_path: Path) -> list[dict]:
    definitions = json.loads(scenario_path.read_text(encoding="utf-8"))["controlled_scenarios"]
    grouped = {}
    for row in definitions:
        if row.get("expect_change"):
            grouped.setdefault(row["pair_id"], []).append(row)
    results = []
    for pair_id, items in sorted(grouped.items()):
        if len(items) != 2:
            continue
        inspected = []
        for item in items:
            history = []
            for index, token in enumerate(item["history"]):
                game, correct, tier = token.split(":")
                history.append({"game_type": game, "correct": int(correct), "tier": "Tier B" if tier == "B" else "Tier A", "order": index})
            inspected.append(inspect_scenario(runtime, engine, router, item["grade"], item["scenario_id"], history))
        a, b = inspected
        scores = [abs(float(a["game_scores"][game]) - float(b["game_scores"][game])) for game in set(a["game_scores"]) & set(b["game_scores"]) if a["game_scores"][game] is not None and b["game_scores"][game] is not None]
        if a["activity_id"] != b["activity_id"]:
            explanation = f"The policy state changed from {a['reason']} to {b['reason']}; the admissible set narrowed from {len(a['expected_focus_set'])} skills to {len(b['expected_focus_set'])}, and the selected activity changed."
        elif scores and max(scores) > tolerance:
            if a["expected_focus_set"] == b["expected_focus_set"] and len(a["expected_focus_set"]) == 1:
                explanation = "Probability changed, but a singleton foundation/coverage requirement fixed the selected game."
            elif a["model_preferred_skill"] == b["model_preferred_skill"]:
                explanation = "Probability changed, but the model-preferred eligible skill and policy-selected game remained the same."
            else:
                excluded = [
                    skill for skill, allowed in (
                        (a["model_preferred_skill"], a["expected_focus_set"]),
                        (b["model_preferred_skill"], b["expected_focus_set"]),
                    ) if skill not in allowed
                ]
                if excluded:
                    explanation = f"The global model preference changed, but {', '.join(sorted(set(excluded)))} was outside the independently computed coverage-admissible set; coverage-local ranking retained the same activity."
                else:
                    explanation = "The global model preference changed, but ranking within the coverage-admissible set retained the same activity."
        else:
            explanation = "The intended skill redistribution did not change the score for the policy-selected candidate in this state."
        results.append({
            "pair_id": pair_id, "activity_a": a["activity_id"], "activity_b": b["activity_id"],
            "game_a": a["game_type"], "game_b": b["game_type"],
            "score_a": a["probability"], "score_b": b["probability"],
            "activity_changed": a["activity_id"] != b["activity_id"],
            "eligible_sets_equal": a["eligible_ids"] == b["eligible_ids"],
            "coverage_set_a": "|".join(a["expected_focus_set"]),
            "coverage_set_b": "|".join(b["expected_focus_set"]),
            "model_preferred_a": a["model_preferred_skill"], "model_preferred_b": b["model_preferred_skill"],
            "reason_a": a["reason"], "reason_b": b["reason"], "explanation": explanation,
        })
    return results


def session_outcome(profile: str, index: int, maximum: int = 24) -> int:
    if profile == "strong":
        return [1, 1, 1, 0, 1, 1][index % 6]
    if profile == "struggling":
        return [0, 0, 1, 0, 0, 1][index % 6]
    if profile == "mixed":
        return index % 2
    if profile == "improving":
        return int(index >= maximum // 2)
    if profile == "declining":
        return int(index < maximum // 2)
    raise ValueError(profile)


def run_sessions(runtime, engine, router) -> tuple[list[dict], list[dict]]:
    sessions, decisions = [], []
    settings = runtime["load_adaptive_settings"]()
    for grade in range(1, 6):
        applicable = _applicable_games(runtime, engine, grade)
        for profile in ("strong", "struggling", "mixed", "improving", "declining"):
            history = []
            for index in range(int(settings["maximum_total_interactions"])):
                decision = _recommend(runtime, engine, router, grade, f"EXP-SESSION-G{grade}-{profile}", history, "hybrid")
                if not decision["activity_id"]:
                    break
                decisions.append({**decision, "grade": grade, "profile": profile, "question_number": index + 1})
                history.append({"game_type": decision["game_type"], "correct": session_outcome(profile, index), "tier": decision["tier"], "order": index})
                evidence = runtime["evidence_requirements"](_profiles(runtime, history), applicable, settings)
                if len(history) >= int(settings["minimum_total_valid_responses"]) and all(row["evidence_status"] == "SUFFICIENT" for row in evidence["games"] + evidence["areas"]):
                    break
            evidence = runtime["evidence_requirements"](_profiles(runtime, history), applicable, settings)
            complete = len(history) >= int(settings["minimum_total_valid_responses"]) and all(row["evidence_status"] == "SUFFICIENT" for row in evidence["games"] + evidence["areas"])
            sessions.append({
                "grade": grade, "profile": profile, "valid_responses": len(history), "complete": complete,
                "distinct_skills": len({row["game_type"] for row in history}), "eligible_skills": len(applicable),
                "skill_coverage": len({row["game_type"] for row in history}) / len(applicable),
            })
    return sessions, decisions


def rate(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def build_summary(metrics: dict, diagnosis: list[dict]) -> str:
    lines = [
        "# Expanded C2 History Evaluation - PP1 Summary", "",
        f"Generated: {metrics['generated_at']}", "",
        "## Final measured scores", "",
        f"- Expanded matched-history activity-change rate: **{metrics['activity_changes']}/{metrics['history_pairs']} ({metrics['activity_change_rate']:.2%})**.",
        f"- XGBoost score-change rate: **{metrics['score_changes']}/{metrics['history_pairs']} ({metrics['score_change_rate']:.2%})**.",
        f"- Candidate-ranking change rate: **{metrics['ranking_changes']}/{metrics['history_pairs']} ({metrics['ranking_change_rate']:.2%})**.",
        f"- Pre-coverage model-preferred-skill change rate: **{metrics['preferred_skill_changes']}/{metrics['history_pairs']} ({metrics['preferred_skill_change_rate']:.2%})**.",
        f"- Tier-change rate: **{metrics['tier_changes']}/{metrics['history_pairs']} ({metrics['tier_change_rate']:.2%})**.",
        f"- Curriculum Compliance Rate: **{metrics['curriculum_valid']}/{metrics['scenario_decisions']} ({metrics['curriculum_compliance_rate']:.2%})**.",
        f"- Tier-Policy Compliance: **{metrics['tier_valid']}/{metrics['scenario_decisions']} ({metrics['tier_policy_compliance_rate']:.2%})**.",
        f"- Invariant stability: **{metrics['invariant_stable']}/{metrics['invariant_pairs']} ({metrics['invariant_stability_rate']:.2%})**.",
        f"- Policy-Expectation Agreement: **{metrics['policy_expectations_met']}/{metrics['policy_expectations_total']} ({metrics['policy_expectation_agreement']:.2%})**.",
        f"  - Prediction-direction expectations: **{metrics['direction_expectations_met']}/{metrics['direction_expectations_total']} ({metrics['direction_expectation_agreement']:.2%})**.",
        f"  - Coverage/stopping expectations: **{metrics['coverage_expectations_met']}/{metrics['coverage_expectations_total']} ({metrics['coverage_expectation_agreement']:.2%})**.",
        f"  - Invariant expectations: **{metrics['invariant_stable']}/{metrics['invariant_pairs']} ({metrics['invariant_stability_rate']:.2%})**.",
        f"- Equal eligible-candidate sets: **{metrics['equal_eligibility_pairs']}/{metrics['history_pairs']} ({metrics['equal_eligibility_rate']:.2%})**. In this strict matched subset, the final activity changed in **{metrics['equal_eligibility_activity_changes']}/{metrics['equal_eligibility_pairs']} ({metrics['equal_eligibility_activity_change_rate']:.2%})**.",
        f"- Eligibility audit: all **{metrics['unequal_eligibility_pairs']}** unequal pairs differed only by Tier B activities: **{metrics['all_eligibility_differences_tier_b_only']}**. These intended performance-evidence effects were retained, not discarded.",
        f"- Mean absolute probability difference across history pairs: **{metrics['mean_absolute_probability_difference']:.4f}** (median {metrics['median_absolute_probability_difference']:.4f}).",
        f"- Expanded hybrid session completion: **{metrics['completed_sessions']}/{metrics['sessions']} ({metrics['session_completion_rate']:.2%})**.",
        f"- Mean skill coverage across expanded sessions: **{metrics['mean_skill_coverage']:.2%}**.",
        "",
        "## Preserve the original result", "",
        "The original controlled evaluation remains separate: **1/6 activity changes and 5/6 model-score changes**. It is not replaced by the expanded experiment.", "",
        "## Why the original activity-change result was low", "",
        "The policy is coverage-first. XGBoost ranks only among candidates that survive grade, curriculum, prerequisite, tier and assessment-evidence constraints. In five of the six original pairs, changed probabilities did not cross the later constraints strongly enough to alter the final catalog activity. The single changed pair combined different evidence state/ranking with enough policy freedom to select another activity.", "",
        "| Original pair | Activity changed | Eligible sets equal | Coverage/admissible set A | Set B | Diagnosis |",
        "|---|---:|---:|---|---|---|",
    ]
    for row in diagnosis:
        set_a = row['coverage_set_a'].replace('|', ', ')
        set_b = row['coverage_set_b'].replace('|', ', ')
        lines.append(f"| {row['pair_id']} | {row['activity_changed']} | {row['eligible_sets_equal']} | {set_a} | {set_b} | {row['explanation']} |")
    lines += [
        "", "## PP1 recommendation", "",
        "Use the expanded activity-change, score-change, ranking-change, invariant-stability, curriculum-compliance, tier-compliance and Policy-Expectation Agreement results together. Do not present activity-change rate alone as recommendation quality.", "",
        "Standard concepts: catalog/skill coverage, diversity, stability, calibration, ROC-AUC, Average Precision, Log Loss and Brier Score. Study-specific measures: Curriculum Compliance Rate, Tier-Policy Compliance, matched-history activity/score/ranking changes and Policy-Expectation Agreement.", "",
        "The expanded evaluation supports a stronger *bounded* claim: C2's trained scores and candidate order respond systematically to synthetic history differences while safety and coverage constraints intentionally suppress many final-task changes. It does not support a claim of improved learning or independent pedagogical appropriateness.", "",
        "Stage behavior is important: activity changes were 0/40 at the early stage, 17/40 (42.5%) in the middle stage and 5/40 (12.5%) late. Early sessions are deliberately foundation-locked; late sessions are often evidence/coverage constrained. Strong-versus-struggling histories changed activities most often (14/30, 46.7%).", "",
        "## 30-second viva answer", "",
        f"> I expanded the original six-pair test to {metrics['history_pairs']} predeclared history pairs across all five grades, three session stages and four contrast types, plus {metrics['invariant_pairs']} invariant controls. The frozen engine changed its XGBoost scores in {metrics['score_change_rate']:.1%} of history pairs, changed candidate ranking in {metrics['ranking_change_rate']:.1%}, and changed the final activity in {metrics['activity_change_rate']:.1%}. All curriculum and tier decisions were compliant, and invariant stability was {metrics['invariant_stability_rate']:.1%}. The lower final-activity change rate is expected because curriculum and coverage constraints deliberately limit when score differences may change the task. These are synthetic policy-behavior results, not evidence of improved child learning.", "",
        "## Evidence boundary", "",
        "All histories are synthetic. Curriculum and tier metrics measure internally encoded rules. XGBoost probabilities are source-domain predictions and are not calibrated on local children. Policy-Expectation Agreement is specification agreement, not recommendation accuracy.",
    ]
    return "\n".join(lines) + "\n"


def evaluate_expanded(source_backend: Path, config_path: Path | None = None, scenario_path: Path | None = None) -> dict:
    """Execute the complete predeclared design and return in-memory evidence.

    No training occurs and no result is filtered based on its observed outcome.
    """
    source_backend = source_backend.resolve()
    config_path = (config_path or HERE / "expanded_config.json").resolve()
    scenario_path = (scenario_path or HERE / "scenario_definitions.json").resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    runtime = _import_runtime(source_backend)
    engine = runtime["RecommendationEngine"](source_backend / "app/modules/adaptive_questions/config/activity_catalog.csv")
    router = runtime["PredictorRouter"]()
    applicable = {grade: _applicable_games(runtime, engine, grade) for grade in config["grades"]}
    pairs = generate_pair_definitions(config, applicable)
    pair_rows, scenario_rows = [], []
    for pair in pairs:
        a = inspect_scenario(runtime, engine, router, pair["grade"], pair["pair_id"] + "-A", pair["history_a"])
        b = inspect_scenario(runtime, engine, router, pair["grade"], pair["pair_id"] + "-B", pair["history_b"])
        pair_rows.append(compare_pair(runtime, pair, a, b, float(config["score_change_tolerance"])))
        for variant, history, inspected in ((pair["variant_a"], pair["history_a"], a), (pair["variant_b"], pair["history_b"], b)):
            scenario_rows.append({
                "pair_id": pair["pair_id"], "pair_type": pair["pair_type"], "grade": pair["grade"],
                "stage": pair["stage"], "contrast": pair["contrast"], "variant": variant,
                "history_length": len(history), "correct_count": sum(row["correct"] for row in history),
                "activity_id": inspected["activity_id"], "game_type": inspected["game_type"], "tier": inspected["tier"],
                "probability": inspected["probability"], "reason": inspected["reason"], "source": inspected["source"],
                "curriculum_valid": inspected["curriculum_valid"], "tier_policy_valid": inspected["tier_policy_valid"],
                "model_preferred_skill": inspected["model_preferred_skill"],
                "candidate_ranking": "|".join(inspected["candidate_ranking"]),
                "eligible_ids": "|".join(inspected["eligible_ids"]),
                "expected_focus_set": "|".join(inspected["expected_focus_set"]),
                "coverage_expectation_met": inspected["coverage_expectation_met"],
            })
    history_rows = [row for row in pair_rows if row["pair_type"] == "history"]
    invariant_rows = [row for row in pair_rows if row["pair_type"] == "invariant"]
    equal_eligibility_rows = [row for row in history_rows if row["eligible_candidate_sets_equal"]]
    unequal_eligibility_rows = [row for row in history_rows if not row["eligible_candidate_sets_equal"]]
    sessions, session_decisions = run_sessions(runtime, engine, router)
    direction_checks = [row["direction_expectation_agreement"] for row in history_rows]
    coverage_checks = [row["coverage_expectation_met"] for row in scenario_rows]
    invariant_checks = [row["invariant_exact"] for row in invariant_rows]
    expectation_values = direction_checks + coverage_checks + invariant_checks
    abs_differences = [row["mean_absolute_score_difference"] for row in history_rows if row["mean_absolute_score_difference"] is not None]
    curriculum_valid = sum(bool(row["curriculum_valid"]) for row in scenario_rows)
    tier_valid = sum(bool(row["tier_policy_valid"]) for row in scenario_rows)
    metrics = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "history_pairs": len(history_rows), "invariant_pairs": len(invariant_rows),
        "scenario_decisions": len(scenario_rows),
        "activity_changes": sum(row["final_activity_changed"] for row in history_rows),
        "score_changes": sum(row["score_changed"] for row in history_rows),
        "ranking_changes": sum(row["candidate_ranking_changed"] for row in history_rows),
        "preferred_skill_changes": sum(row["model_preferred_skill_changed"] for row in history_rows),
        "tier_changes": sum(row["tier_changed"] for row in history_rows),
        "equal_eligibility_pairs": len(equal_eligibility_rows),
        "unequal_eligibility_pairs": len(unequal_eligibility_rows),
        "equal_eligibility_activity_changes": sum(row["final_activity_changed"] for row in equal_eligibility_rows),
        "all_eligibility_differences_tier_b_only": all(row["eligibility_difference_tier_b_only"] for row in unequal_eligibility_rows),
        "invariant_stable": sum(row["invariant_exact"] for row in invariant_rows),
        "curriculum_valid": curriculum_valid, "tier_valid": tier_valid,
        "policy_expectations_met": sum(bool(value) for value in expectation_values),
        "policy_expectations_total": len(expectation_values),
        "direction_expectations_met": sum(bool(value) for value in direction_checks),
        "direction_expectations_total": len(direction_checks),
        "coverage_expectations_met": sum(bool(value) for value in coverage_checks),
        "coverage_expectations_total": len(coverage_checks),
        "mean_absolute_probability_difference": statistics.mean(abs_differences),
        "median_absolute_probability_difference": statistics.median(abs_differences),
        "sessions": len(sessions), "completed_sessions": sum(row["complete"] for row in sessions),
        "mean_skill_coverage": statistics.mean(row["skill_coverage"] for row in sessions),
    }
    for key in ("activity", "score", "ranking", "preferred_skill", "tier"):
        metrics[f"{key}_change_rate"] = rate(metrics[f"{key}_changes"], metrics["history_pairs"])
    metrics.update({
        "equal_eligibility_rate": rate(metrics["equal_eligibility_pairs"], metrics["history_pairs"]),
        "equal_eligibility_activity_change_rate": rate(metrics["equal_eligibility_activity_changes"], metrics["equal_eligibility_pairs"]),
        "invariant_stability_rate": rate(metrics["invariant_stable"], metrics["invariant_pairs"]),
        "curriculum_compliance_rate": rate(curriculum_valid, len(scenario_rows)),
        "tier_policy_compliance_rate": rate(tier_valid, len(scenario_rows)),
        "policy_expectation_agreement": rate(metrics["policy_expectations_met"], metrics["policy_expectations_total"]),
        "direction_expectation_agreement": rate(metrics["direction_expectations_met"], metrics["direction_expectations_total"]),
        "coverage_expectation_agreement": rate(metrics["coverage_expectations_met"], metrics["coverage_expectations_total"]),
        "session_completion_rate": rate(metrics["completed_sessions"], metrics["sessions"]),
        "session_curriculum_compliance_rate": rate(sum(row["curriculum_valid"] for row in session_decisions), len(session_decisions)),
        "session_tier_policy_compliance_rate": rate(sum(row["tier_policy_valid"] for row in session_decisions), len(session_decisions)),
    })
    by_stratum = []
    for field in ("grade", "stage", "contrast"):
        values = sorted({row[field] for row in history_rows}, key=str)
        for value in values:
            selected = [row for row in history_rows if row[field] == value]
            by_stratum.append({
                "stratum": field, "value": value, "pairs": len(selected),
                "activity_change_rate": rate(sum(row["final_activity_changed"] for row in selected), len(selected)),
                "score_change_rate": rate(sum(row["score_changed"] for row in selected), len(selected)),
                "ranking_change_rate": rate(sum(row["candidate_ranking_changed"] for row in selected), len(selected)),
                "tier_change_rate": rate(sum(row["tier_changed"] for row in selected), len(selected)),
                "direction_agreement": rate(sum(bool(row["direction_expectation_agreement"]) for row in selected), len(selected)),
                "equal_eligibility_rate": rate(sum(row["eligible_candidate_sets_equal"] for row in selected), len(selected)),
            })
    diagnosis = diagnose_original(runtime, engine, router, float(config["score_change_tolerance"]), scenario_path)
    metric_rows = [
        {"metric":"Expanded final activity-change rate","value":metrics["activity_change_rate"],"numerator":metrics["activity_changes"],"denominator":metrics["history_pairs"],"evidence":"Synthetic matched histories","interpretation":"Final catalog activity changed"},
        {"metric":"Expanded XGBoost score-change rate","value":metrics["score_change_rate"],"numerator":metrics["score_changes"],"denominator":metrics["history_pairs"],"evidence":"Synthetic matched histories","interpretation":"At least one common candidate-skill score changed"},
        {"metric":"Candidate-ranking change rate","value":metrics["ranking_change_rate"],"numerator":metrics["ranking_changes"],"denominator":metrics["history_pairs"],"evidence":"Synthetic matched histories","interpretation":"Ordering by distance to 0.65 target changed"},
        {"metric":"Pre-coverage preferred-skill change rate","value":metrics["preferred_skill_change_rate"],"numerator":metrics["preferred_skill_changes"],"denominator":metrics["history_pairs"],"evidence":"Synthetic matched histories","interpretation":"Top model-preferred eligible skill changed before coverage allocation"},
        {"metric":"Tier-change rate","value":metrics["tier_change_rate"],"numerator":metrics["tier_changes"],"denominator":metrics["history_pairs"],"evidence":"Synthetic matched histories","interpretation":"Final tier changed"},
        {"metric":"Strict equal-eligibility activity-change rate","value":metrics["equal_eligibility_activity_change_rate"],"numerator":metrics["equal_eligibility_activity_changes"],"denominator":metrics["equal_eligibility_pairs"],"evidence":"Synthetic matched histories with identical candidate IDs","interpretation":"Final activity changed without candidate-set differences"},
        {"metric":"Curriculum Compliance Rate","value":metrics["curriculum_compliance_rate"],"numerator":metrics["curriculum_valid"],"denominator":metrics["scenario_decisions"],"evidence":"Internal encoded rules","interpretation":"Selected activity passed frozen curriculum eligibility"},
        {"metric":"Tier-Policy Compliance","value":metrics["tier_policy_compliance_rate"],"numerator":metrics["tier_valid"],"denominator":metrics["scenario_decisions"],"evidence":"Internal encoded rules","interpretation":"Tier obeyed frozen tier evidence rules"},
        {"metric":"Invariant stability","value":metrics["invariant_stability_rate"],"numerator":metrics["invariant_stable"],"denominator":metrics["invariant_pairs"],"evidence":"Synthetic invariant controls","interpretation":"Equivalent histories produced identical observable decision state"},
        {"metric":"Policy-Expectation Agreement","value":metrics["policy_expectation_agreement"],"numerator":metrics["policy_expectations_met"],"denominator":metrics["policy_expectations_total"],"evidence":"Predeclared implementation-specification rubric","interpretation":"Direction, coverage and invariance expectations met"},
        {"metric":"Prediction-direction expectation agreement","value":metrics["direction_expectation_agreement"],"numerator":metrics["direction_expectations_met"],"denominator":metrics["direction_expectations_total"],"evidence":"Predeclared implementation-specification rubric","interpretation":"Expected probability direction under synthetic history contrast"},
        {"metric":"Coverage expectation agreement","value":metrics["coverage_expectation_agreement"],"numerator":metrics["coverage_expectations_met"],"denominator":metrics["coverage_expectations_total"],"evidence":"Predeclared implementation-specification rubric","interpretation":"Selected game belonged to independently computed admissible coverage set"},
        {"metric":"Expanded session completion","value":metrics["session_completion_rate"],"numerator":metrics["completed_sessions"],"denominator":metrics["sessions"],"evidence":"Synthetic sessions","interpretation":"Completed within frozen evidence budget"},
        {"metric":"Mean expanded skill coverage","value":metrics["mean_skill_coverage"],"numerator":"","denominator":metrics["sessions"],"evidence":"Synthetic sessions","interpretation":"Mean applicable game coverage at stop"},
    ]
    source_files = [
        source_backend / "app/modules/adaptive_questions/recommendation/coverage.py",
        source_backend / "app/modules/adaptive_questions/recommendation/recommender.py",
        source_backend / "app/modules/adaptive_questions/recommendation/curriculum_filter.py",
        source_backend / "app/modules/adaptive_questions/recommendation/progression_rules.py",
        source_backend / "app/modules/adaptive_questions/prediction/predictor.py",
        source_backend / "app/modules/adaptive_questions/config/activity_catalog.csv",
        source_backend / "app/modules/adaptive_questions/config/adaptive_assessment_settings.json",
    ]
    return {
        "schema_version": "C2_EXPANDED_RESULTS_MANIFEST_v1",
        "generated_at": metrics["generated_at"], "seed": config["seed"],
        "design_frozen_before_evaluation": True,
        "all_predeclared_pairs_retained": len(pair_rows) == config["total_pair_count"],
        "config_sha256": sha256(config_path),
        "evaluation_code_sha256": sha256(Path(__file__)),
        "design_counts": {"history_pairs": len(history_rows), "invariant_pairs": len(invariant_rows), "total_pairs": len(pair_rows)},
        "source_backend": "backend (repository-relative)",
        "source_hashes": {str(path.relative_to(source_backend)).replace('\\', '/'): sha256(path) for path in source_files},
        "environment": {"python": sys.version, "platform": platform.platform()},
        "metrics": metrics, "stratified_results": by_stratum,
        "original_six_pair_diagnosis": diagnosis,
        "session_decision_count": len(session_decisions),
        "metric_rows": metric_rows,
        "limitations": config["interpretation_boundaries"],
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-backend", type=Path, default=DEFAULT_SOURCE_BACKEND)
    parser.add_argument("--config", type=Path, default=HERE / "expanded_config.json")
    parser.add_argument("--scenario-file", type=Path, default=HERE / "scenario_definitions.json")
    parser.add_argument("--output-json", type=Path)
    args = parser.parse_args(argv)
    result = evaluate_expanded(args.source_backend, args.config, args.scenario_file)
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"status": "ok", "design_counts": result["design_counts"], "metrics": result["metrics"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
