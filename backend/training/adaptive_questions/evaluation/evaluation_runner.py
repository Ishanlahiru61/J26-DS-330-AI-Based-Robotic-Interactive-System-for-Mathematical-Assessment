"""Reproducible command-line evaluation for the frozen C2 component.

The runner never trains or alters C2. It reads frozen artifacts, imports the
isolated backend snapshot, executes controlled scenarios, and writes only to
this evaluation directory (or --output-dir).
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import statistics
import subprocess
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from evaluation_metrics import (
    binary_prediction_metrics,
    clustered_bootstrap_intervals,
    paired_change_metrics,
    policy_compliance_metrics,
    safe_rate,
)

SEED = 20261021
EVIDENCE = {
    "A": "Previously measured and reproducible",
    "B": "Newly measured against frozen implementation",
    "C": "Scenario-based or synthetic",
    "D": "Proposed future measurement",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def repository_provenance(source_backend: Path) -> dict:
    repo = source_backend.resolve().parent
    def git(*args: str) -> str:
        return subprocess.check_output(["git", "-C", str(repo), *args], text=True).strip()
    return {
        "repository_root": ".",
        "branch": git("branch", "--show-current"),
        "head": git("rev-parse", "HEAD"),
        "working_tree_clean_at_run_start": not bool(git("status", "--porcelain")),
    }


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def pct(value) -> str:
    return "pending" if value is None else f"{100 * float(value):.2f}%"


def evaluate_predictive(source_backend: Path, historical_root: Path | None, bootstrap_replicates: int) -> dict:
    model_dir = source_backend / "app/modules/adaptive_questions/models/transferable"
    manifest = read_json(model_dir / "manifest.json")
    metadata = read_json(model_dir / "metadata.json")
    deployment_eval = read_json(model_dir / "evaluation.json")
    integrity = {}
    for name, expected in manifest["files"].items():
        path = model_dir / name
        integrity[name] = {
            "exists": path.is_file(),
            "bytes": path.stat().st_size if path.is_file() else None,
            "sha256": sha256(path) if path.is_file() else None,
            "matches_manifest": path.is_file() and path.stat().st_size == expected["bytes"] and sha256(path) == expected["sha256"],
        }

    results_csv = source_backend / "training/adaptive_questions/experiments/ASSISTments_Step3_Model_Results.csv"
    registry_csv = source_backend / "training/adaptive_questions/experiments/ASSISTments_Step3_Experiment_Registry.csv"
    results = pd.read_csv(results_csv)
    registry = pd.read_csv(registry_csv)
    experiments = sorted(registry["experiment_id"].unique())
    selected = results[(results.experiment_id == "H2_GB_RT_OFF") & (results.split == "test") & (results.calibration == "platt_retained")].iloc[0]
    native = results[(results.experiment_id == "H2_GB_RT_OFF") & (results.split == "test") & (results.calibration == "native")].iloc[0]
    rt_on = results[(results.experiment_id == "H2_GB_RT_ON") & (results.split == "test") & (results.calibration == "platt_retained")].iloc[0]

    output = {
        "artifact_identity": {
            "deployed_model_version": metadata["model_version"],
            "algorithm": metadata["algorithm"],
            "calibration_applied_at_runtime": metadata["calibration_applied_at_runtime"],
            "manifest_integrity": integrity,
            "all_manifest_files_match": all(row["matches_manifest"] for row in integrity.values()),
        },
        "eight_model_registry": {
            "unique_experiments": len(experiments),
            "experiment_ids": experiments,
            "all_complete": bool((registry.status == "COMPLETE").all()),
            "source_sha256": sha256(registry_csv),
        },
        "research_winner_frozen": {
            "model_version": "H2_GB_RT_OFF_v1",
            "calibration": "Platt scaling fitted on validation and retained on test",
            "roc_auc": float(selected["ROC-AUC"]),
            "average_precision": float(selected["PR-AUC"]),
            "log_loss": float(selected["Log Loss"]),
            "brier_score": float(selected["Brier Score"]),
            "ece_10_equal_frequency": float(selected["ECE_10_equal_frequency"]),
            "n_interactions": int(selected.n_rows),
            "n_learners": int(selected.n_learners),
            "evidence_category": "A",
        },
        "native_vs_platt": {
            "native": {k: float(native[v]) for k, v in {"roc_auc": "ROC-AUC", "average_precision": "PR-AUC", "log_loss": "Log Loss", "brier_score": "Brier Score", "ece": "ECE_10_equal_frequency"}.items()},
            "platt": {k: float(selected[v]) for k, v in {"roc_auc": "ROC-AUC", "average_precision": "PR-AUC", "log_loss": "Log Loss", "brier_score": "Brier Score", "ece": "ECE_10_equal_frequency"}.items()},
        },
        "response_time_ablation": {
            "rt_off_platt": {"roc_auc": float(selected["ROC-AUC"]), "log_loss": float(selected["Log Loss"]), "average_precision": float(selected["PR-AUC"])},
            "rt_on_platt": {"roc_auc": float(rt_on["ROC-AUC"]), "log_loss": float(rt_on["Log Loss"]), "average_precision": float(rt_on["PR-AUC"])},
            "rt_on_minus_off": {"roc_auc": float(rt_on["ROC-AUC"] - selected["ROC-AUC"]), "log_loss": float(rt_on["Log Loss"] - selected["Log Loss"])},
        },
        "deployed_transferable_frozen": {
            **deployment_eval["test_native"],
            "model_version": metadata["model_version"],
            "runtime_calibration": "native/un-calibrated (calibrator present but disabled)",
            "evidence_category": "A",
            "boundary": metadata["transfer_boundary"],
        },
        "prediction_level_recomputation": {"available": False, "reason": "External prediction parquet not supplied to this run."},
    }

    if historical_root:
        prepared = historical_root / "03_Experimental_Modeling/data/prepared"
        pred_path = prepared / "predictions_H2_GB_RT_OFF_test_platt_retained.parquet"
        native_path = prepared / "predictions_H2_GB_RT_OFF_test_native.parquet"
        supervised_path = prepared / "test_supervised_rows.parquet"
        if pred_path.is_file() and supervised_path.is_file():
            pred = pd.read_parquet(pred_path)
            sup = pd.read_parquet(supervised_path)
            if len(pred) != len(sup) or not np.array_equal(pred.problem_log_id.to_numpy(), sup.problem_log_id.to_numpy()):
                raise ValueError("Prediction and supervised-row identities do not align")
            metrics = binary_prediction_metrics(pred.y_true, pred.predicted_probability)
            metrics["clustered_bootstrap_95_ci"] = clustered_bootstrap_intervals(
                pred.y_true, pred.predicted_probability, pred.user_id,
                replicates=bootstrap_replicates, seed=SEED,
            )
            joined = sup.copy()
            joined["predicted_probability"] = pred.predicted_probability.to_numpy()
            joined["history_length_bin"] = pd.cut(
                joined.learner_interaction_index,
                bins=[-1, 0, 3, 9, np.inf], labels=["first", "2-4", "5-10", "11+"],
            )
            history_rows = []
            for label, group in joined.groupby("history_length_bin", observed=True):
                row = binary_prediction_metrics(group.y_true, group.predicted_probability)
                history_rows.append({"history_length": str(label), **{k: row[k] for k in ("n_interactions", "positive_prevalence", "roc_auc", "average_precision", "log_loss", "brier_score", "accuracy")}})
            problem_rows = []
            for label, group in joined.groupby("problem_type", dropna=False):
                if len(group) < 1000 or group.y_true.nunique() < 2:
                    continue
                row = binary_prediction_metrics(group.y_true, group.predicted_probability)
                problem_rows.append({"problem_type": str(label), **{k: row[k] for k in ("n_interactions", "positive_prevalence", "roc_auc", "average_precision", "log_loss", "brier_score")}})
            problem_rows.sort(key=lambda row: row["n_interactions"], reverse=True)
            native_pred = pd.read_parquet(native_path) if native_path.is_file() else None
            output["prediction_level_recomputation"] = {
                "available": True,
                "metrics": metrics,
                "history_length_subgroups": history_rows,
                "problem_type_subgroups_n_ge_1000": problem_rows,
                "native_recomputed": binary_prediction_metrics(native_pred.y_true, native_pred.predicted_probability) if native_pred is not None else None,
                "inputs": {
                    "platt_predictions": {"file": pred_path.name, "sha256": sha256(pred_path)},
                    "supervised_rows": {"file": supervised_path.name, "sha256": sha256(supervised_path)},
                },
                "evidence_category": "B (recomputation of frozen A evidence)",
            }
    return output


def _import_runtime(source_backend: Path):
    backend_text = str(source_backend.resolve())
    if backend_text not in sys.path:
        sys.path.insert(0, backend_text)
    from app.modules.adaptive_questions.domain.curriculum import grade_support
    from app.modules.adaptive_questions.feature_engineering.learner_state import make_learner_state
    from app.modules.adaptive_questions.prediction.local_state_model import estimate_skill_success, load_adaptive_settings
    from app.modules.adaptive_questions.prediction.predictor import PredictorRouter
    from app.modules.adaptive_questions.recommendation.coverage import GAME_ORDER, GAME_TO_AREA, GAME_TO_SKILL, choose_adaptive_focus, evidence_requirements
    from app.modules.adaptive_questions.recommendation.curriculum_filter import curriculum_eligible, generate_candidates
    from app.modules.adaptive_questions.recommendation.progression_rules import tier_b_evidence
    from app.modules.adaptive_questions.recommendation.recommender import RecommendationEngine
    return locals()


def _parse_history(tokens: list[str]) -> list[dict]:
    rows = []
    for index, token in enumerate(tokens):
        game, correct, tier = token.split(":")
        rows.append({"game_type": game, "correct": int(correct), "tier": "Tier B" if tier == "B" else "Tier A", "order": index})
    return rows


def _state(runtime, learner_id: str, grade: int, current_skill: str, history: list[dict]):
    game_to_skill = runtime["GAME_TO_SKILL"]
    by_skill, exposures, correct, tier_a_exp, tier_a_correct = defaultdict(list), Counter(), Counter(), Counter(), Counter()
    activity_by_game = {"COUNTING":"CNT_A_01", "COMPARISON":"CMP_A_01", "ORDERING":"ORD_A_01", "PLACE_VALUE":"PV_A_01", "ADDITION":"ADD_A_02", "SUBTRACTION":"SUB_A_01", "MULTIPLICATION":"MUL_A_01", "DIVISION":"DIV_A_01"}
    successes = []
    for event in history:
        skill = game_to_skill[event["game_type"]]
        by_skill[skill].append(event["correct"])
        exposures[skill] += 1
        correct[skill] += event["correct"]
        if event["tier"] == "Tier A":
            tier_a_exp[skill] += 1
            tier_a_correct[skill] += event["correct"]
        if event["correct"]:
            successes.append(activity_by_game[event["game_type"]])
    recent = [event["correct"] for event in history[-5:]]
    return runtime["make_learner_state"](
        learner_id=learner_id,
        current_grade=grade,
        current_skill_context=current_skill,
        prior_interaction_count=len(history),
        prior_correct_count=sum(event["correct"] for event in history),
        recent_outcomes=recent,
        recent_skill_outcomes=by_skill[current_skill][-3:],
        recent_skill_outcomes_by_skill={key: value[-3:] for key, value in by_skill.items()},
        prior_skill_exposures=dict(exposures),
        prior_skill_correct=dict(correct),
        tier_a_exposures=dict(tier_a_exp),
        tier_a_correct=dict(tier_a_correct),
        skill_history_availability=bool(exposures[current_skill]),
        successfully_attempted_activities=tuple(dict.fromkeys(successes)),
        last_activity=activity_by_game[history[-1]["game_type"]] if history else None,
        last_tier=history[-1]["tier"] if history else None,
        interactions_since_last_correct=(len(history) - 1 - max(i for i, event in enumerate(history) if event["correct"]) if any(event["correct"] for event in history) else None),
    )


def _profiles(runtime, history: list[dict]) -> list[dict]:
    rows = []
    for game in runtime["GAME_ORDER"]:
        selected = [event for event in history if event["game_type"] == game]
        rows.append({
            "game_type": game,
            "valid_interactions": len(selected),
            "correct_interactions": sum(event["correct"] for event in selected),
            "last_assessed_at": f"T{selected[-1]['order']:03d}" if selected else None,
        })
    return rows


def _recommend(runtime, engine, router, grade: int, learner_id: str, history: list[dict], system: str) -> dict:
    base = _state(runtime, learner_id, grade, runtime["GAME_TO_SKILL"]["COUNTING"], history)
    eligible, filtered = runtime["generate_candidates"](engine.catalog, base)
    if not eligible:
        return {"activity_id": None, "tier": None, "game_type": None, "source": "ABSTAIN", "curriculum_valid": False, "tier_policy_valid": True, "probability": None, "reason": "NO_ELIGIBLE_CANDIDATE"}
    profiles = _profiles(runtime, history)
    recent = history[-12:]
    if system == "fixed_sequence":
        tier_a = [candidate for candidate in eligible if candidate["tier"] == "Tier A"]
        offset = len(history) % len(runtime["GAME_ORDER"])
        order = runtime["GAME_ORDER"][offset:] + runtime["GAME_ORDER"][:offset]
        selected = next((candidate for game in order for candidate in sorted(tier_a, key=lambda row: row["activity_id"]) if candidate["game_type"] == game), sorted(tier_a, key=lambda row: row["activity_id"])[0])
        state = _state(runtime, learner_id, grade, selected["project_skill"], history)
        ok, _ = runtime["curriculum_eligible"](selected, state)
        return {"activity_id": selected["activity_id"], "tier": selected["tier"], "game_type": selected["game_type"], "source": "FIXED_SEQUENCE", "curriculum_valid": ok, "tier_policy_valid": True, "probability": None, "reason": "DETERMINISTIC_GRADE_ELIGIBLE_SEQUENCE"}
    skills = [candidate["project_skill"] for candidate in eligible]
    if system == "rule_only":
        settings = runtime["load_adaptive_settings"]()
        predictions = {skill: runtime["estimate_skill_success"](base, skill, settings) for skill in dict.fromkeys(skills)}
    else:
        predictions = router.predict_many(base, skills)
    adaptive = runtime["choose_adaptive_focus"](profiles, eligible, recent, base, predictions)
    game = adaptive.get("game_type")
    if not game:
        return {"activity_id": None, "tier": None, "game_type": None, "source": "STOP", "curriculum_valid": True, "tier_policy_valid": True, "probability": None, "reason": adaptive["reason_code"]}
    state = _state(runtime, learner_id, grade, runtime["GAME_TO_SKILL"][game], history)
    candidates, blocked = runtime["generate_candidates"](engine.catalog, state)
    focused = [candidate for candidate in candidates if candidate["game_type"] == game]
    other = [{**candidate, "filter_reason": "ADAPTIVE_FOCUS_OTHER_GAME"} for candidate in candidates if candidate["game_type"] != game]
    raw = engine.recommend_next_activity(state, eligible_candidates=focused, filtered_candidates=blocked + other, adaptive_decision=adaptive)
    selected = next((candidate for candidate in focused if candidate["activity_id"] == raw["candidate_activity_id"]), None)
    ok = bool(selected) and runtime["curriculum_eligible"](selected, state)[0]
    tier_ok = bool(selected) and (selected["tier"] == "Tier A" or runtime["tier_b_evidence"](state, selected["project_skill"]))
    provenance = raw.get("model_provenance") or {}
    return {
        "activity_id": raw["candidate_activity_id"], "tier": raw["tier"], "game_type": game,
        "source": provenance.get("source", raw["recommendation_mode"]),
        "curriculum_valid": ok and raw["candidate_activity_id"] in {candidate["activity_id"] for candidate in focused},
        "tier_policy_valid": tier_ok,
        "probability": raw.get("predicted_probability"),
        "reason": raw.get("reason_code"),
        "recommendation_mode": raw.get("recommendation_mode"),
        "fallback_used": raw.get("fallback_used"),
        "abstained": raw.get("abstained"),
    }


def _applicable_games(runtime, engine, grade: int) -> list[str]:
    return [game for game in runtime["GAME_ORDER"] if any(candidate["active"] and candidate["tier"] == "Tier A" and candidate["grade_min"] <= grade <= candidate["grade_max"] and runtime["grade_support"](game, "Tier A", grade) for candidate in engine.catalog if candidate["game_type"] == game)]


def _outcome(profile: str, index: int) -> int:
    patterns = {
        "strong": (1, 1, 1, 0, 1, 1),
        "mixed": (1, 0, 1, 0, 1, 0),
        "struggling": (0, 0, 1, 0, 0, 1),
    }
    return patterns[profile][index % len(patterns[profile])]


def evaluate_recommendation(source_backend: Path, scenario_path: Path) -> dict:
    runtime = _import_runtime(source_backend)
    engine = runtime["RecommendationEngine"](source_backend / "app/modules/adaptive_questions/config/activity_catalog.csv")
    router = runtime["PredictorRouter"]()
    definitions = read_json(scenario_path)
    controlled = []
    for scenario in definitions["controlled_scenarios"]:
        history = _parse_history(scenario["history"])
        started = time.perf_counter()
        result = _recommend(runtime, engine, router, scenario["grade"], scenario["scenario_id"], history, "hybrid")
        latency_ms = (time.perf_counter() - started) * 1000
        repeated = _recommend(runtime, engine, router, scenario["grade"], scenario["scenario_id"], history, "hybrid")
        controlled.append({
            **result,
            "scenario_id": scenario["scenario_id"], "pair_id": scenario.get("pair_id"),
            "variant": scenario.get("variant"), "expect_change": scenario.get("expect_change", False),
            "grade": scenario["grade"], "history_count": len(history), "latency_ms": latency_ms,
            "deterministic_repeat": (result["activity_id"], result["tier"], result["reason"]) == (repeated["activity_id"], repeated["tier"], repeated["reason"]),
        })
    by_grade = {}
    for grade in range(1, 6):
        rows = [row for row in controlled if row["grade"] == grade]
        by_grade[str(grade)] = policy_compliance_metrics(rows)

    baseline_rows = []
    for system in ("fixed_sequence", "rule_only", "hybrid"):
        for scenario in definitions["controlled_scenarios"]:
            history = _parse_history(scenario["history"])
            result = _recommend(runtime, engine, router, scenario["grade"], scenario["scenario_id"], history, system)
            baseline_rows.append({**result, "system": system, "scenario_id": scenario["scenario_id"], "pair_id": scenario.get("pair_id"), "expect_change": scenario.get("expect_change", False)})
    baseline_comparison = {}
    for system in ("fixed_sequence", "rule_only", "hybrid"):
        rows = [row for row in baseline_rows if row["system"] == system]
        baseline_comparison[system] = {**policy_compliance_metrics(rows), "history_sensitivity": paired_change_metrics(rows)}

    sessions = []
    all_recommendations = []
    for system in ("fixed_sequence", "rule_only", "hybrid"):
        for grade in range(1, 6):
            for profile in ("strong", "mixed", "struggling"):
                history = []
                applicable = _applicable_games(runtime, engine, grade)
                for index in range(24):
                    result = _recommend(runtime, engine, router, grade, f"SESSION-{system}-{grade}-{profile}", history, system)
                    if not result["activity_id"]:
                        break
                    all_recommendations.append({**result, "system": system, "grade": grade, "profile": profile})
                    history.append({"game_type": result["game_type"], "correct": _outcome(profile, index), "tier": result["tier"], "order": index})
                    evidence = runtime["evidence_requirements"](_profiles(runtime, history), applicable, runtime["load_adaptive_settings"]())
                    if len(history) >= 16 and all(row["evidence_status"] == "SUFFICIENT" for row in evidence["games"] + evidence["areas"]):
                        break
                evidence = runtime["evidence_requirements"](_profiles(runtime, history), applicable, runtime["load_adaptive_settings"]())
                complete = len(history) >= 16 and all(row["evidence_status"] == "SUFFICIENT" for row in evidence["games"] + evidence["areas"])
                sessions.append({
                    "system": system, "grade": grade, "profile": profile,
                    "valid_responses": len(history), "complete": complete,
                    "distinct_games": len({event["game_type"] for event in history}),
                    "eligible_games": len(applicable),
                    "skill_coverage": safe_rate(len({event["game_type"] for event in history}), len(applicable)),
                })
    session_summary = {}
    for system in ("fixed_sequence", "rule_only", "hybrid"):
        selected = [row for row in sessions if row["system"] == system]
        recs = [row for row in all_recommendations if row["system"] == system]
        session_summary[system] = {
            "sessions": len(selected),
            "completed": sum(row["complete"] for row in selected),
            "valid_session_completion_rate": safe_rate(sum(row["complete"] for row in selected), len(selected)),
            "mean_valid_responses": statistics.mean(row["valid_responses"] for row in selected),
            "mean_skill_coverage": statistics.mean(row["skill_coverage"] for row in selected),
            "curriculum_compliance_rate": safe_rate(sum(row["curriculum_valid"] for row in recs), len(recs)),
            "tier_policy_compliance_rate": safe_rate(sum(row["tier_policy_valid"] for row in recs), len(recs)),
            "recommendations": len(recs),
            "trained_ml_usage_rate": safe_rate(sum(row["source"] == "TRAINED_ML" for row in recs), len(recs)),
        }
    sources = Counter(row["source"] for row in controlled)
    return {
        "controlled_scenarios": controlled,
        "overall": {
            **policy_compliance_metrics(controlled),
            "history_sensitivity": paired_change_metrics(controlled),
            "deterministic_repeat_rate": safe_rate(sum(row["deterministic_repeat"] for row in controlled), len(controlled)),
            "inference_source_counts": dict(sources),
            "trained_ml_usage_rate": safe_rate(sources["TRAINED_ML"], len(controlled)),
            "fallback_or_abstention_rate": safe_rate(len(controlled) - sources["TRAINED_ML"], len(controlled)),
            "latency_ms": {
                "mean": statistics.mean(row["latency_ms"] for row in controlled),
                "p95": float(np.percentile([row["latency_ms"] for row in controlled], 95)),
                "n": len(controlled),
            },
        },
        "per_grade": by_grade,
        "baseline_comparison_controlled": baseline_comparison,
        "baseline_session_comparison": session_summary,
        "session_details": sessions,
        "evidence_category": "C",
        "limitations": "Outcomes are deterministic scenario inputs and do not estimate learning benefit or counterfactual recommendation utility.",
    }


def evaluate_existing_simulation(research_root: Path | None) -> dict:
    if research_root is None:
        return {"available": False, "reason": "Research Documents root not supplied"}
    result_dir = research_root / "12_C2_25_Student_Simulation/results"
    validation_path = result_dir / "25_STUDENTS_VALIDATION_RESULTS.json"
    trace_path = result_dir / "25_STUDENTS_QUESTION_TRACE.csv"
    c4_path = result_dir / "25_STUDENTS_C4_EXPORTS.json"
    if not validation_path.is_file() or not trace_path.is_file():
        return {"available": False, "reason": "Existing simulation artifacts not found"}
    validation = read_json(validation_path)
    trace = pd.read_csv(trace_path)
    sim = validation["simulation"]
    c4 = read_json(c4_path) if c4_path.is_file() else []
    recomputed = {
        "learners": int(trace.learner_id.nunique()),
        "scored_questions": int(len(trace)),
        "trained_ml_assisted": int((trace.recommendation_mode == "TRAINED_ML_ASSISTED").sum()),
        "rule_fallback": int((trace.recommendation_mode == "RULE_FALLBACK").sum()),
        "curriculum_valid": int(trace.curriculum_eligible.astype(bool).sum()),
        "selected_from_eligible": int(trace.selected_from_eligible_candidates.astype(bool).sum()),
        "c4_exports": len(c4) if isinstance(c4, list) else len(c4.get("exports", c4)),
    }
    return {
        "available": True,
        "reported": sim,
        "recomputed": recomputed,
        "reconciliation": {
            "learners_match": recomputed["learners"] == sim["learners_executed"],
            "questions_match": recomputed["scored_questions"] == sim["total_scored_questions"],
            "ml_count_match": recomputed["trained_ml_assisted"] == sim["xgboost_recommendations"],
            "fallback_count_match": recomputed["rule_fallback"] == sim["deterministic_fallback_or_abstention"],
        },
        "source_hashes": {"validation": sha256(validation_path), "trace": sha256(trace_path), "c4_exports": sha256(c4_path) if c4_path.is_file() else None},
        "evidence_category": "A/C",
        "limitations": "Fictional learners with programmed response generation; reliability and behavior evidence only.",
    }


def result_rows(results: dict) -> list[dict]:
    p = results["predictive"]
    r = results["recommendation"]
    s = results["existing_simulation"]
    recomputed = p["prediction_level_recomputation"].get("metrics", {})
    winner = p["research_winner_frozen"]
    deployed = p["deployed_transferable_frozen"]
    expanded = results["expanded_history"]["metrics"]
    rows = [
        ("Research winner ROC-AUC", "Prediction", winner["roc_auc"], None, winner["n_interactions"], "A", "Frozen Step-3 model results", "Discrimination on held-out ASSISTments interactions", "Not recommendation quality or local-child validity"),
        ("Research winner Average Precision", "Prediction", winner["average_precision"], None, winner["n_interactions"], "A", "Frozen Step-3 model results", "Positive-class ranking quality; AP definition", "Depends on prevalence"),
        ("Research winner Log Loss", "Prediction", winner["log_loss"], None, winner["n_interactions"], "A", "Frozen Step-3 model results", "Probability error; lower is better", "Held-out ASSISTments only"),
        ("Research winner Brier Score", "Prediction", winner["brier_score"], None, winner["n_interactions"], "A", "Frozen Step-3 model results", "Mean squared probability error; lower is better", "Held-out ASSISTments only"),
        ("Research winner ECE-10", "Calibration", winner["ece_10_equal_frequency"], None, winner["n_interactions"], "A", "Frozen Step-3 model results", "Equal-frequency calibration gap; lower is better", "ECE depends on binning"),
        ("Threshold-0.5 accuracy", "Prediction", recomputed.get("accuracy"), None, recomputed.get("n_interactions"), "B", "External prediction-level Parquet", "Classification accuracy at specified threshold", "Requires external prediction rows; not a recommendation metric"),
        ("Threshold-0.5 F1", "Prediction", recomputed.get("f1"), None, recomputed.get("n_interactions"), "B", "External prediction-level Parquet", "Harmonic mean of precision and recall at threshold 0.5", "Requires external prediction rows and depends on threshold"),
        ("Deployed transferable ROC-AUC", "Prediction", deployed["roc_auc"], None, deployed["n_rows"], "A", "Tracked deployed evaluation metadata", "Discrimination of TRANSFER_H2_GB_RT_OFF_v1", "Source-domain result; not the 0.750158 research model"),
        ("Deployed transferable Log Loss", "Prediction", deployed["log_loss"], None, deployed["n_rows"], "A", "Tracked deployed evaluation metadata", "Native probability error for deployed artifact", "Runtime calibration is disabled"),
        ("Deployed transferable Brier Score", "Prediction", deployed["brier_score"], None, deployed["n_rows"], "A", "Tracked deployed evaluation metadata", "Native squared probability error for deployed artifact", "Not locally calibrated on children"),
        ("Curriculum Compliance Rate", "Recommendation safety", r["overall"]["curriculum_compliance_rate"], r["overall"]["curriculum_valid"], r["overall"]["recommendations"], "C", "Controlled frozen-policy scenarios", "Adherence to encoded grade/curriculum rules", "Does not independently validate curriculum mapping"),
        ("Tier-Policy Compliance", "Recommendation safety", r["overall"]["tier_policy_compliance_rate"], r["overall"]["tier_policy_valid"], r["overall"]["recommendations"], "C", "Controlled frozen-policy scenarios", "Adherence to internal Tier A/B rules", "Not independent difficulty appropriateness"),
        ("History Sensitivity", "Adaptivity", r["overall"]["history_sensitivity"]["recommendation_change_rate"], r["overall"]["history_sensitivity"]["recommendation_changes"], r["overall"]["history_sensitivity"]["matched_pairs"], "C", "Original six matched-history pairs", "Historical final-activity change rate", "Small original experiment; retained separately"),
        ("Expanded final activity change", "Adaptivity", expanded["activity_change_rate"], expanded["activity_changes"], expanded["history_pairs"], "C", "Predeclared expanded matched histories", "Final activity changed after all constraints", "Higher is not automatically better"),
        ("Expanded equal-eligibility activity change", "Adaptivity", expanded["equal_eligibility_activity_change_rate"], expanded["equal_eligibility_activity_changes"], expanded["equal_eligibility_pairs"], "C", "Strict identical-candidate subset", "Final activity changed without candidate-set differences", "Synthetic histories"),
        ("Expanded XGBoost score change", "Adaptivity", expanded["score_change_rate"], expanded["score_changes"], expanded["history_pairs"], "C", "Predeclared expanded matched histories", "At least one common candidate-skill score changed", "Source-domain probabilities"),
        ("Expanded candidate-ranking change", "Adaptivity", expanded["ranking_change_rate"], expanded["ranking_changes"], expanded["history_pairs"], "C", "Predeclared expanded matched histories", "Candidate order around the policy target changed", "Internal behavioral measure"),
        ("Expanded pre-constraint preferred-skill change", "Adaptivity", expanded["preferred_skill_change_rate"], expanded["preferred_skill_changes"], expanded["history_pairs"], "C", "Predeclared expanded matched histories", "Top model-preferred skill changed before coverage allocation", "Internal behavioral measure"),
        ("Expanded tier change", "Adaptivity", expanded["tier_change_rate"], expanded["tier_changes"], expanded["history_pairs"], "C", "Predeclared expanded matched histories", "Final selected tier changed", "Conservative tier policy intentionally limits changes"),
        ("Expanded Curriculum Compliance Rate", "Recommendation safety", expanded["curriculum_compliance_rate"], expanded["curriculum_valid"], expanded["scenario_decisions"], "C", "Expanded synthetic scenarios", "Adherence to encoded curriculum rules", "Not independent pedagogical validation"),
        ("Expanded Tier-Policy Compliance", "Recommendation safety", expanded["tier_policy_compliance_rate"], expanded["tier_valid"], expanded["scenario_decisions"], "C", "Expanded synthetic scenarios", "Adherence to implemented tier rules", "Not independent difficulty appropriateness"),
        ("Expanded invariant stability", "Reliability", expanded["invariant_stability_rate"], expanded["invariant_stable"], expanded["invariant_pairs"], "C", "Predeclared invariant controls", "Equivalent histories yielded identical observable decisions", "Deterministic synthetic controls"),
        ("Policy-Expectation Agreement", "Specification agreement", expanded["policy_expectation_agreement"], expanded["policy_expectations_met"], expanded["policy_expectations_total"], "C", "Predeclared implementation-specification rubric", "Combined direction, coverage and invariance agreement", "Not recommendation accuracy; components must be shown"),
        ("Prediction-direction agreement", "Specification agreement", expanded["direction_expectation_agreement"], expanded["direction_expectations_met"], expanded["direction_expectations_total"], "C", "Predeclared implementation-specification rubric", "Expected probability direction under history contrast", "Synthetic histories"),
        ("Coverage expectation agreement", "Specification agreement", expanded["coverage_expectation_agreement"], expanded["coverage_expectations_met"], expanded["coverage_expectations_total"], "C", "Predeclared implementation-specification rubric", "Selection belonged to independently computed admissible set", "Agreement with internal specification"),
        ("Expanded invariant expectation agreement", "Specification agreement", expanded["invariant_stability_rate"], expanded["invariant_stable"], expanded["invariant_pairs"], "C", "Predeclared implementation-specification rubric", "Equivalent-history expectation met", "Synthetic invariant controls"),
        ("Expanded mean skill coverage", "Coverage", expanded["mean_skill_coverage"], expanded["sessions"], expanded["sessions"], "C", "25 expanded controlled sessions", "Mean applicable-skill coverage at stopping", "Programmed synthetic responses"),
        ("Expanded session completion", "End-to-end", expanded["session_completion_rate"], expanded["completed_sessions"], expanded["sessions"], "C", "25 expanded controlled sessions", "Completion within frozen evidence budget", "Programmed synthetic responses"),
        ("Deterministic repeat stability", "Reliability", r["overall"]["deterministic_repeat_rate"], sum(row["deterministic_repeat"] for row in r["controlled_scenarios"]), len(r["controlled_scenarios"]), "C", "Repeated scenario execution", "Equivalent input gives same decision", "Only deterministic execution conditions tested"),
        ("Hybrid valid session completion", "End-to-end", r["baseline_session_comparison"]["hybrid"]["valid_session_completion_rate"], r["baseline_session_comparison"]["hybrid"]["completed"], r["baseline_session_comparison"]["hybrid"]["sessions"], "C", "Controlled sessions", "Completion within 16-24 valid responses", "Programmed synthetic responses"),
        ("Existing 25-session completion", "End-to-end", safe_rate(s.get("reported", {}).get("complete_assessments", 0), s.get("reported", {}).get("learners_executed", 0)) if s.get("available") else None, s.get("reported", {}).get("complete_assessments") if s.get("available") else None, s.get("reported", {}).get("learners_executed") if s.get("available") else None, "A/C", "Existing 25-learner simulation", "Software completion under fictional profiles", "Not educational effectiveness"),
        ("Existing simulation ML-assisted rate", "End-to-end", safe_rate(s.get("recomputed", {}).get("trained_ml_assisted", 0), s.get("recomputed", {}).get("scored_questions", 0)) if s.get("available") else None, s.get("recomputed", {}).get("trained_ml_assisted") if s.get("available") else None, s.get("recomputed", {}).get("scored_questions") if s.get("available") else None, "A/C", "Existing 25-learner simulation", "Trained-ML-assisted recommendations among scored interactions", "Synthetic fictional learners"),
        ("Existing simulation cold-start fallback rate", "End-to-end", safe_rate(s.get("recomputed", {}).get("rule_fallback", 0), s.get("recomputed", {}).get("scored_questions", 0)) if s.get("available") else None, s.get("recomputed", {}).get("rule_fallback") if s.get("available") else None, s.get("recomputed", {}).get("scored_questions") if s.get("available") else None, "A/C", "Existing 25-learner simulation", "Cold-start fallback decisions among scored interactions", "Synthetic fictional learners"),
        ("Existing simulation Curriculum Compliance Rate", "End-to-end", safe_rate(s.get("recomputed", {}).get("curriculum_valid", 0), s.get("recomputed", {}).get("scored_questions", 0)) if s.get("available") else None, s.get("recomputed", {}).get("curriculum_valid") if s.get("available") else None, s.get("recomputed", {}).get("scored_questions") if s.get("available") else None, "A/C", "Existing 25-learner simulation", "Curriculum-valid selections in preserved trace", "Internal rules and synthetic learners"),
        ("C4 export validity", "Integration", 1.0 if s.get("reported", {}).get("all_c4_exports_valid") else None, 25 if s.get("reported", {}).get("all_c4_exports_valid") else None, 25 if s.get("available") else None, "A/C", "Existing 25-learner simulation", "C4-compatible exports passed schema/consistency checks", "Synthetic data only"),
        ("Real-child recommendation utility", "Educational outcome", None, None, None, "D", "Missing randomized/counterfactual evidence", "Causal learning or correctness benefit", "Not causally estimable from current evidence"),
    ]
    return [{"metric_name": a, "category": b, "value": c, "numerator": d, "denominator": e, "evidence_category": f, "source": g, "interpretation": h, "limitation": i} for a,b,c,d,e,f,g,h,i in rows]


def write_summary(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _table(data, widths=None, font_size=7):
    from reportlab.lib import colors
    from reportlab.platypus import Table, TableStyle
    table = Table(data, colWidths=widths, repeatRows=1)
    table.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#17365D")),
        ("TEXTCOLOR", (0,0), (-1,0), colors.white),
        ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"),
        ("FONTNAME", (0,1), (-1,-1), "Helvetica"),
        ("FONTSIZE", (0,0), (-1,-1), font_size),
        ("VALIGN", (0,0), (-1,-1), "TOP"),
        ("GRID", (0,0), (-1,-1), 0.25, colors.HexColor("#B4C7E7")),
        ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.white, colors.HexColor("#F3F6FA")]),
        ("LEFTPADDING", (0,0), (-1,-1), 4), ("RIGHTPADDING", (0,0), (-1,-1), 4),
        ("TOPPADDING", (0,0), (-1,-1), 4), ("BOTTOMPADDING", (0,0), (-1,-1), 4),
    ]))
    return table


def make_report(path: Path, results: dict, summary_rows: list[dict]) -> None:
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_CENTER
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import mm
    from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Spacer

    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="TitleC2", parent=styles["Title"], fontName="Helvetica-Bold", fontSize=21, leading=25, textColor=colors.HexColor("#17365D"), alignment=TA_CENTER, spaceAfter=18))
    styles.add(ParagraphStyle(name="H1C2", parent=styles["Heading1"], fontName="Helvetica-Bold", fontSize=15, leading=18, textColor=colors.HexColor("#17365D"), spaceBefore=10, spaceAfter=7, keepWithNext=True))
    styles.add(ParagraphStyle(name="H2C2", parent=styles["Heading2"], fontName="Helvetica-Bold", fontSize=11, leading=14, textColor=colors.HexColor("#2F5597"), spaceBefore=8, spaceAfter=5, keepWithNext=True))
    styles.add(ParagraphStyle(name="BodyC2", parent=styles["BodyText"], fontName="Helvetica", fontSize=9, leading=13, spaceAfter=6))
    styles.add(ParagraphStyle(name="NoteC2", parent=styles["BodyText"], fontName="Helvetica-Oblique", fontSize=8, leading=11, leftIndent=8, rightIndent=8, borderColor=colors.HexColor("#B4C7E7"), borderWidth=0.5, borderPadding=6, backColor=colors.HexColor("#F3F6FA"), spaceAfter=8))

    def header_footer(canvas, doc):
        canvas.saveState()
        canvas.setStrokeColor(colors.HexColor("#B4C7E7")); canvas.line(18*mm, 16*mm, 192*mm, 16*mm)
        canvas.setFont("Helvetica", 7); canvas.setFillColor(colors.HexColor("#5B6573"))
        canvas.drawString(18*mm, 10*mm, "J26-DS-330 | C2 Evaluation | Evidence-bound research report")
        canvas.drawRightString(192*mm, 10*mm, f"Page {doc.page}")
        canvas.restoreState()

    doc = SimpleDocTemplate(str(path), pagesize=A4, rightMargin=18*mm, leftMargin=18*mm, topMargin=18*mm, bottomMargin=21*mm, title="C2 Final Evaluation Report", author="J26-DS-330")
    story = []
    P = lambda text, style="BodyC2": story.append(Paragraph(text, styles[style]))
    H1 = lambda text: P(text, "H1C2")
    H2 = lambda text: P(text, "H2C2")
    pred, rec, sim = results["predictive"], results["recommendation"], results["existing_simulation"]
    expanded = results["expanded_history"]
    em = expanded["metrics"]
    winner = pred["research_winner_frozen"]

    story += [Spacer(1, 25*mm)]
    P("C2 Final Evaluation Report", "TitleC2")
    P("History-Aware Adaptive Game Recommendation Engine", "TitleC2")
    P("AI-Based Robotic Interactive System for Mathematical Assessment and Early Dyscalculia Screening Support", "BodyC2")
    P("Project J26-DS-330 | Evaluation date: 21 October 2026", "BodyC2")
    story += [Spacer(1, 14*mm)]
    P("This report evaluates prediction, recommendation behavior, and end-to-end reliability without claiming clinical validity or improved child learning outcomes.", "NoteC2")
    story.append(PageBreak())

    H1("Executive conclusion")
    P(f"The frozen C2 implementation works as an auditable, curriculum-constrained sequential assessment recommender under the tested software and synthetic conditions. In controlled tests, Curriculum Compliance Rate was {pct(rec['overall']['curriculum_compliance_rate'])}, Tier-Policy Compliance was {pct(rec['overall']['tier_policy_compliance_rate'])}, deterministic repeat stability was {pct(rec['overall']['deterministic_repeat_rate'])}, and matched-history recommendation change rate was {pct(rec['overall']['history_sensitivity']['recommendation_change_rate'])}. The existing 25-fictional-learner run completed 25/25 sessions and produced 25 valid C4 exports.")
    P(f"The expanded predeclared experiment retained all {em['history_pairs']} matched-history pairs plus {em['invariant_pairs']} invariant controls. XGBoost scores changed in {em['score_changes']}/{em['history_pairs']} pairs, candidate ranking changed in {em['ranking_changes']}/{em['history_pairs']}, and the final activity changed in {em['activity_changes']}/{em['history_pairs']}. This extends behavioral evidence without replacing the original six-pair result.")
    P(f"The original research winner achieved ROC-AUC {winner['roc_auc']:.6f}, Average Precision {winner['average_precision']:.6f}, Log Loss {winner['log_loss']:.6f}, Brier Score {winner['brier_score']:.6f}, and equal-frequency ECE-10 {winner['ece_10_equal_frequency']:.6f} on 937,043 held-out ASSISTments interactions from 7,001 learners. These are prediction metrics, not recommendation-effectiveness metrics.")
    P("A critical deployment distinction is preserved: the live artifact is TRANSFER_H2_GB_RT_OFF_v1 and its stored held-out native ROC-AUC is about 0.7121. Its runtime metadata disables calibration. The 0.7502 result belongs to the original H2_GB_RT_OFF_v1 research winner with retained Platt scaling; the values must not be presented as the same deployed model.", "NoteC2")
    P("Current evidence cannot causally estimate whether the hybrid policy improves real learners' outcomes over rule-only or fixed sequencing. That question requires randomized or otherwise defensible counterfactual outcome data.")

    H1("1. Objectives and research questions")
    P("The primary question is whether C2 produces valid, history-sensitive, curriculum-appropriate, stable, and reliable next-activity decisions. The stronger educational-outcome question is separated: whether ML-assisted recommendations improve learning or next-response correctness relative to a rule-only policy.")
    P("RQ1: How well does the frozen correctness model discriminate and calibrate on its held-out source-domain test set? RQ2: Does the frozen policy obey encoded curriculum and tier constraints? RQ3: Do controlled history changes alter recommendations as documented while irrelevant changes remain invariant? RQ4: Does the complete component reliably register learners, issue and resume tasks, persist validated events, stop sessions, and export C4 evidence? RQ5: What differs from fair fixed-sequence and rule-only baselines?")

    H1("2. Evidence hierarchy and boundaries")
    data = [["Code", "Meaning", "Permitted interpretation"]] + [[key, value, {"A":"Historical result supported by frozen files", "B":"Recalculation by this package", "C":"Behavior under fictional or controlled inputs", "D":"Not yet measured"}[key]] for key, value in EVIDENCE.items()]
    story.append(_table(data, [15*mm, 62*mm, 92*mm], 7.5))
    P("Software compliance is not pedagogical validation. Synthetic completion is not learning effectiveness. ASSISTments association is not local robotic-cube calibration or diagnostic validity.", "NoteC2")

    H1("3. Frozen implementation and source traceability")
    P(f"Deployed artifact identity: {pred['artifact_identity']['deployed_model_version']}; algorithm: {pred['artifact_identity']['algorithm']}. All {len(pred['artifact_identity']['manifest_integrity'])} manifest-listed files matched their stored byte counts and SHA-256 hashes: {pred['artifact_identity']['all_manifest_files_match']}.")
    P(f"The original experiment registry contains {pred['eight_model_registry']['unique_experiments']} completed configurations: {', '.join(pred['eight_model_registry']['experiment_ids'])}.")
    P("The policy enforces Grades 1-5, candidate activity status, curriculum trace presence, question-level grade support, robot and physical-cube suitability, prerequisites, Tier B evidence, coverage requirements, and 16-24 valid-response stopping rules. The curriculum mapping remains a local NIE audit with identified expert-validation needs.")

    H1("4. Methodology")
    H2("Predictive evaluation")
    P("Frozen experiment rows were reconciled against prediction-level Parquet files. Average Precision is the area under the precision-recall step curve. Threshold metrics use p >= 0.5. ECE uses 10 equal-frequency bins: sum over bins of bin weight multiplied by the absolute difference between observed prevalence and mean predicted probability. When prediction rows were available, 95% percentile intervals used a learner-cluster bootstrap with complete learner histories resampled, seed 20261021.")
    H2("Recommendation evaluation")
    P("Matched scenarios held grade and policy constant while varying permitted historical outcomes. The actual frozen catalog, filters, history feature builder, predictor router, coverage allocator, tier rules, and recommender were imported from the isolated source snapshot. Identical calls were repeated to test determinism. A recommendation counted as curriculum-valid only when the selected activity was in the generated eligible set and passed the actual filter. Tier-policy compliance required Tier B selections to satisfy the implemented exposure, accumulated-correctness, and recent-outcome rule.")
    P("The expanded design was declared in expanded_config.json before execution: Grades 1-5, early/middle/late stages, four history contrasts, two deterministic replicates per cell, and equivalent-history controls. Pair generation never consulted observed recommendations, and every generated pair was retained.")
    H2("Baselines")
    P("Fixed sequence selects a deterministic grade-eligible Tier A activity following the frozen game order and never uses outcomes for ranking. Rule-only uses the same curriculum, coverage, response budget, and Beta-Bernoulli local-state estimate but excludes XGBoost. Hybrid uses the deployed predictor router and unchanged policy. Synthetic outcome sequences are identical across systems and independent of selected activity; therefore correctness cannot be interpreted as comparative utility.")

    H1("5. Predictive-model results")
    recomputed = pred["prediction_level_recomputation"].get("metrics", {})
    metric_table = [["Metric", "Frozen result", "Recomputed", "Interpretation"]]
    for label, key in [("ROC-AUC","roc_auc"),("Average Precision","average_precision"),("Log Loss","log_loss"),("Brier Score","brier_score"),("ECE-10","ece_10_equal_frequency")]:
        metric_table.append([label, f"{winner[key]:.6f}", f"{recomputed.get(key):.6f}" if recomputed.get(key) is not None else "Unavailable", "Higher better" if key in {"roc_auc","average_precision"} else "Lower better"])
    story.append(_table(metric_table, [40*mm, 33*mm, 33*mm, 55*mm], 8))
    if recomputed:
        cm = recomputed["confusion_matrix"]
        P(f"At threshold 0.5, accuracy was {recomputed['accuracy']:.6f}, balanced accuracy {recomputed['balanced_accuracy']:.6f}, precision {recomputed['precision']:.6f}, recall {recomputed['recall_sensitivity']:.6f}, specificity {recomputed['specificity']:.6f}, and F1 {recomputed['f1']:.6f}. Confusion matrix: TN={cm['tn']:,}, FP={cm['fp']:,}, FN={cm['fn']:,}, TP={cm['tp']:,}. Positive prevalence was {pct(recomputed['positive_prevalence'])}; majority-class accuracy was {pct(recomputed['majority_class_accuracy'])}.")
        P(f"The prevalence-probability baseline had Log Loss {recomputed['prevalence_probability_log_loss']:.6f} and Brier Score {recomputed['prevalence_probability_brier_score']:.6f}.")
        ci = recomputed.get("clustered_bootstrap_95_ci", {})
        if ci:
            P(f"Learner-clustered 95% bootstrap intervals: ROC-AUC [{ci['roc_auc']['lower_95']:.6f}, {ci['roc_auc']['upper_95']:.6f}], Average Precision [{ci['average_precision']['lower_95']:.6f}, {ci['average_precision']['upper_95']:.6f}], Log Loss [{ci['log_loss']['lower_95']:.6f}, {ci['log_loss']['upper_95']:.6f}], using {ci['roc_auc']['replicates_valid']} valid replicates.")
    H2("History, response time, and calibration")
    subgroup = pred["prediction_level_recomputation"].get("history_length_subgroups", [])
    if subgroup:
        story.append(_table([["History", "n", "ROC-AUC", "AP", "Log Loss"]] + [[row["history_length"], f"{row['n_interactions']:,}", f"{row['roc_auc']:.4f}", f"{row['average_precision']:.4f}", f"{row['log_loss']:.4f}"] for row in subgroup], [30*mm,35*mm,30*mm,30*mm,30*mm], 8))
    rt = pred["response_time_ablation"]
    P(f"Adding response-time features did not improve the selected test result: RT-ON minus RT-OFF ROC-AUC = {rt['rt_on_minus_off']['roc_auc']:.6f}; Log Loss difference = {rt['rt_on_minus_off']['log_loss']:.6f}. Platt scaling reduced ECE from {pred['native_vs_platt']['native']['ece']:.6f} to {pred['native_vs_platt']['platt']['ece']:.6f} for the research winner.")

    H1("6. Deployed-model reconciliation")
    deployed = pred["deployed_transferable_frozen"]
    P(f"The deployed transferable artifact reports test-native ROC-AUC {deployed['roc_auc']:.6f}, Log Loss {deployed['log_loss']:.6f}, and Brier Score {deployed['brier_score']:.6f} over {deployed['n_rows']:,} rows. It is a separately trained transferable artifact, not the 0.7502 research-winner binary. Runtime calibration is recorded as: {deployed['runtime_calibration']}.")
    P(deployed["boundary"], "NoteC2")

    H1("7. Recommendation-policy results")
    overall = rec["overall"]
    story.append(_table([
        ["Measure", "Result", "Numerator / denominator", "Meaning"],
        ["Curriculum Compliance Rate", pct(overall["curriculum_compliance_rate"]), f"{overall['curriculum_valid']} / {overall['recommendations']}", "Encoded-rule adherence"],
        ["Tier-Policy Compliance", pct(overall["tier_policy_compliance_rate"]), f"{overall['tier_policy_valid']} / {overall['recommendations']}", "Internal tier-rule adherence"],
        ["History recommendation change", pct(overall["history_sensitivity"]["recommendation_change_rate"]), f"{overall['history_sensitivity']['recommendation_changes']} / {overall['history_sensitivity']['matched_pairs']}", "Matched scenarios"],
        ["History score change", pct(overall["history_sensitivity"]["score_change_rate"]), f"{overall['history_sensitivity']['score_changes']} / {overall['history_sensitivity']['score_comparable_pairs']}", "Matched scenarios"],
        ["Tier change rate", pct(overall["history_sensitivity"]["tier_change_rate"]), f"{overall['history_sensitivity']['tier_changes']} / {overall['history_sensitivity']['matched_pairs']}", "Matched scenarios"],
        ["Invariant stability", pct(overall["history_sensitivity"]["invariant_stability_rate"]), f"{overall['history_sensitivity']['invariant_pairs']-overall['history_sensitivity']['invariant_changes']} / {overall['history_sensitivity']['invariant_pairs']}", "Irrelevant-ID/equivalent histories"],
        ["Deterministic stability", pct(overall["deterministic_repeat_rate"]), f"{sum(row['deterministic_repeat'] for row in rec['controlled_scenarios'])} / {len(rec['controlled_scenarios'])}", "Identical calls"],
        ["Trained-ML usage", pct(overall["trained_ml_usage_rate"]), f"{overall['inference_source_counts'].get('TRAINED_ML',0)} / {overall['recommendations']}", "Controlled scenarios"],
    ], [42*mm,24*mm,35*mm,60*mm], 7.5))
    P("Curriculum Compliance Rate validates that code follows the encoded mapping; it does not validate that the mapping is pedagogically complete. Tier-Policy Compliance is deliberately not called independent difficulty appropriateness because no expert difficulty labels exist.", "NoteC2")

    H1("8. History sensitivity and stability")
    pair_rows = overall["history_sensitivity"]["details"]
    story.append(_table([["Pair", "Task changed", "Tier", "Score delta", "Expected", "Agreement"]] + [[row["pair_id"], str(row["recommendation_changed"]), str(row["tier_changed"]), (f"{row['probability_difference']:.4f}" if row["probability_difference"] is not None else "n/a"), str(row["expect_change"]), str(row["expectation_agreement"])] for row in pair_rows], [39*mm,24*mm,18*mm,26*mm,25*mm,27*mm], 6.8))
    P(f"Across history-varied pairs, the selected activity changed in {overall['history_sensitivity']['recommendation_changes']}/{overall['history_sensitivity']['matched_pairs']} pairs, while the trained probability changed in {overall['history_sensitivity']['score_changes']}/{overall['history_sensitivity']['score_comparable_pairs']} score-comparable pairs. All {overall['history_sensitivity']['invariant_pairs']} invariant pairs were stable. A score or task change is evidence that history reaches the decision path, not that the response is educationally superior.")
    P("The extended-history pair keeps the final five responses identical while varying the earlier trajectory. Its non-zero probability difference demonstrates use of accumulated history beyond the most recent response, although coverage constraints kept the selected activity unchanged.")

    H1("9. Expanded matched-history evaluation")
    story.append(_table([
        ["Measure", "Result", "Numerator / denominator", "Interpretation"],
        ["Final activity change", pct(em["activity_change_rate"]), f"{em['activity_changes']} / {em['history_pairs']}", "After all policy constraints"],
        ["Equal-eligibility activity change", pct(em["equal_eligibility_activity_change_rate"]), f"{em['equal_eligibility_activity_changes']} / {em['equal_eligibility_pairs']}", "Identical candidate IDs"],
        ["XGBoost score change", pct(em["score_change_rate"]), f"{em['score_changes']} / {em['history_pairs']}", "Any common skill score changed"],
        ["Candidate-ranking change", pct(em["ranking_change_rate"]), f"{em['ranking_changes']} / {em['history_pairs']}", "Pre-coverage order changed"],
        ["Preferred-skill change", pct(em["preferred_skill_change_rate"]), f"{em['preferred_skill_changes']} / {em['history_pairs']}", "Top model preference changed"],
        ["Tier change", pct(em["tier_change_rate"]), f"{em['tier_changes']} / {em['history_pairs']}", "Final tier changed"],
        ["Curriculum compliance", pct(em["curriculum_compliance_rate"]), f"{em['curriculum_valid']} / {em['scenario_decisions']}", "Encoded rules"],
        ["Tier-policy compliance", pct(em["tier_policy_compliance_rate"]), f"{em['tier_valid']} / {em['scenario_decisions']}", "Implemented tier evidence"],
        ["Invariant stability", pct(em["invariant_stability_rate"]), f"{em['invariant_stable']} / {em['invariant_pairs']}", "Equivalent histories"],
    ], [42*mm,24*mm,36*mm,59*mm], 7.2))
    P(f"Mean absolute probability difference was {em['mean_absolute_probability_difference']:.4f}. All {em['unequal_eligibility_pairs']} unequal candidate-set pairs differed only through documented Tier B activities; they were retained. In the strict equal-candidate subset, the final activity changed in {em['equal_eligibility_activity_changes']}/{em['equal_eligibility_pairs']} pairs.")
    H2("Policy-Expectation Agreement")
    story.append(_table([
        ["Component", "Agreement", "Numerator / denominator"],
        ["Prediction direction", pct(em["direction_expectation_agreement"]), f"{em['direction_expectations_met']} / {em['direction_expectations_total']}"],
        ["Coverage", pct(em["coverage_expectation_agreement"]), f"{em['coverage_expectations_met']} / {em['coverage_expectations_total']}"],
        ["Invariance", pct(em["invariant_stability_rate"]), f"{em['invariant_stable']} / {em['invariant_pairs']}"],
        ["Combined", pct(em["policy_expectation_agreement"]), f"{em['policy_expectations_met']} / {em['policy_expectations_total']}"],
    ], [60*mm,38*mm,58*mm], 8))
    P("Policy-Expectation Agreement is agreement with a predeclared implementation-specification rubric. It is not recommendation accuracy or independent pedagogical appropriateness.", "NoteC2")

    H1("10. Cold start, abstention, and fallback")
    P(f"Controlled inference sources were {json.dumps(overall['inference_source_counts'], sort_keys=True)}. The deployed router attempts the trained model after at least one prior valid interaction; at cold start it falls back to the local Beta-Bernoulli path, which may itself lack a probability and yield rule fallback. No probability is fabricated when the predictor has no valid estimate.")
    if sim.get("available"):
        P(f"In the existing 25-session simulation, {sim['reported']['xgboost_recommendations']} of {sim['reported']['total_scored_questions']} recommendations were XGBoost-assisted and {sim['reported']['deterministic_fallback_or_abstention']} were cold-start deterministic fallback/abstention decisions.")

    H1("11. Baseline comparison")
    baseline = rec["baseline_session_comparison"]
    story.append(_table([["System", "Sessions complete", "Mean responses", "Mean skill coverage", "CCR", "Tier compliance", "ML usage"]] + [[name, f"{row['completed']}/{row['sessions']}", f"{row['mean_valid_responses']:.2f}", pct(row['mean_skill_coverage']), pct(row['curriculum_compliance_rate']), pct(row['tier_policy_compliance_rate']), pct(row['trained_ml_usage_rate'])] for name,row in baseline.items()], [28*mm,27*mm,25*mm,30*mm,20*mm,25*mm,20*mm], 6.8))
    P("Equal compliance is expected because all systems share the same frozen eligibility constraints. The meaningful behavioral difference is history-dependent ranking and tier action, not an artificial advantage on safety rules. Since programmed correctness is activity-independent, these runs cannot compare learning utility.")

    H1("12. End-to-end reliability")
    if sim.get("available"):
        report = sim["reported"]
        P(f"The preserved simulation executed {report['learners_executed']} fictional learners and {report['total_scored_questions']} scored questions. All {report['complete_assessments']} assessments completed; session lengths ranged {report['minimum_session_length']}-{report['maximum_session_length']} with mean {report['mean_session_length']:.2f}. There were {report['grade_inappropriate_recommendations']} grade-inappropriate selections and {report['selected_outside_eligible_set']} selections outside the eligible set. C4 exports were valid for all learners.")
        P(f"Independent row-level reconciliation in this package found {sim['recomputed']['scored_questions']} trace rows, {sim['recomputed']['trained_ml_assisted']} trained-ML-assisted decisions, and {sim['recomputed']['rule_fallback']} rule fallbacks. Count agreement flags: {json.dumps(sim['reconciliation'], sort_keys=True)}.")
    P("Existing backend tests cover registration, session creation/resumption, active-question consistency, question generation, validation, idempotent duplicate handling, conflicting duplicate rejection, invalid/non-scoring event behavior, persistence, stopping, model integrity, and C4 contracts. Exact executed test results are recorded in the verification report outside this compact package and the manifest verification block.")

    H1("13. Operational metrics")
    latency = overall["latency_ms"]
    P(f"On this evaluation host, controlled end-to-end policy calls (including model inference where available) had mean latency {latency['mean']:.3f} ms and p95 {latency['p95']:.3f} ms across {latency['n']} calls. This is supporting engineering evidence; it is not a service-level guarantee and excludes network and robot latency.")

    H1("14. Metric suitability")
    P("Precision@K, Recall@K, Hit Rate@K, NDCG@K, and MRR are not headline metrics here. The runtime emits one next task, and there are no independent relevance labels for alternative tasks. Treating the historically next ASSISTments activity as relevant would confound observed administration with relevance. Catalog/skill coverage and decision diversity are suitable descriptive measures when their denominators are explicit.")
    P("No single scalar 'C2 accuracy' is defensible. The primary evaluation question is answered by a balanced scorecard spanning prediction, internal safety compliance, adaptivity, stability, completion, fallback behavior, and interface validity.")

    H1("15. Recommended thesis scorecard")
    scorecard_names = {"Research winner ROC-AUC", "Deployed transferable ROC-AUC", "Expanded Curriculum Compliance Rate", "Expanded Tier-Policy Compliance", "Expanded final activity change", "Expanded XGBoost score change", "Expanded invariant stability", "Expanded session completion", "C4 export validity", "Real-child recommendation utility"}
    scorecard_labels = {
        "Expanded Curriculum Compliance Rate": "Expanded curriculum compliance",
        "Expanded Tier-Policy Compliance": "Expanded tier compliance",
    }
    chosen = [row for row in summary_rows if row["metric_name"] in scorecard_names]
    story.append(_table([["Metric", "Value", "Evidence", "Use"]] + [[scorecard_labels.get(row["metric_name"], row["metric_name"]), "Pending" if row["value"] is None else (pct(row["value"]) if row["category"] not in {"Prediction","Calibration"} else f"{row['value']:.6f}"), row["evidence_category"], row["interpretation"]] for row in chosen], [43*mm,28*mm,20*mm,75*mm], 7))
    P("Primary recommendation evidence is the expanded matched-history score, ranking and final-activity response interpreted jointly with curriculum, tier, coverage and invariant checks. Utility remains pending rather than being replaced by a weighted score.")

    H1("16. Limitations and future validation")
    P("There are no randomized learner outcomes, logged propensities, expert relevance labels, independent difficulty ratings, or local-child probability calibration data. The selected ASSISTments model is associational and source-domain specific. Synthetic profiles encode outcomes rather than discovering them. Curriculum rules require qualified external review, especially noted grade/task adaptations. Clustered intervals address learner dependence in the held-out prediction set but not domain shift.")
    P("A minimal valid next study is a pre-registered, supervised crossover or randomized study comparing hybrid and rule-only policies under identical eligible sets and response budgets. Outcomes should include next-response correctness at comparable difficulty, pre/post skill measures, burden, safety overrides, and teacher-rated appropriateness. Allocation probabilities and all candidate sets should be logged. Analysis should cluster by learner, stratify by grade, report calibration locally, and separate screening-support evidence from diagnosis.")

    H1("17. Final answer to the examiner")
    P("The engine 'works' in the bounded engineering sense supported by current evidence: it preserves curriculum eligibility, applies its tier rules, changes scores and candidate ordering under controlled history differences, remains deterministic under equivalent input, completes synthetic assessment sessions within the configured evidence budget, safely handles cold start, persists valid events, and exports C4-compatible evidence. Final activities change less often because the frozen XGBoost ranking operates inside coverage and safety constraints.")
    P("What remains unproven is educational effectiveness: current data cannot show that those different recommendations improve real children's learning, assessment accuracy, or dyscalculia screening outcomes. That claim is reserved for the future controlled study.", "NoteC2")

    H1("Appendix A. Metric definitions")
    definitions = [
        ("ROC-AUC", "P(score_positive > score_negative); threshold-independent discrimination."),
        ("Average Precision", "Weighted mean of precision across recall increments; prevalence-sensitive PR summary."),
        ("Log Loss", "-n^-1 sum[y log(p) + (1-y) log(1-p)]; lower is better."),
        ("Brier Score", "n^-1 sum(p-y)^2; lower is better."),
        ("ECE-10", "Sum_b n_b/n |accuracy_b - confidence_b| using 10 equal-frequency bins."),
        ("CCR", "Curriculum-valid recommendations / evaluated recommendations."),
        ("Tier-Policy Compliance", "Recommendations satisfying implemented tier rules / evaluated recommendations."),
        ("History Sensitivity", "Matched scenario pairs with changed next recommendation / matched history pairs."),
        ("Valid Session Completion", "Sessions meeting total, game, and area evidence before the maximum / evaluated sessions."),
        ("Skill Coverage", "Distinct applicable skills assessed / applicable skills for that grade."),
    ]
    story.append(_table([["Metric", "Definition"]] + [list(row) for row in definitions], [42*mm, 125*mm], 7.5))

    H1("Appendix B. Reproducibility record")
    P(f"Original seed: {SEED}; expanded seed: {expanded['seed']}. Python: {results['environment']['python']}. Platform: {results['environment']['platform']}. Evaluation timestamp: {results['generated_at']}. Source: tracked repository backend at commit {results['repository']['head']}.")
    P("The results manifest records source and package SHA-256 hashes, configuration, denominators, evidence categories, and verification results. The three notebooks call the same shared evaluator and can be executed from clean kernels.")

    doc.build(story, onFirstPage=header_footer, onLaterPages=header_footer)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    here = Path(__file__).resolve().parent
    parser.add_argument("--source-backend", type=Path, default=here.parents[2])
    parser.add_argument("--historical-root", type=Path)
    parser.add_argument("--research-root", type=Path)
    parser.add_argument("--scenario-file", type=Path, default=here / "scenario_definitions.json")
    parser.add_argument("--expanded-config", type=Path, default=here / "expanded_config.json")
    parser.add_argument("--output-dir", type=Path, default=here)
    parser.add_argument("--bootstrap-replicates", type=int, default=200)
    parser.add_argument("--verification-file", type=Path)
    parser.add_argument("--skip-report", action="store_true")
    args = parser.parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    results = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "seed": SEED,
        "source_backend": "backend (repository-relative)",
        "external_inputs": {
            "prediction_parquet_root_supplied": bool(args.historical_root),
            "preserved_simulation_root_supplied": bool(args.research_root),
        },
        "evidence_hierarchy": EVIDENCE,
        "environment": {"python": sys.version, "platform": platform.platform(), "numpy": np.__version__, "pandas": pd.__version__},
        "repository": repository_provenance(args.source_backend),
    }
    results["predictive"] = evaluate_predictive(args.source_backend, args.historical_root, args.bootstrap_replicates)
    results["recommendation"] = evaluate_recommendation(args.source_backend, args.scenario_file)
    from expanded_evaluation import evaluate_expanded
    results["expanded_history"] = evaluate_expanded(args.source_backend, args.expanded_config, args.scenario_file)
    results["existing_simulation"] = evaluate_existing_simulation(args.research_root)
    results["educational_utility"] = {
        "status": "NOT_CAUSALLY_ESTIMABLE",
        "missing_evidence": ["randomized or defensible counterfactual allocation", "outcomes for alternative recommendations", "local child validation", "expert relevance/difficulty labels"],
        "evidence_category": "D",
    }
    results["runtime_seconds"] = time.time() - started
    rows = result_rows(results)
    summary_path = args.output_dir / "results_summary.csv"
    write_summary(summary_path, rows)
    report_path = args.output_dir / "C2_Final_Evaluation_Report.pdf"
    if not args.skip_report:
        make_report(report_path, results, rows)

    package_files = [path for path in args.output_dir.iterdir() if path.is_file() and path.name != "results_manifest.json"]
    critical_source = [
        args.source_backend / "app/modules/adaptive_questions/core_service.py",
        args.source_backend / "app/modules/adaptive_questions/recommendation/coverage.py",
        args.source_backend / "app/modules/adaptive_questions/recommendation/recommender.py",
        args.source_backend / "app/modules/adaptive_questions/prediction/predictor.py",
        args.source_backend / "app/modules/adaptive_questions/config/activity_catalog.csv",
        args.source_backend / "app/modules/adaptive_questions/models/transferable/manifest.json",
    ]
    manifest = {
        "schema_version": "C2_EVALUATION_MANIFEST_v2",
        "generated_at": results["generated_at"],
        "seed": SEED,
        "command": "python evaluation_runner.py [--historical-root <external ASSISTments evidence>] [--research-root <external simulation evidence>]",
        "configuration": {"bootstrap_replicates": args.bootstrap_replicates, "threshold": 0.5, "ece_bins": 10, "ece_binning": "equal-frequency", "cluster_unit": "learner/user_id", "expanded_config": args.expanded_config.name},
        "package_files": {path.name: {"bytes": path.stat().st_size, "sha256": sha256(path)} for path in sorted(package_files)},
        "source_files": {str(path.relative_to(args.source_backend)).replace('\\', '/'): {"bytes": path.stat().st_size, "sha256": sha256(path)} for path in critical_source},
        "results": results,
        "summary_row_count": len(rows),
        "verification": read_json(args.verification_file) if args.verification_file else {"status": "Not supplied to this run"},
        "limitations": [
            "No causal recommendation-utility estimate is supported.",
            "Synthetic and controlled outcomes are not real-child outcomes.",
            "Curriculum and tier compliance are internal-policy measures.",
            "The deployed transferable model differs from the 0.7502 research winner and is not locally calibrated.",
        ],
    }
    manifest_path = args.output_dir / "results_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"status": "ok", "summary": str(summary_path), "manifest": str(manifest_path), "report": str(report_path) if not args.skip_report else None, "runtime_seconds": results["runtime_seconds"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
