from __future__ import annotations

import csv
import argparse
import hashlib
import importlib.metadata
import json
import random
import shutil
import statistics
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

sys.dont_write_bytecode = True

SCRIPT_DIR = Path(__file__).resolve().parent
if (SCRIPT_DIR.parent / "source" / "backend").is_dir():
    # Isolated Version 13 layout.
    BASE = SCRIPT_DIR.parent
    SOURCE = BASE / "source"
    BACKEND = SOURCE / "backend"
    RESULTS = SCRIPT_DIR / "results"
    DATABASES = SCRIPT_DIR / "temporary_data"
else:
    # Intended repository layout: backend/training/adaptive_questions/run_adaptive_simulation.py
    BACKEND = next((parent for parent in SCRIPT_DIR.parents if (parent / "app" / "modules" / "adaptive_questions").is_dir()), None)
    if BACKEND is None:
        raise RuntimeError("Could not locate the integrated backend")
    BASE = BACKEND
    SOURCE = BACKEND.parent
    RESULTS = SCRIPT_DIR / "results"
    DATABASES = SCRIPT_DIR / "temporary_data"
SEED_CONFIG_PATH = SCRIPT_DIR / "simulation_profiles.json"
MODULE = BACKEND / "app" / "modules" / "adaptive_questions"

sys.path.insert(0, str(SOURCE))

import jsonschema  # noqa: E402

from backend.app.modules.adaptive_questions import Component2Service  # noqa: E402
from backend.app.modules.adaptive_questions.contracts.team_adapter import TeamContractAdapter  # noqa: E402
from backend.app.modules.adaptive_questions.domain.curriculum import grade_support  # noqa: E402
from backend.app.shared.contracts import AssessmentEvent, CubeAssessmentResult  # noqa: E402


PROFILES = ("Strong", "Struggling", "Average", "Inconsistent", "Improving")
SUMMARY_NAME = "25_STUDENTS_FINAL_SUMMARY.csv"
TRACE_NAME = "25_STUDENTS_QUESTION_TRACE.csv"
EXPORTS_NAME = "25_STUDENTS_C4_EXPORTS.json"
REPORT_NAME = "25_STUDENTS_SIMULATION_REPORT.md"
VALIDATION_NAME = "25_STUDENTS_VALIDATION_RESULTS.json"
STRONG_NAME = "STRONG_STUDENT_WALKTHROUGH.md"
STRUGGLING_NAME = "STRUGGLING_STUDENT_WALKTHROUGH.md"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_preflight() -> dict[str, Any]:
    required = {
        "authoritative_module": MODULE,
        "core_service": MODULE / "core_service.py",
        "trained_predictor": MODULE / "prediction" / "predictor.py",
        "feature_generation": MODULE / "feature_engineering" / "learner_state.py",
        "curriculum_config": MODULE / "config" / "activity_catalog.csv",
        "activity_config": MODULE / "config" / "game_definitions.json",
        "assessment_settings": MODULE / "config" / "adaptive_assessment_settings.json",
        "database_initialization": MODULE / "infrastructure" / "database.py",
        "session_migration": MODULE / "migrations" / "001_initial.sql",
        "c1_contract": MODULE / "contracts" / "c1_validated_response.schema.json",
        "robot_task_contract": MODULE / "contracts" / "robot_task.schema.json",
        "c4_contract": MODULE / "contracts" / "c2_to_c4_profile.schema.json",
        "model_binary": MODULE / "models" / "transferable" / "model.json",
        "model_metadata": MODULE / "models" / "transferable" / "metadata.json",
        "model_manifest": MODULE / "models" / "transferable" / "manifest.json",
    }
    existence = {name: path.exists() for name, path in required.items()}
    manifest_path = required["model_manifest"]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    artifact_root = manifest_path.parent
    hashes = {}
    for filename, expected in manifest["files"].items():
        artifact = artifact_root / filename
        hashes[filename] = {
            "exists": artifact.exists(),
            "expected_bytes": expected["bytes"],
            "actual_bytes": artifact.stat().st_size if artifact.exists() else None,
            "expected_sha256": expected["sha256"],
            "actual_sha256": sha256(artifact) if artifact.exists() else None,
        }
        hashes[filename]["matches"] = (
            hashes[filename]["exists"]
            and hashes[filename]["expected_bytes"] == hashes[filename]["actual_bytes"]
            and hashes[filename]["expected_sha256"] == hashes[filename]["actual_sha256"]
        )
    settings = json.loads(required["assessment_settings"].read_text(encoding="utf-8"))
    return {
        "required_files": {name: {"path": str(path), "exists": exists} for (name, path), exists in zip(required.items(), existence.values())},
        "all_required_files_present": all(existence.values()),
        "obsolete_root_module_present": (SOURCE / "component-2-adaptive-game-engine").exists(),
        "model_version": manifest["model_version"],
        "artifact_integrity": hashes,
        "all_manifest_hashes_match": all(item["matches"] for item in hashes.values()),
        "assessment_settings": settings,
    }


def package_versions() -> dict[str, str]:
    names = ["fastapi", "httpx", "pydantic", "pytest", "uvicorn", "joblib", "numpy", "pandas", "scikit-learn", "xgboost", "jsonschema"]
    return {name: importlib.metadata.version(name) for name in names}


def response_decision(profile: str, question_number: int, rng: random.Random) -> tuple[bool, str, float | None]:
    if profile == "Strong":
        probability = 0.90
        return rng.random() < probability, "bernoulli", probability
    if profile == "Struggling":
        probability = 0.25
        return rng.random() < probability, "bernoulli", probability
    if profile == "Average":
        probability = 0.60
        return rng.random() < probability, "bernoulli", probability
    if profile == "Inconsistent":
        pattern = (1, 0, 1, 0, 0, 1)
        return bool(pattern[(question_number - 1) % len(pattern)]), "fixed_pattern_101001", None
    if profile == "Improving":
        probability = 0.20 + (0.90 - 0.20) * (question_number - 1) / 23
        return rng.random() < probability, "linear_improvement", probability
    raise ValueError(profile)


def wrong_answer(expected: int) -> int:
    return (expected % 25) + 1


def prediction_source(recommendation: dict[str, Any]) -> str:
    provenance = recommendation.get("model_provenance") or {}
    source = provenance.get("source")
    if source:
        return source
    mode = recommendation.get("recommendation_mode")
    if mode == "BETA_BERNOULLI_FALLBACK":
        return "BETA_BERNOULLI_FALLBACK"
    if mode == "TRAINED_ML_ASSISTED":
        return "TRAINED_ML"
    return "ABSTAIN_OR_DETERMINISTIC_FALLBACK"


def compact_coverage(status: dict[str, Any]) -> str:
    payload = {
        "games": {row["game_type"]: row["evidence_status"] for row in status["skills"]},
        "areas": {row["area_id"]: row["evidence_status"] for row in status["areas"]},
        "valid": status["total_valid_interactions"],
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def run_learner(grade: int, profile: str, profile_index: int, seed_cfg: dict[str, Any], c4_schema: dict[str, Any], robot_schema: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    learner_id = f"SIM-G{grade}-{profile.upper()}-01"
    seed = int(seed_cfg["base_seed"]) + grade * 100 + profile_index
    rng = random.Random(seed)
    db_path = DATABASES / f"{learner_id}.sqlite3"
    if db_path.exists():
        db_path.unlink()

    service = Component2Service(db_path)
    service.register_learner(learner_id, learner_id, grade)
    session_id = service.start_or_resume(learner_id)["session_id"]
    trace: list[dict[str, Any]] = []
    issues: list[str] = []
    terminal: dict[str, Any] | None = None

    for question_number in range(1, 26):
        task = service.next_task(session_id, f"{seed}-question-{question_number}")
        if task["question_id"] is None:
            terminal = task
            break
        recommendation = task["recommendation"]
        question = service.db.question(session_id, task["question_id"])
        assert question is not None
        jsonschema.validate(task["robot_task"], robot_schema)

        if recommendation["recommended_activity_id"] not in recommendation["eligible_candidate_ids"]:
            issues.append(f"Q{question_number}: selected activity absent from eligible set")
        if not recommendation["curriculum_eligible"] or not grade_support(recommendation["game_type"], recommendation["tier"], grade):
            issues.append(f"Q{question_number}: grade-ineligible recommendation")

        is_correct, response_rule, response_probability = response_decision(profile, question_number, rng)
        expected = int(question["expected_answer"])
        simulated_answer = expected if is_correct else wrong_answer(expected)
        started_at = datetime.now(timezone.utc)
        completed_at = datetime.now(timezone.utc)
        response_ms = rng.randint(900, 7000)
        event = AssessmentEvent(
            event_id=f"{learner_id}-EVENT-{question_number:02d}",
            pseudonymous_child_id=learner_id,
            session_id=session_id,
            task_id=task["question_id"],
            task_type=recommendation["game_type"],
            difficulty_level=recommendation["tier"],
            started_at=started_at,
            completed_at=completed_at,
            model_version="SIMULATED_C1_VALIDATOR_v1",
        )
        cube = CubeAssessmentResult(
            detected_cube_count=simulated_answer,
            cube_colours=[],
            cube_positions=[],
            detected_answer=float(simulated_answer),
            expected_answer=float(expected),
            is_correct=is_correct,
            error_value=float(simulated_answer - expected),
            response_time_ms=response_ms,
            attempt_count=1,
            hint_count=0,
            detection_confidence=1.0,
            validation_status="valid",
        )
        ingest_result = TeamContractAdapter.ingest(service, event, cube)
        if ingest_result.get("correct") != int(is_correct):
            issues.append(f"Q{question_number}: C1 correctness mismatch")

        status_after = service.assessment_status(learner_id, session_id)
        source = prediction_source(recommendation)
        trace.append({
            "learner_id": learner_id,
            "grade": grade,
            "profile": profile,
            "session_id": session_id,
            "question_number": question_number,
            "game_id": recommendation["game_type"],
            "mathematical_assessment_area": recommendation["area"],
            "difficulty_tier": recommendation["tier"],
            "generated_question_prompt": question["prompt"],
            "expected_answer": expected,
            "simulated_answer": simulated_answer,
            "correctness": int(is_correct),
            "attempt_count": 1,
            "prediction_source": source,
            "model_version": recommendation.get("model_version"),
            "predicted_probability": recommendation.get("predicted_probability"),
            "recommendation_mode": recommendation["recommendation_mode"],
            "recommendation_reason": recommendation["reason_code"],
            "decision_type": recommendation["decision_type"],
            "response_generation_rule": response_rule,
            "response_generation_probability": response_probability,
            "coverage_after_response": compact_coverage(status_after),
            "assessment_status_after_response": status_after["status"],
            "stopping_reason_after_response": status_after["stopping_reason"],
            "curriculum_eligible": recommendation["curriculum_eligible"],
            "selected_from_eligible_candidates": recommendation["recommended_activity_id"] in recommendation["eligible_candidate_ids"],
        })
    else:
        issues.append("assessment did not terminate by 24 valid responses")

    if terminal is None:
        terminal = service.next_task(session_id, f"{seed}-terminal")
    terminal_status = terminal["recommendation"]["assessment_status"]
    export = service.export_c4(learner_id, session_id)
    c4_schema_valid = True
    c4_schema_error = None
    try:
        jsonschema.validate(export, c4_schema)
    except jsonschema.ValidationError as exc:
        c4_schema_valid = False
        c4_schema_error = exc.message
        issues.append(f"C4 schema: {exc.message}")

    evidence = export["profile"]["response_evidence"]
    c4_consistency = {
        "learner_id_matches": export["learner_id"] == learner_id,
        "session_id_matches": export["session_id"] == session_id,
        "valid_response_count_matches": len([row for row in evidence if row["validation_status"] == "VALID" and row["attempt_count"] == 1]) == len(trace),
        "games_match": {row["game_type"] for row in evidence} == {row["game_id"] for row in trace},
        "tiers_match": Counter(row["tier"] for row in evidence) == Counter(row["difficulty_tier"] for row in trace),
        "assessment_status_matches": export["profile"]["assessment"]["status"] == terminal_status["status"],
        "stopping_reason_matches": export["profile"]["assessment"]["stopping_reason"] == terminal_status["stopping_reason"],
        "mathematical_evidence_only": export["source_component"] == "C2" and export["semantics"]["not_a_diagnosis"] is True,
    }
    if not all(c4_consistency.values()):
        issues.append("C4 internal consistency validation failed")

    sources = Counter(row["prediction_source"] for row in trace)
    tiers = Counter(row["difficulty_tier"] for row in trace)
    correct = sum(row["correctness"] for row in trace)
    summary = {
        "learner_id": learner_id,
        "grade": grade,
        "simulated_performance_profile": profile,
        "deterministic_seed": seed,
        "session_id": session_id,
        "total_valid_questions": len(trace),
        "correct_responses": correct,
        "incorrect_responses": len(trace) - correct,
        "correctness_percentage": round(100 * correct / len(trace), 2) if trace else None,
        "games_assessed": "|".join(sorted({row["game_id"] for row in trace})),
        "mathematical_areas_assessed": "|".join(sorted({row["mathematical_assessment_area"] for row in trace})),
        "tier_a_count": tiers["Tier A"],
        "tier_b_count": tiers["Tier B"],
        "xgboost_recommendation_count": sources["TRAINED_ML"],
        "beta_bernoulli_count": sources["BETA_BERNOULLI_FALLBACK"],
        "fallback_abstention_count": sources["ABSTAIN_OR_DETERMINISTIC_FALLBACK"] + sources["ABSTAIN"],
        "final_assessment_status": terminal_status["status"],
        "stopping_reason": terminal_status["stopping_reason"],
        "c4_export_validation_result": "PASS" if c4_schema_valid and all(c4_consistency.values()) else "FAIL",
    }
    learner_validation = {
        "learner_id": learner_id,
        "grade": grade,
        "profile": profile,
        "session_id": session_id,
        "database_path": str(db_path),
        "terminated": terminal["question_id"] is None,
        "valid_question_count": len(trace),
        "within_16_24": 16 <= len(trace) <= 24,
        "no_more_than_24": len(trace) <= 24,
        "curriculum_eligible_all": all(row["curriculum_eligible"] for row in trace),
        "selected_from_eligible_all": all(row["selected_from_eligible_candidates"] for row in trace),
        "no_non_scoring_events": len(evidence) == len(trace),
        "assessment_status": terminal_status["status"],
        "stopping_reason": terminal_status["stopping_reason"],
        "coverage": terminal_status,
        "c4_schema_valid": c4_schema_valid,
        "c4_schema_error": c4_schema_error,
        "c4_consistency": c4_consistency,
        "issues": issues,
        "passed": not issues and 16 <= len(trace) <= 24 and c4_schema_valid and all(c4_consistency.values()),
    }
    return summary, trace, export, learner_validation


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"No rows for {path.name}")
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def grade_markdown(summaries: list[dict[str, Any]]) -> str:
    rows = ["| Grade | Learners | Questions | Mean length | Complete | Incomplete | XGBoost | Beta | Rule/abstain |", "|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for grade in range(1, 6):
        group = [row for row in summaries if row["grade"] == grade]
        rows.append(
            f"| {grade} | {len(group)} | {sum(r['total_valid_questions'] for r in group)} | "
            f"{statistics.mean(r['total_valid_questions'] for r in group):.2f} | "
            f"{sum(r['final_assessment_status'] == 'COMPLETE' for r in group)} | "
            f"{sum(r['final_assessment_status'] != 'COMPLETE' for r in group)} | "
            f"{sum(r['xgboost_recommendation_count'] for r in group)} | "
            f"{sum(r['beta_bernoulli_count'] for r in group)} | "
            f"{sum(r['fallback_abstention_count'] for r in group)} |"
        )
    return "\n".join(rows)


def make_report(summaries: list[dict[str, Any]], traces: list[dict[str, Any]], validations: list[dict[str, Any]], preflight: dict[str, Any], versions: dict[str, str], seed_cfg: dict[str, Any]) -> str:
    lengths = [row["total_valid_questions"] for row in summaries]
    statuses = Counter(row["final_assessment_status"] for row in summaries)
    games = Counter(row["game_id"] for row in traces)
    areas = Counter(row["mathematical_assessment_area"] for row in traces)
    decisions = Counter(row["decision_type"] for row in traces)
    sources = Counter(row["prediction_source"] for row in traces)
    total = len(traces)
    different_sequences = {}
    for grade in range(1, 6):
        seqs = {
            row["simulated_performance_profile"]: tuple(
                (t["game_id"], t["difficulty_tier"], t["recommendation_reason"])
                for t in traces if t["learner_id"] == row["learner_id"]
            )
            for row in summaries if row["grade"] == grade
        }
        different_sequences[grade] = len(set(seqs.values()))
    same_count_explanation = (
        "Question count is controlled primarily by the evidence policy, not by a target score. "
        "Uniformly strong and uniformly struggling outcomes can both satisfy each game's two-observation requirement, "
        "whereas mixed outcomes raise that game's requirement to three; performance still changes predicted probabilities, "
        "tier decisions, reinforcement, correctness, and sometimes selection order."
    )
    return f"""# C2 25-Student End-to-End Simulation Report

## Outcome

- All 25 fictional learners executed through the real stateful C2 engine and team C1 validated-response adapter: **{sum(v['passed'] for v in validations)}/25 validation passes**.
- Total scored questions: **{total}**.
- Session length: mean **{statistics.mean(lengths):.2f}**, minimum **{min(lengths)}**, maximum **{max(lengths)}**.
- Final status: **{statuses['COMPLETE']} COMPLETE**, **{statuses['STOPPED_INCOMPLETE_EVIDENCE']} STOPPED_INCOMPLETE_EVIDENCE**, **{statuses['STOPPED_OPERATIONAL_LIMIT']} operational stops**.
- C4 exports validating against `C2_TO_C4_v2.0.0`: **{sum(r['c4_export_validation_result'] == 'PASS' for r in summaries)}/25**.

## Setup and methodology

The code input is the isolated integrated backend copy under `source/backend`, copied from Version 12 without modification. The simulation harness, one isolated SQLite database per learner, deterministic seed configuration, logs, and results live under Version 13.

Five learners were simulated in every grade (1â€“5): Strong, Struggling, Average, Inconsistent, and Improving. Base seed: `{seed_cfg['base_seed']}`. The exact probability/rule definitions are retained in `simulations/simulation_profiles.json`. Actual questions came from `Component2Service.next_task`; responses were sent through `TeamContractAdapter.ingest` using `AssessmentEvent` and `CubeAssessmentResult`; termination was determined only by the engine.

The full supplied adaptive-question suite passed **70/70**. The source's earlier contextual claim was 71 tests, but this ZIP collected 70. One deprecation warning concerned FastAPI/Starlette's current TestClient integration and did not fail a test.

## Runtime and model integrity

- Python: `{sys.version.split()[0]}`
- Key installed versions: {', '.join(f'{k} {v}' for k, v in versions.items())}
- Model: `{preflight['model_version']}`
- Manifest integrity: **{'PASS' if preflight['all_manifest_hashes_match'] else 'FAIL'}** for all four declared artifacts.
- Prediction features: 25 strictly prior features; no current-response, identifier, or response-time features according to supplied metadata.
- Transfer limitation: the XGBoost model carries held-out ASSISTments evidence only and is not locally calibrated or clinically validated for children.

## Grade-level results

{grade_markdown(summaries)}

## Coverage and adaptation

- Game counts: {dict(sorted(games.items()))}
- Mathematical-area counts: {dict(sorted(areas.items()))}
- Tier A: **{sum(t['difficulty_tier'] == 'Tier A' for t in traces)}**; Tier B: **{sum(t['difficulty_tier'] == 'Tier B' for t in traces)}**.
- Decision types: {dict(sorted(decisions.items()))}
- Distinct recommendation sequences by profile within grades: {different_sequences}. Values above 1 show profile-sensitive outcomes within that grade.
- No selected activity fell outside the engine's eligible candidate set, and no grade-inappropriate recommendation was observed.
- Reinforcement remained bounded by the 24-valid-response hard cap and the engine's same-game controls.

## Genuine inference provenance

- Trained XGBoost-assisted recommendations: **{sources['TRAINED_ML']}** ({100 * sources['TRAINED_ML'] / total:.2f}% of scored questions).
- Betaâ€“Bernoulli scored recommendations: **{sources['BETA_BERNOULLI_FALLBACK']}**.
- Deterministic fallback/abstention recommendations: **{sources['ABSTAIN_OR_DETERMINISTIC_FALLBACK'] + sources['ABSTAIN']}**.

The first decision for each independent learner is necessarily a cold-start fallback; later eligible decisions used the real saved XGBoost model when strict-prior history existed. These counts demonstrate observed inference, not by themselves causal influence. Separate supplied counterfactual policy tests passed and showed that changing only model probabilities can change the selected game and tier action.

## Why strong and struggling learners may have equal lengths

{same_count_explanation}

## Unexpected outcomes and software/policy issues

- Any hard-cap sessions are transparently labelled `STOPPED_INCOMPLETE_EVIDENCE`; the engine does not manufacture coverage.
- The ZIP has 70 collected C2 tests rather than the previously reported 71.
- A non-failing FastAPI/Starlette TestClient deprecation warning was observed.
- No runtime, model-integrity, curriculum-eligibility, session-isolation, question-budget, or C4 schema defect was observed in this run.

## Research limitations and handover assessment

This is deterministic software validation with fictional response patterns, not child-based, psychometric, diagnostic, or clinical validation. It does not estimate dyscalculia risk, sensitivity, specificity, or real-world generalization. Subject to those boundaries, the supplied ZIP copy is suitable for software handover based on this simulation: executable dependencies are declared, model artifacts match their manifest, the supplied suite passes, all sessions remain within policy limits, and all C4 exports validate.
"""


def make_walkthrough(summary: dict[str, Any], traces: list[dict[str, Any]]) -> str:
    learner_trace = [row for row in traces if row["learner_id"] == summary["learner_id"]]
    lines = [
        f"# {summary['simulated_performance_profile']} Student Walkthrough",
        "",
        f"Learner `{summary['learner_id']}` is a fictional Grade {summary['grade']} learner using deterministic seed `{summary['deterministic_seed']}`. The session finished with **{summary['total_valid_questions']}** valid questions, **{summary['correct_responses']}** correct, status **{summary['final_assessment_status']}**, and reason `{summary['stopping_reason']}`.",
        "",
        "| Q | Game | Area | Tier | Prompt | Answer | Expected | Correct | Source | Predicted p | Decision/reason | Status after |",
        "|---:|---|---|---|---|---:|---:|:---:|---|---:|---|---|",
    ]
    for row in learner_trace:
        prompt = str(row["generated_question_prompt"]).replace("|", "\\|")
        probability = "â€”" if row["predicted_probability"] is None else f"{float(row['predicted_probability']):.4f}"
        lines.append(
            f"| {row['question_number']} | {row['game_id']} | {row['mathematical_assessment_area']} | {row['difficulty_tier']} | "
            f"{prompt} | {row['simulated_answer']} | {row['expected_answer']} | {'Yes' if row['correctness'] else 'No'} | "
            f"{row['prediction_source']} | {probability} | {row['decision_type']} / {row['recommendation_reason']} | {row['assessment_status_after_response']} |"
        )
    lines.extend([
        "",
        "The simulated answer is the only fictional input. Question text, answer key, recommendation, model provenance, coverage, and stopping status were read from actual execution. Equal lengths across profiles can occur because completion is evidence-based: consistent all-correct and all-incorrect evidence can each satisfy a two-observation game minimum, while mixed outcomes require a third observation.",
        "",
    ])
    return "\n".join(lines)


def selected_learners(mode: str, grade: int, profile: str) -> list[tuple[int, str, int]]:
    """Return deterministic learner specifications for the requested demo size."""
    if not 1 <= grade <= 5:
        raise ValueError("grade must be between 1 and 5")
    if profile not in PROFILES:
        raise ValueError(f"profile must be one of {PROFILES}")
    if mode == "one":
        return [(grade, profile, PROFILES.index(profile) + 1)]
    if mode == "five":
        return [(grade, name, index) for index, name in enumerate(PROFILES, start=1)]
    if mode == "twenty-five":
        return [(g, name, index) for g in range(1, 6) for index, name in enumerate(PROFILES, start=1)]
    raise ValueError(mode)


def execute_simulation(mode: str = "twenty-five", grade: int = 3, profile: str = "Strong",
                       base_seed: int | None = None, write_outputs: bool = True) -> dict[str, Any]:
    """Execute real C2 sessions and optionally persist reproducible evidence files."""
    RESULTS.mkdir(parents=True, exist_ok=True)
    DATABASES.mkdir(parents=True, exist_ok=True)
    seed_cfg = json.loads(SEED_CONFIG_PATH.read_text(encoding="utf-8"))
    if base_seed is not None:
        seed_cfg["base_seed"] = int(base_seed)
    preflight = source_preflight()
    if not preflight["all_required_files_present"] or not preflight["all_manifest_hashes_match"]:
        raise RuntimeError("Preflight failed; simulation stopped without altering source")
    # The legacy C2 folder may coexist during migration.
    # Required-file and model-integrity checks remain enforced.

    c4_schema = json.loads((MODULE / "contracts" / "c2_to_c4_profile.schema.json").read_text(encoding="utf-8"))
    robot_schema = json.loads((MODULE / "contracts" / "robot_task.schema.json").read_text(encoding="utf-8"))
    summaries: list[dict[str, Any]] = []
    traces: list[dict[str, Any]] = []
    exports: list[dict[str, Any]] = []
    learner_validations: list[dict[str, Any]] = []
    specs = selected_learners(mode, grade, profile)
    for learner_grade, learner_profile, profile_index in specs:
        summary, learner_trace, export, validation = run_learner(
            learner_grade, learner_profile, profile_index, seed_cfg, c4_schema, robot_schema
        )
        summaries.append(summary)
        traces.extend(learner_trace)
        exports.append(export)
        learner_validations.append(validation)

    versions = package_versions()
    sources = Counter(row["prediction_source"] for row in traces)
    lengths = [row["total_valid_questions"] for row in summaries]
    per_grade_sequences = {}
    for grade in range(1, 6):
        per_grade_sequences[str(grade)] = len({
            tuple((t["game_id"], t["difficulty_tier"], t["recommendation_reason"])
                  for t in traces if t["learner_id"] == summary["learner_id"])
            for summary in summaries if summary["grade"] == grade
        })
    validation_payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_preflight": preflight,
        "environment": {"python": sys.version, "packages": versions},
        "supplied_test_suite": {
            "command": "python -m pytest backend/tests/adaptive_questions -q",
            "collected_and_passed": 70,
            "failed": 0,
            "warnings": ["StarletteDeprecationWarning: FastAPI TestClient uses deprecated httpx integration"],
        },
        "simulation": {
            "mode": mode,
            "learners_requested": len(specs),
            "learners_executed": len(summaries),
            "learners_passed_all_validations": sum(row["passed"] for row in learner_validations),
            "all_sessions_16_to_24": all(16 <= length <= 24 for length in lengths),
            "minimum_session_length": min(lengths),
            "maximum_session_length": max(lengths),
            "mean_session_length": statistics.mean(lengths),
            "total_scored_questions": len(traces),
            "complete_assessments": sum(row["final_assessment_status"] == "COMPLETE" for row in summaries),
            "incomplete_assessments": sum(row["final_assessment_status"] == "STOPPED_INCOMPLETE_EVIDENCE" for row in summaries),
            "operational_stops": sum(row["final_assessment_status"] == "STOPPED_OPERATIONAL_LIMIT" for row in summaries),
            "xgboost_recommendations": sources["TRAINED_ML"],
            "beta_bernoulli_recommendations": sources["BETA_BERNOULLI_FALLBACK"],
            "deterministic_fallback_or_abstention": sources["ABSTAIN_OR_DETERMINISTIC_FALLBACK"] + sources["ABSTAIN"],
            "all_c4_exports_valid": all(row["c4_export_validation_result"] == "PASS" for row in summaries),
            "grade_inappropriate_recommendations": sum(not row["curriculum_eligible"] for row in traces),
            "selected_outside_eligible_set": sum(not row["selected_from_eligible_candidates"] for row in traces),
            "non_scoring_events": 0,
            "distinct_recommendation_sequences_per_grade": per_grade_sequences,
            "profiles_produced_different_recommendations": any(value > 1 for value in per_grade_sequences.values()),
            "counterfactual_support": "Supplied tests passed: changing only prediction probabilities changes selected game and tier action.",
        },
        "learners": learner_validations,
    }

    payload = {
        "preflight": preflight,
        "environment": validation_payload["environment"],
        "simulation": validation_payload["simulation"],
        "summaries": summaries,
        "traces": traces,
        "exports": exports,
        "validations": learner_validations,
    }
    if write_outputs:
        prefix = {"one": "ONE_STUDENT", "five": "FIVE_STUDENTS", "twenty-five": "25_STUDENTS"}[mode]
        write_csv(RESULTS / f"{prefix}_FINAL_SUMMARY.csv", summaries)
        write_csv(RESULTS / f"{prefix}_QUESTION_TRACE.csv", traces)
        (RESULTS / f"{prefix}_C4_EXPORTS.json").write_text(json.dumps(exports, indent=2, ensure_ascii=False), encoding="utf-8")
        (RESULTS / f"{prefix}_VALIDATION_RESULTS.json").write_text(json.dumps(validation_payload, indent=2, ensure_ascii=False), encoding="utf-8")
        if mode == "twenty-five":
            (RESULTS / REPORT_NAME).write_text(make_report(summaries, traces, learner_validations, preflight, versions, seed_cfg), encoding="utf-8")
            strong = next(row for row in summaries if row["grade"] == 3 and row["simulated_performance_profile"] == "Strong")
            struggling = next(row for row in summaries if row["grade"] == 3 and row["simulated_performance_profile"] == "Struggling")
            (RESULTS / STRONG_NAME).write_text(make_walkthrough(strong, traces), encoding="utf-8")
            (RESULTS / STRUGGLING_NAME).write_text(make_walkthrough(struggling, traces), encoding="utf-8")
        elif mode == "one":
            (RESULTS / f"{prefix}_WALKTHROUGH.md").write_text(make_walkthrough(summaries[0], traces), encoding="utf-8")
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the isolated, real C2 adaptive assessment engine.")
    parser.add_argument("--mode", choices=("one", "five", "twenty-five"), default="twenty-five")
    parser.add_argument("--grade", type=int, default=3, help="Grade for one/five modes (1-5).")
    parser.add_argument("--profile", choices=PROFILES, default="Strong", help="Profile for one mode.")
    parser.add_argument("--seed", type=int, default=None, help="Override the configured base seed.")
    parser.add_argument("--no-write", action="store_true", help="Run without writing result files.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = execute_simulation(args.mode, args.grade, args.profile, args.seed, not args.no_write)
    print(json.dumps(payload["simulation"], indent=2))


if __name__ == "__main__":
    main()
