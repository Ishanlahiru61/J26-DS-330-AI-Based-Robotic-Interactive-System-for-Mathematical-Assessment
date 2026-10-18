#!/usr/bin/env python3
"""One end-to-end Grade 3 demonstration through Component2Service."""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from component2_app.service import Component2Service


def run_learner(label: str, always_correct: bool) -> dict:
    with tempfile.TemporaryDirectory() as folder:
        service = Component2Service(Path(folder) / "assessment.db")
        learner_id = f"PP1_G3_{label}"
        service.db.upsert_learner(learner_id, label, 3, True)
        session_id = service.start_or_resume(learner_id)["session_id"]
        rows = []
        for step in range(48):
            task = service.next_task(session_id, f"{label}-{step}")
            recommendation = task["recommendation"]
            if task["question_id"] is None:
                terminal = recommendation
                break
            provenance = recommendation.get("model_provenance") or {}
            candidate_predictions = provenance.get("candidate_skill_predictions", {})
            with service.db.connect() as db:
                expected = db.execute(
                    "SELECT expected_answer FROM question_instances WHERE question_id=?",
                    (task["question_id"],),
                ).fetchone()[0]
            submitted = expected if always_correct else (expected % 25) + 1
            result = service.submit(session_id, task["question_id"], submitted, response_ms=1200 + step)
            rows.append({
                "step": step + 1,
                "eligible_candidate_ids": recommendation["eligible_candidate_ids"],
                "candidate_skill_predictions": candidate_predictions,
                "selected_game": recommendation["game_type"],
                "selected_tier": recommendation["tier"],
                "decision_type": recommendation["decision_type"],
                "probability": recommendation["predicted_probability"],
                "prediction_source": provenance.get("source", "ABSTAIN"),
                "model_version": recommendation.get("model_version"),
                "recommendation_mode": recommendation["recommendation_mode"],
                "reason": recommendation["reason_code"],
                "history_count_before_answer": recommendation["history_count"],
                "answer_recorded": result["status"],
                "correct": result["correct"],
            })
        else:
            raise RuntimeError("assessment did not reach its documented stopping rule")
        export = service.export_c4(learner_id, session_id)
        return {"learner": label, "session_id": session_id, "decisions": rows,
                "terminal": terminal, "c4_export": export}


def main() -> int:
    runs = [run_learner("STRONG", True), run_learner("STRUGGLING", False)]
    print("Unified trained-ML adaptive assessment demonstration")
    print("Initial task -> answer -> transactional history -> curriculum candidates -> prediction -> recommendation -> repeat -> C4 export")
    for run in runs:
        print(f"\n{run['learner']} learner")
        for row in run["decisions"]:
            p = "ABSTAIN" if row["probability"] is None else f"{row['probability']:.3f}"
            print(
                f"{row['step']:02d} history={row['history_count_before_answer']:02d} "
                f"{row['selected_game']:<14} {row['selected_tier']:<6} "
                f"{row['decision_type']:<9} p={p:<7} {row['prediction_source']:<12} "
                f"answer={row['correct']}"
            )
            if row["candidate_skill_predictions"]:
                compact = ", ".join(
                    f"{skill}={value['success_probability']:.3f}"
                    for skill, value in row["candidate_skill_predictions"].items()
                    if value["success_probability"] is not None
                )
                if compact:
                    print(f"   eligible tied-skill predictions: {compact}")
        assessment = run["c4_export"]["profile"]["assessment"]
        print(f"Final: {assessment['status']}; responses={assessment['total_valid_interactions']}; remaining={assessment['areas_requiring_further_assessment']}")
        print(f"C4 schema: {run['c4_export']['schema_version']}; descriptive/non-diagnostic={run['c4_export']['semantics']['not_a_diagnosis']}")
    evidence = {
        run["learner"]: {
            "steps": len(run["decisions"]),
            "decisions": sorted({row["decision_type"] for row in run["decisions"]}),
            "games": sorted({row["selected_game"] for row in run["decisions"]}),
            "trained_ml_decisions": sum(row["prediction_source"] == "TRAINED_ML" for row in run["decisions"]),
            "assessment": run["c4_export"]["profile"]["assessment"],
        } for run in runs
    }
    print("\nEvidence summary")
    print(json.dumps(evidence, indent=2))
    print("\nClaim boundary: held-out ASSISTments predictive evidence plus functional local simulation; no real-child local validation or diagnosis.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
