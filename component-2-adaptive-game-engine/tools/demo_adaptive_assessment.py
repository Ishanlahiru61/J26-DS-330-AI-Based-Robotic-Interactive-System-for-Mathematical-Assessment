#!/usr/bin/env python3
"""Reproducible standalone Grade 3 adaptive assessment demonstration."""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from component2_app.service import Component2Service


def run_pattern(label: str, always_correct: bool) -> dict:
    with tempfile.TemporaryDirectory() as folder:
        service = Component2Service(Path(folder) / "assessment.db")
        learner_id = f"GRADE3_{label}"
        service.db.upsert_learner(learner_id, label, 3, True)
        session_id = service.start_or_resume(learner_id)["session_id"]
        decisions = []
        for step in range(48):
            task = service.next_task(session_id, f"{label}-{step}")
            recommendation = task["recommendation"]
            if task["question_id"] is None:
                terminal = recommendation
                break
            with service.db.connect() as db:
                expected = db.execute("SELECT expected_answer FROM question_instances WHERE question_id=?", (task["question_id"],)).fetchone()[0]
            submitted = expected if always_correct else 0
            result = service.submit(session_id, task["question_id"], submitted)
            decisions.append({
                "step": step + 1, "game": recommendation["game_type"], "tier": recommendation["tier"],
                "decision_type": recommendation["decision_type"], "mode": recommendation["recommendation_mode"],
                "success_probability": recommendation["predicted_probability"],
                "reason": recommendation["reason_code"], "correct": result["correct"],
            })
        else:
            raise RuntimeError("assessment did not reach a stopping condition")
        profile = service.export_c4(learner_id, session_id)
        return {"learner": label, "response_pattern": "all correct" if always_correct else "repeated incorrect",
                "decisions": decisions, "terminal": terminal, "summary": profile}


def run_demo() -> dict:
    return {"stronger_learner": run_pattern("STRONGER", True),
            "reinforcement_learner": run_pattern("REINFORCEMENT", False),
            "claim_boundary": "The live primary model is TRANSFER_H2_GB_RT_OFF_v1, trained on transferable strict-prior ASSISTments features; Beta-Bernoulli is fallback. Neither is validated for real local children."}


def main() -> int:
    result = run_demo()
    for key in ("stronger_learner", "reinforcement_learner"):
        run = result[key]
        print(f"\n{run['learner']} ({run['response_pattern']})")
        for row in run["decisions"]:
            probability = "abstain" if row["success_probability"] is None else f"{row['success_probability']:.3f}"
            print(f"{row['step']:02d} {row['game']:<14} {row['tier']:<6} {row['decision_type']:<9} p={probability:<7} {row['mode']}")
        assessment = run["summary"]["profile"]["assessment"]
        print(f"Final status: {assessment['status']} | valid responses: {assessment['total_valid_interactions']}")
        print(f"Areas still requiring assessment: {assessment['areas_requiring_further_assessment']}")
    print("\nClaim boundary:", result["claim_boundary"])
    print("\nJSON summary:")
    compact = {key: {"learner": result[key]["learner"],
                     "response_pattern": result[key]["response_pattern"],
                     "decision_count": len(result[key]["decisions"]),
                     "decision_types": sorted({row["decision_type"] for row in result[key]["decisions"]}),
                     "games": sorted({row["game"] for row in result[key]["decisions"]}),
                     "assessment": result[key]["summary"]["profile"]["assessment"]}
               for key in ("stronger_learner", "reinforcement_learner")}
    print(json.dumps(compact, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
