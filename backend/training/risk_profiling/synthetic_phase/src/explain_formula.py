"""Step 4g (part 1) - per-child explanation for the selected model (evidence formula).

The formula is R = sum(w_i * z_i) / sum(w_i) over the features the child has, so each feature's
contribution c_i = w_i * z_i / sum(w_available) is exact and the contributions add up to R.
Features are grouped into skills (accuracy + speed of one task), as for SHAP (explain_shap.py).

For each skill:
- z      = evidence-weighted mean of its available feature z-scores (positive = worse)
- level  = needs_extra_help (z >= 1.645, <= 5th percentile, register F2)
           needs_practice   (z >= 1.0,   <= 16th percentile, F1)
           strength         (z <= -1.0,  >= 84th percentile, the same distance on the good side)
           as_expected      (otherwise)
- status = "not_assessed" when the skill has no data (e.g. symbolic comparison in Grade 1)
- practise = Component 2 activities for this skill that C2 offers in the child's grade

The output uses keys only (no sentences), so child_report.py can word it in any language.
Usage: python explain_formula.py   (writes artifacts/child_explanations_formula_v1.json)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from model_formula import ARTIFACTS, MODULE, WEIGHTS  # noqa: E402
from profiles import rule_profiles, screening_flag  # noqa: E402

SKILL_FEATURES = {  # skill -> features (task taxonomy: T05, T04, T02, T10-T14)
    "number_comparison": ["z_f1_symbolic_comparison_acc", "z_f2_symbolic_comparison_rt"],
    "quantity_comparison": ["z_f7_nonsymbolic_comparison_acc", "z_f8_nonsymbolic_comparison_rt"],
    "counting": ["z_f5_counting_rt", "z_f6_counting_acc"],
    "arithmetic": ["z_f3_arithmetic_acc", "z_f4_arithmetic_rt"],
}
LEVELS = [(1.645, "needs_extra_help"), (1.0, "needs_practice")]  # register F2, F1
STRENGTH_Z = -1.0

# Component 2 activities to practise per skill (C2 activity list, team_interface.md); offered only
# when C2 has the activity in the child's grade (NIE grade tables, nie_curriculum_map.md Section 0).
PRACTISE = {
    "number_comparison": ["comparison", "ordering"],
    "quantity_comparison": ["comparison"],
    "counting": ["counting", "place_value"],
    "arithmetic": ["addition", "subtraction", "multiplication", "division"],
}
C2_ACTIVITIES_BY_GRADE = {
    1: {"counting", "comparison", "place_value", "addition", "subtraction"},
    2: {"counting", "comparison", "ordering", "place_value", "addition", "subtraction"},
    3: {"counting", "comparison", "ordering", "place_value", "addition", "subtraction", "multiplication", "division"},
}


def level(z: float) -> str:
    for cut, name in LEVELS:
        if z >= cut:
            return name
    return "strength" if z <= STRENGTH_Z else "as_expected"


def explain_child(row: pd.Series, band: str, n_sessions: int | None = None, n_questions: int | None = None,
                  flagged: bool = False, profile: str | None = None) -> dict:
    """Keys-only explanation of one child's formula score.

    flagged: at the screening cut-off (worst 35%, register F5). A flagged child in the low band is
    reported as "monitor"; flagged children also get a difficulty profile (Step 5, profiles.py).
    """
    grade = int(row["grade"])
    available = {f: float(row[f]) for f in WEIGHTS if f in row and pd.notna(row[f])}
    weight_sum = sum(WEIGHTS[f] for f in available)
    out = {"student_id": row["student_id"], "grade": grade, "band": band,
           "data": {"n_sessions": n_sessions, "n_questions": n_questions}, "skills": {}}
    if band == "insufficient" or weight_sum == 0:
        out["band"] = "insufficient"
        return out

    if band == "low" and flagged:
        out["band"] = "monitor"
    out["profile"] = profile if flagged else None
    out["risk_index"] = round(sum(WEIGHTS[f] * z for f, z in available.items()) / weight_sum, 3)
    for skill, feats in SKILL_FEATURES.items():
        have = [f for f in feats if f in available]
        if not have:
            out["skills"][skill] = {"status": "not_assessed"}
            continue
        w = sum(WEIGHTS[f] for f in have)
        z = sum(WEIGHTS[f] * available[f] for f in have) / w
        skill_level = level(z)
        practise = [a for a in PRACTISE[skill] if a in C2_ACTIVITIES_BY_GRADE[grade]]
        out["skills"][skill] = {
            "status": "assessed", "z": round(z, 2), "level": skill_level,
            "contribution": round(sum(WEIGHTS[f] * available[f] for f in have) / weight_sum, 3),
            "practise": practise if skill_level in ("needs_practice", "needs_extra_help") else [],
        }
    assessed = {s: v for s, v in out["skills"].items() if v["status"] == "assessed"}
    out["main_reasons"] = [s for s, v in sorted(assessed.items(), key=lambda kv: -kv[1]["contribution"])
                           if v["level"] in ("needs_practice", "needs_extra_help")][:2]
    out["strengths"] = [s for s, v in assessed.items() if v["level"] == "strength"]
    out["on_track"] = [s for s, v in assessed.items() if v["level"] == "as_expected"]
    out["not_assessed"] = [s for s, v in out["skills"].items() if v["status"] == "not_assessed"]
    return out


def main() -> None:
    preds = pd.read_csv(ARTIFACTS / "predictions_formula_v1.csv")[["student_id", "risk_index", "band"]]
    features = pd.read_csv(ARTIFACTS / "features_v1.csv").merge(preds, on="student_id")
    children = pd.read_csv(ARTIFACTS / "preprocessed_children_v1.csv").set_index("student_id")
    selected = json.loads((MODULE / "config" / "selected_model.json").read_text(encoding="utf-8"))
    features["flagged"] = screening_flag(features, selected["screening_flag_worst_pct"])
    features["profile"] = rule_profiles(features)
    out = [explain_child(row, row["band"], int(children.at[row["student_id"], "n_sessions"]),
                         int(children.at[row["student_id"], "n_questions"]), bool(row["flagged"]), row["profile"])
           for _, row in features.iterrows()]
    (ARTIFACTS / "child_explanations_formula_v1.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(pd.Series([c["band"] for c in out]).value_counts().to_string())
    levels = pd.Series([v["level"] for c in out for v in c["skills"].values() if v.get("status") == "assessed"])
    print(f"{len(out)} children explained; skill levels:\n{levels.value_counts().to_string()}")
    print(json.dumps(next(c for c in out if c["band"] == "high"), indent=1))


if __name__ == "__main__":
    main()
