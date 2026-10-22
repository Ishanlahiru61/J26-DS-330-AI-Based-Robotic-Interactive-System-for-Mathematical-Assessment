"""Step 6 - final evaluation of the selected model on the untouched test data (run once).

The model is frozen in config/selected_model.json (evidence formula, 4f): weights from
feature_evidence.md, band cut-offs from results/formula_cutoffs.json (train children), screening
flag at the worst 35% of same-grade train children (4-S). Nothing is fitted or tuned here.

Data opened for the first time:
- synthetic test split (Step 2 split, 15% of children)
- the 14 real Decarli test children (data/private, not in Git)

Measures: AUC with bootstrap 95% CI; sensitivity / specificity (Wilson 95% CI) at the high band,
moderate-or-high band and screening flag; AUC and flag rates by grade; calibration of the flag
rates against their nominal percentiles; share of children with insufficient data; and a safety
check of the child report for every test child. Real children: AUC only (synthetic cut-offs do
not apply to Italian norms).

Usage: python final_evaluation.py
"""

from __future__ import annotations

import datetime as dt
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from child_report import load_template, parent_report, teacher_report  # noqa: E402
from explain_formula import explain_child  # noqa: E402
from model_formula import ARTIFACTS, DECARLI, MODULE, REPO, RESULTS, auc_with_ci, band, risk_index  # noqa: E402
from profiles import rule_profiles, screening_flag  # noqa: E402

FORBIDDEN = ("dyscalcul", "disorder", "disabilit")


def wilson(k: int, n: int, z: float = 1.96) -> list[float] | None:
    if n == 0:
        return None
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return [round(float(centre - half), 3), round(float(centre + half), 3)]


def screening(y: np.ndarray, flagged: np.ndarray) -> dict:
    tp, fn = int((flagged & (y == 1)).sum()), int((~flagged & (y == 1)).sum())
    tn, fp = int((~flagged & (y == 0)).sum()), int((flagged & (y == 0)).sum())
    return {"sensitivity": round(tp / (tp + fn), 3) if tp + fn else None, "sensitivity_ci95": wilson(tp, tp + fn),
            "specificity": round(tn / (tn + fp), 3) if tn + fp else None, "specificity_ci95": wilson(tn, tn + fp),
            "flagged_share": round(float(flagged.mean()), 3), "tp": tp, "fn": fn, "fp": fp, "tn": tn}


def score_synthetic(selected: dict) -> pd.DataFrame:
    features = pd.read_csv(ARTIFACTS / "features_v1.csv")
    labels = pd.read_csv(ARTIFACTS / "synthetic_trials_v1_children.csv").set_index("child_id")["dd_label"]
    cutoffs = {int(g): c for g, c in json.loads((MODULE / selected["cutoffs_file"]).read_text(encoding="utf-8"))
               ["cutoffs_by_grade"].items()}
    scored = pd.concat([features, risk_index(features)], axis=1)
    scored["dd_label"] = scored["student_id"].map(labels)
    scored["band"] = scored.apply(band, axis=1, cutoffs=cutoffs)
    scored["flagged"] = screening_flag(scored, selected["screening_flag_worst_pct"])  # train reference only
    scored["profile"] = rule_profiles(scored)
    return scored


def evaluate_synthetic_test(scored: pd.DataFrame, selected: dict) -> dict:
    test = scored[scored["split"] == "test"]
    usable = test[test["band"] != "insufficient"]
    y = usable["dd_label"].to_numpy()
    nominal = {"high": 100 - selected["bands_percentiles"]["high"],
               "moderate_or_high": 100 - selected["bands_percentiles"]["moderate"],
               "screening_flag": selected["screening_flag_worst_pct"]}
    rules = {"high": (usable["band"] == "high").to_numpy(),
             "moderate_or_high": usable["band"].isin(["moderate", "high"]).to_numpy(),
             "screening_flag": usable["flagged"].to_numpy()}
    by_grade = {}
    for g, d in usable.groupby("grade"):
        typical = d["dd_label"] == 0
        by_grade[int(g)] = {
            "n": int(len(d)), "n_dd": int(d["dd_label"].sum()),
            "auc": auc_with_ci(d["dd_label"].to_numpy(), d["risk_index"].to_numpy(), 1000)
            if d["dd_label"].sum() >= 2 else None,
            "screening_flag": screening(d["dd_label"].to_numpy(), d["flagged"].to_numpy()),
            "flag_rate_typical_children_pct": round(float(d.loc[typical, "flagged"].mean() * 100), 1),
        }
    return {
        "children": int(len(test)), "dd": int(test["dd_label"].sum()),
        "insufficient": {"n": int((test["band"] == "insufficient").sum()),
                         "share": round(float((test["band"] == "insufficient").mean()), 3)},
        "auc": auc_with_ci(y, usable["risk_index"].to_numpy()),
        "rules": {name: screening(y, f) for name, f in rules.items()},
        "calibration_flag_rates_pct": {
            name: {"nominal": nominal[name], "observed_all": round(float(f.mean() * 100), 1),
                   "observed_typical_children": round(float(f[y == 0].mean() * 100), 1)}
            for name, f in rules.items()},
        "by_grade": by_grade,
    }


def evaluate_real_test() -> dict | None:
    path = REPO / "data" / "private" / "risk_profiling" / "real_decarli_test_z.csv"
    if not path.exists():
        return None
    real = pd.read_csv(path).rename(columns=DECARLI)
    real[list(DECARLI.values())] = real[list(DECARLI.values())].clip(-3, 3)  # E5
    score = risk_index(real)["risk_index"].to_numpy()
    y = real["dd_label"].to_numpy()
    return {"auc": auc_with_ci(y, score), "n": int(len(y)), "n_dd": int(y.sum()),
            "mean_score_dd": round(float(score[y == 1].mean()), 3),
            "mean_score_typical": round(float(score[y == 0].mean()), 3),
            "note": "Real Italian children aged 7-14; F3/F4 not measured; cut-offs not applied (norms differ)."}


def report_safety(scored: pd.DataFrame) -> dict:
    """Generate teacher and parent reports for every test child and check the report rules."""
    t = load_template("en")
    children = pd.read_csv(ARTIFACTS / "preprocessed_children_v1.csv").set_index("student_id")
    test = scored[scored["split"] == "test"]
    problems, bands = [], []
    for _, row in test.iterrows():
        exp = explain_child(row, row["band"], int(children.at[row["student_id"], "n_sessions"]),
                            int(children.at[row["student_id"], "n_questions"]), bool(row["flagged"]), row["profile"])
        bands.append(exp["band"])
        teacher, parent = teacher_report(exp, t).lower(), parent_report(exp, t).lower()
        if any(w in teacher or w in parent for w in FORBIDDEN):
            problems.append((row["student_id"], "diagnostic word"))
        for skill in exp.get("not_assessed", []):
            if t["skill_names_parent"][skill] in parent:
                problems.append((row["student_id"], f"not-assessed skill in parent text: {skill}"))
        if exp["band"] == "insufficient" and "not enough data" not in teacher:
            problems.append((row["student_id"], "insufficient child given a result"))
    return {"reports_generated": int(2 * len(test)), "problems": problems,
            "report_levels": pd.Series(bands).value_counts().to_dict()}


def git_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True,
                              cwd=MODULE).stdout.strip()
    except OSError:
        return "unknown"


def main() -> None:
    selected = json.loads((MODULE / "config" / "selected_model.json").read_text(encoding="utf-8"))
    scored = score_synthetic(selected)
    report = {
        "model": selected["model"], "frozen_config": "config/selected_model.json",
        "test_data_opened": dt.date.today().isoformat(), "code_commit": git_commit(),
        "synthetic_test": evaluate_synthetic_test(scored, selected),
        "real_decarli_test": evaluate_real_test(),
        "child_report_safety": report_safety(scored),
        "note": "Single evaluation of the frozen model; no changes after this point without new test data (pilot).",
    }
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "final_evaluation.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
    main()
