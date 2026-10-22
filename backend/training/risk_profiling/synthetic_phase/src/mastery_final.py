"""Mastery classification - final evaluation of the selected model on the test split (run once).

The selected model (mastery_models.py, saved in artifacts/) is applied unchanged to the test children,
which were not used for tuning, selection, ablation or explanations. Reported: the proposal's metrics
(3.6.2) and robustness by grade and by amount of missing data (3.6.4). The accuracy-rule baseline is
reported alongside for reference (it has no trained parameters).

Usage: python mastery_final.py
"""

from __future__ import annotations

import datetime as dt
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from final_evaluation import git_commit  # noqa: E402
from mastery_models import FEATURES, accuracy_rule, metrics, urgent_recall  # noqa: E402
from model_formula import ARTIFACTS, RESULTS  # noqa: E402


def main() -> None:
    selected = json.loads((RESULTS / "mastery_models.json").read_text(encoding="utf-8"))["selected"]
    model = joblib.load(ARTIFACTS / f"mastery_{selected}_v1.joblib")
    data = pd.read_csv(ARTIFACTS / "mastery_features_v1.csv")
    test = data[data["split"] == "test"].copy()
    y = test["mastery"].to_numpy()
    pred = np.asarray(model.predict(test[FEATURES]))
    proba = model.predict_proba(test[FEATURES])

    test["pred"] = pred
    test["missing_features"] = test[FEATURES].isna().sum(axis=1)
    by_grade = {int(g): {"n": int(len(d)), "macro_f1": round(float(f1_score(d["mastery"], d["pred"], average="macro")), 3),
                         "urgent_recall": round(urgent_recall(d["mastery"].to_numpy(), d["pred"].to_numpy()), 3)}
                for g, d in test.groupby("grade")}
    by_missing = {("none" if k == 0 else "some"): {"n": int(len(d)), "macro_f1": round(float(
        f1_score(d["mastery"], d["pred"], average="macro")), 3)}
        for k, d in test.groupby(test["missing_features"] > 0)}

    report = {
        "model": selected, "test_data_opened": dt.date.today().isoformat(), "code_commit": git_commit(),
        "children": int(len(test)), "class_counts": np.bincount(y, minlength=3).tolist(),
        "selected_model": metrics(y, pred, proba),
        "accuracy_rule_baseline": metrics(y, accuracy_rule(test[FEATURES]), None),
        "by_grade": by_grade, "by_missing_data": by_missing,
        "note": "Single evaluation of the frozen model on synthetic test children.",
    }
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "mastery_final_evaluation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
