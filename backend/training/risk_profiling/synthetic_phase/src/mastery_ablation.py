"""Mastery classification (proposal 3.4) - part 3: feature-group ablation.

Does adding feature groups improve mastery classification? Feature sets are nested
(mastery_data.FEATURE_SETS): accuracy_only -> performance -> multimodal -> historical.
For each set, the selected model (SVM) and XGBoost are re-tuned by stratified 5-fold CV on the
TRAIN split (macro-F1) and scored on validation. Each step is compared with the previous set by
a paired bootstrap (same validation children) of macro-F1 and Needs Urgent Assistance recall.
The test split is not used.

Usage: python mastery_ablation.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mastery_data import FEATURE_SETS  # noqa: E402
from mastery_models import N_BOOT, tune, urgent_recall  # noqa: E402
from model_formula import ARTIFACTS, RESULTS, SEED  # noqa: E402

ORDER = ["accuracy_only", "performance", "multimodal", "historical"]
MODELS = ["svm", "xgboost"]


def step_difference(y: np.ndarray, before: np.ndarray, after: np.ndarray) -> dict:
    """Paired bootstrap of (after - before) for macro-F1 and urgent recall."""
    rng = np.random.default_rng(SEED)
    groups = [np.where(y == c)[0] for c in np.unique(y)]
    diffs = {"macro_f1": [], "urgent_recall": []}
    for _ in range(N_BOOT):
        idx = np.concatenate([rng.choice(g, len(g)) for g in groups])
        diffs["macro_f1"].append(f1_score(y[idx], after[idx], average="macro") - f1_score(y[idx], before[idx], average="macro"))
        diffs["urgent_recall"].append(urgent_recall(y[idx], after[idx]) - urgent_recall(y[idx], before[idx]))
    return {k: [round(float(v), 3) for v in np.percentile(d, [2.5, 97.5])] for k, d in diffs.items()}


def main() -> None:
    data = pd.read_csv(ARTIFACTS / "mastery_features_v1.csv")
    train, val = data[data["split"] == "train"], data[data["split"] == "validation"]
    yt, yv = train["mastery"].to_numpy(), val["mastery"].to_numpy()
    report = {"feature_sets": {k: len(FEATURE_SETS[k]) for k in ORDER}, "models": {}}
    for name in MODELS:
        rows, previous = {}, None
        for fs in ORDER:
            search = tune(name, train[FEATURE_SETS[fs]], yt)
            pred = np.asarray(search.best_estimator_.predict(val[FEATURE_SETS[fs]]))
            rows[fs] = {"n_features": len(FEATURE_SETS[fs]), "cv_macro_f1_train": round(float(search.best_score_), 3),
                        "validation_macro_f1": round(float(f1_score(yv, pred, average="macro")), 3),
                        "validation_urgent_recall": round(urgent_recall(yv, pred), 3)}
            if previous is not None:
                rows[fs]["change_vs_previous_ci95"] = step_difference(yv, previous, pred)
            previous = pred
            print(name, fs, rows[fs])
        report["models"][name] = rows
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "mastery_ablation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
