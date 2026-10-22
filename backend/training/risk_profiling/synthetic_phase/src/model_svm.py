"""Step 4b - Support Vector Machine (proposal model).

Same input, split and risk bands as the formula (model_formula.py) for a fair comparison:
features F1-F8, missing features set to 0 (= expected for grade and age) with missing
indicators, class_weight="balanced" (5% DD), RBF kernel with C and gamma tuned by
stratified 5-fold cross-validation on the TRAIN split only (scoring: AUC).

Usage: python model_svm.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

sys.path.insert(0, str(Path(__file__).resolve().parent))
from model_formula import (  # noqa: E402
    ARTIFACTS, DECARLI, REPO, RESULTS, SEED, WEIGHTS, auc_with_ci, band, fit_cutoffs, screening_metrics,
)

FEATURES = list(WEIGHTS)  # F1-F8, same as the formula
PARAM_GRID = {"svm__C": [0.01, 0.1, 1, 10], "svm__gamma": ["scale", 0.01, 0.1]}


def make_pipeline() -> Pipeline:
    return Pipeline([
        ("impute", SimpleImputer(strategy="constant", fill_value=0.0, add_indicator=True)),
        ("svm", SVC(kernel="rbf", class_weight="balanced")),
    ])


def tune(x: pd.DataFrame, y: np.ndarray) -> GridSearchCV:
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    search = GridSearchCV(make_pipeline(), PARAM_GRID, scoring="roc_auc", cv=cv, n_jobs=-1)
    return search.fit(x, y)


def evaluate_split(scored: pd.DataFrame, split: str) -> dict:
    d = scored[(scored["split"] == split) & (scored["band"] != "insufficient")]
    y = d["dd_label"].to_numpy()
    return {
        "auc": auc_with_ci(y, d["risk_index"].to_numpy()),
        "moderate_or_high": screening_metrics(y, d["band"].isin(["moderate", "high"]).to_numpy()),
        "high": screening_metrics(y, (d["band"] == "high").to_numpy()),
        "auc_by_grade": {int(g): round(float(auc_with_ci(dg["dd_label"].to_numpy(),
                                                          dg["risk_index"].to_numpy(), 200)["auc"]), 3)
                         for g, dg in d.groupby("grade") if dg["dd_label"].sum() >= 2},
    }


def evaluate_real_calibration(model: GridSearchCV) -> dict | None:
    path = REPO / "data" / "private" / "risk_profiling" / "real_decarli_calibration_z.csv"
    if not path.exists():
        return None
    real = pd.read_csv(path).rename(columns=DECARLI)
    real[list(DECARLI.values())] = real[list(DECARLI.values())].clip(-3, 3)
    x = real.reindex(columns=FEATURES)  # F3/F4 not measured -> set to 0 by the imputer
    return {"auc": auc_with_ci(real["dd_label"].to_numpy(), model.decision_function(x)),
            "note": "F3/F4 not measured in Decarli; imputed as 0. Cut-offs not applied (norms differ)."}


def main() -> None:
    features = pd.read_csv(ARTIFACTS / "features_v1.csv")
    labels = pd.read_csv(ARTIFACTS / "synthetic_trials_v1_children.csv").set_index("child_id")["dd_label"]
    features["dd_label"] = features["student_id"].map(labels)
    train = features[features["split"] == "train"]

    model = tune(train[FEATURES], train["dd_label"].to_numpy())
    scored = features.copy()
    scored["risk_index"] = model.decision_function(features[FEATURES])
    cutoffs = fit_cutoffs(scored)
    scored["band"] = scored.apply(band, axis=1, cutoffs=cutoffs)

    report = {
        "model": "SVM (RBF) v1", "features": FEATURES,
        "best_params": model.best_params_, "cv_auc_train": round(float(model.best_score_), 3),
        "synthetic": {s: evaluate_split(scored, s) for s in ("train", "validation")},
        "real_decarli_calibration": evaluate_real_calibration(model),
    }
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "model_svm_metrics.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    scored[["student_id", "grade", "split", "risk_index", "band"]].round(4) \
        .to_csv(ARTIFACTS / "predictions_svm_v1.csv", index=False)
    print(json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
    main()
