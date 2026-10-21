"""Step 4c - Random Forest (proposal model).

Same input, split and risk bands as the formula and the SVM (fair comparison): features F1-F8,
missing features set to 0 (= expected for grade and age) with missing indicators,
class_weight="balanced_subsample" (5% DD). Tree depth, leaf size and features per split are
tuned by stratified 5-fold cross-validation on the TRAIN split only (scoring: AUC).

A forest memorises its training children, so train results and the band cut-offs use
OUT-OF-FOLD train scores (cross_val_predict). Results are reported at the 16% / 5% bands and at
the screening cut-off (4-S, register F5). Permutation importance on validation shows which
features the forest relies on.

Usage: python model_random_forest.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline

sys.path.insert(0, str(Path(__file__).resolve().parent))
from model_formula import ARTIFACTS, DECARLI, REPO, RESULTS, SEED, auc_with_ci, band, fit_cutoffs  # noqa: E402
from model_svm import FEATURES, evaluate_split  # noqa: E402
from sensitivity_analysis import choose_cutoff, flag, sweep  # noqa: E402
from model_formula import screening_metrics  # noqa: E402

PARAM_GRID = {
    "rf__max_depth": [3, 5, 8],
    "rf__min_samples_leaf": [10, 30, 100],
    "rf__max_features": ["sqrt", 0.5],
}


def make_pipeline() -> Pipeline:
    return Pipeline([
        ("impute", SimpleImputer(strategy="constant", fill_value=0.0, add_indicator=True)),
        ("rf", RandomForestClassifier(n_estimators=500, class_weight="balanced_subsample",
                                      random_state=SEED, n_jobs=-1)),
    ])


def cv_folds() -> StratifiedKFold:
    return StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)


def tune(x: pd.DataFrame, y: np.ndarray) -> GridSearchCV:
    return GridSearchCV(make_pipeline(), PARAM_GRID, scoring="roc_auc", cv=cv_folds()).fit(x, y)


def score_children(features: pd.DataFrame, search: GridSearchCV) -> pd.DataFrame:
    """Out-of-fold scores for train children, fitted-model scores for everyone else."""
    train = (features["split"] == "train").to_numpy()
    score = np.empty(len(features))
    oof = make_pipeline().set_params(**search.best_params_)
    score[train] = cross_val_predict(oof, features.loc[train, FEATURES], features.loc[train, "dd_label"],
                                     cv=cv_folds(), method="predict_proba")[:, 1]
    score[~train] = search.predict_proba(features.loc[~train, FEATURES])[:, 1]
    return features.assign(risk_index=score)


def screening(scored: pd.DataFrame) -> dict:
    usable = scored[(scored["split"] != "test") & (scored["sufficient_data_area"] == 1)]
    rows = sweep(usable)
    return {"sweep": rows, "screening_cutoff": choose_cutoff(rows)}


def importance(model: GridSearchCV, features: pd.DataFrame) -> dict:
    val = features[(features["split"] == "validation") & (features["sufficient_data_area"] == 1)]
    result = permutation_importance(model, val[FEATURES], val["dd_label"], scoring="roc_auc",
                                    n_repeats=30, random_state=SEED, n_jobs=-1)
    order = np.argsort(-result.importances_mean)
    return {FEATURES[i]: {"auc_drop": round(float(result.importances_mean[i]), 4),
                          "sd": round(float(result.importances_std[i]), 4)} for i in order}


def evaluate_real_calibration(model: GridSearchCV) -> dict | None:
    path = REPO / "data" / "private" / "risk_profiling" / "real_decarli_calibration_z.csv"
    if not path.exists():
        return None
    real = pd.read_csv(path).rename(columns=DECARLI)
    real[list(DECARLI.values())] = real[list(DECARLI.values())].clip(-3, 3)
    x = real.reindex(columns=FEATURES)
    return {"auc": auc_with_ci(real["dd_label"].to_numpy(), model.predict_proba(x)[:, 1]),
            "note": "F3/F4 not measured in Decarli; imputed as 0. Cut-offs not applied (norms differ)."}


def main() -> None:
    features = pd.read_csv(ARTIFACTS / "features_v1.csv")
    labels = pd.read_csv(ARTIFACTS / "synthetic_trials_v1_children.csv").set_index("child_id")["dd_label"]
    features["dd_label"] = features["student_id"].map(labels)
    train = features[features["split"] == "train"]

    search = tune(train[FEATURES], train["dd_label"].to_numpy())
    scored = score_children(features, search)
    cutoffs = fit_cutoffs(scored)
    scored["band"] = scored.apply(band, axis=1, cutoffs=cutoffs)

    screen = screening(scored)
    chosen = screen["screening_cutoff"]
    val = scored[(scored["split"] == "validation") & (scored["sufficient_data_area"] == 1)]
    at_screen = None
    if chosen:
        pct = 100 - chosen["flag_worst_pct"]
        at_screen = {"flag_worst_pct": chosen["flag_worst_pct"],
                     "validation": screening_metrics(val["dd_label"].to_numpy(), flag(val, pct, scored[
                         (scored["split"] == "train") & (scored["sufficient_data_area"] == 1)]).to_numpy())}

    report = {
        "model": "Random Forest v1", "features": FEATURES,
        "best_params": search.best_params_, "cv_auc_train": round(float(search.best_score_), 3),
        "note_train": "train metrics use out-of-fold scores",
        "synthetic": {s: evaluate_split(scored, s) for s in ("train", "validation")},
        "screening_cutoff": at_screen,
        "screening_sweep": screen["sweep"],
        "permutation_importance_validation": importance(search, features),
        "real_decarli_calibration": evaluate_real_calibration(search),
    }
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "model_random_forest_metrics.json").write_text(json.dumps(report, indent=2, default=str),
                                                               encoding="utf-8")
    scored[["student_id", "grade", "split", "risk_index", "band"]].round(4) \
        .to_csv(ARTIFACTS / "predictions_random_forest_v1.csv", index=False)
    print(json.dumps({k: v for k, v in report.items() if k != "screening_sweep"}, indent=2, default=str))


if __name__ == "__main__":
    main()
