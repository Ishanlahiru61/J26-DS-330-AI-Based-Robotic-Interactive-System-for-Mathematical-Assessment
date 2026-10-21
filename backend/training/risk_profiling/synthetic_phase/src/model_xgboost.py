"""Step 4d - XGBoost (proposal model).

Same input, split, imputation and risk bands as the formula, SVM and Random Forest (fair
comparison): features F1-F8, missing features set to 0 (= expected for grade and age) with
missing indicators. Class imbalance (5% DD) is handled with scale_pos_weight = typical / DD
in the training split. Tree depth, learning rate, number of trees and minimum child weight
are tuned by stratified 5-fold cross-validation on the TRAIN split only (scoring: AUC);
subsample and colsample are fixed at 0.8 to limit overfitting.

As in 4c, train results and band cut-offs use OUT-OF-FOLD train scores, and results are
reported at the 16% / 5% bands and at the screening cut-off (4-S, register F5).
The fitted pipeline is saved to artifacts/ for the SHAP explanations (explain_shap.py).

Usage: python model_xgboost.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

sys.path.insert(0, str(Path(__file__).resolve().parent))
from model_formula import ARTIFACTS, DECARLI, REPO, RESULTS, SEED, auc_with_ci, band, fit_cutoffs  # noqa: E402
from model_formula import screening_metrics  # noqa: E402
from model_svm import FEATURES, evaluate_split  # noqa: E402
from sensitivity_analysis import choose_cutoff, flag, sweep  # noqa: E402

MODEL_PATH = ARTIFACTS / "model_xgboost_v1.joblib"
# v1 grid {depth 2-4, rate 0.03/0.1, trees 100/300} chose its smallest corner (CV), so the grid
# was extended toward simpler models; the choice is still made by train cross-validation only.
PARAM_GRID = {
    "xgb__max_depth": [1, 2, 3, 4],
    "xgb__learning_rate": [0.01, 0.03, 0.1],
    "xgb__n_estimators": [50, 100, 300],
    "xgb__min_child_weight": [1, 5],
}


def make_pipeline(scale_pos_weight: float = 1.0) -> Pipeline:
    return Pipeline([
        ("impute", SimpleImputer(strategy="constant", fill_value=0.0, add_indicator=True)),
        ("xgb", XGBClassifier(objective="binary:logistic", eval_metric="auc", subsample=0.8,
                              colsample_bytree=0.8, scale_pos_weight=scale_pos_weight,
                              random_state=SEED, n_jobs=1, tree_method="hist")),
    ])


def cv_folds() -> StratifiedKFold:
    return StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)


def imbalance(y: np.ndarray) -> float:
    return float((y == 0).sum() / max((y == 1).sum(), 1))


def tune(x: pd.DataFrame, y: np.ndarray) -> GridSearchCV:
    search = GridSearchCV(make_pipeline(imbalance(y)), PARAM_GRID, scoring="roc_auc", cv=cv_folds(), n_jobs=-1)
    return search.fit(x, y)


def score_children(features: pd.DataFrame, search: GridSearchCV) -> pd.DataFrame:
    """Out-of-fold scores for train children, fitted-model scores for everyone else."""
    train = (features["split"] == "train").to_numpy()
    y = features.loc[train, "dd_label"]
    score = np.empty(len(features))
    oof = make_pipeline(imbalance(y.to_numpy())).set_params(**search.best_params_)
    score[train] = cross_val_predict(oof, features.loc[train, FEATURES], y, cv=cv_folds(),
                                     method="predict_proba", n_jobs=-1)[:, 1]
    score[~train] = search.predict_proba(features.loc[~train, FEATURES])[:, 1]
    return features.assign(risk_index=score)


def evaluate_real_calibration(model) -> dict | None:
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

    usable = scored[(scored["split"] != "test") & (scored["sufficient_data_area"] == 1)]
    rows = sweep(usable)
    chosen = choose_cutoff(rows)
    at_screen = None
    if chosen:
        val = usable[usable["split"] == "validation"]
        ref = usable[usable["split"] == "train"]
        at_screen = {"flag_worst_pct": chosen["flag_worst_pct"],
                     "validation": screening_metrics(val["dd_label"].to_numpy(),
                                                     flag(val, 100 - chosen["flag_worst_pct"], ref).to_numpy())}

    report = {
        "model": "XGBoost v1", "features": FEATURES,
        "best_params": search.best_params_, "cv_auc_train": round(float(search.best_score_), 3),
        "scale_pos_weight": round(imbalance(train["dd_label"].to_numpy()), 2),
        "note_train": "train metrics use out-of-fold scores",
        "synthetic": {s: evaluate_split(scored, s) for s in ("train", "validation")},
        "screening_cutoff": at_screen,
        "screening_sweep": rows,
        "real_decarli_calibration": evaluate_real_calibration(search),
    }
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "model_xgboost_metrics.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    joblib.dump(search.best_estimator_, MODEL_PATH)
    scored[["student_id", "grade", "split", "risk_index", "band"]].round(4) \
        .to_csv(ARTIFACTS / "predictions_xgboost_v1.csv", index=False)
    print(json.dumps({k: v for k, v in report.items() if k != "screening_sweep"}, indent=2, default=str))


if __name__ == "__main__":
    main()
