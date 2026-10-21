"""Step 4e - logistic regression (reference model, TRIPOD+AI).

risk (log-odds) = b0 + sum(b_i * z_i). Same input, split, imputation and risk bands as 4a-4d:
features F1-F8, missing features set to 0 (= expected for grade and age) with missing
indicators, class_weight="balanced" (5% DD). L2 penalty; its strength C is tuned by stratified
5-fold cross-validation on the TRAIN split only (scoring: AUC). Train results and band cut-offs
use out-of-fold scores, as in 4c-4d.

The learned weights are compared with the meta-analysis evidence weights (feature_evidence.md):
does learning from labels give the same weighting as the published evidence?

Usage: python model_logistic.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline

sys.path.insert(0, str(Path(__file__).resolve().parent))
from model_formula import ARTIFACTS, DECARLI, REPO, RESULTS, SEED, WEIGHTS, auc_with_ci, band, fit_cutoffs  # noqa: E402
from model_formula import screening_metrics  # noqa: E402
from model_svm import FEATURES, evaluate_split  # noqa: E402
from sensitivity_analysis import choose_cutoff, flag, sweep  # noqa: E402

PARAM_GRID = {"lr__C": [0.001, 0.01, 0.1, 1, 10, 100]}  # 100 added: v1 grid chose its edge (10)


def make_pipeline() -> Pipeline:
    return Pipeline([
        ("impute", SimpleImputer(strategy="constant", fill_value=0.0, add_indicator=True)),
        ("lr", LogisticRegression(class_weight="balanced", max_iter=1000)),
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
                                     cv=cv_folds(), method="decision_function")
    score[~train] = search.decision_function(features.loc[~train, FEATURES])
    return features.assign(risk_index=score)


def coefficients(model: Pipeline) -> dict:
    names = [n.replace("missingindicator_", "missing_")
             for n in model.named_steps["impute"].get_feature_names_out(FEATURES)]
    coef = pd.Series(model.named_steps["lr"].coef_[0], index=names)
    learned = coef[FEATURES]
    evidence = pd.Series(WEIGHTS)[FEATURES]
    rho, p = spearmanr(learned, evidence)
    return {
        "intercept": round(float(model.named_steps["lr"].intercept_[0]), 4),
        "weights": {f: {"learned": round(float(learned[f]), 4), "evidence": evidence[f],
                        "learned_share": round(float(learned[f] / learned.clip(lower=0).sum()), 3),
                        "evidence_share": round(float(evidence[f] / evidence.sum()), 3)} for f in FEATURES},
        "missing_indicators": {n: round(float(v), 4) for n, v in coef.drop(FEATURES).items()},
        "spearman_learned_vs_evidence": {"rho": round(float(rho), 3), "p_value": round(float(p), 3)},
    }


def evaluate_real_calibration(model) -> dict | None:
    path = REPO / "data" / "private" / "risk_profiling" / "real_decarli_calibration_z.csv"
    if not path.exists():
        return None
    real = pd.read_csv(path).rename(columns=DECARLI)
    real[list(DECARLI.values())] = real[list(DECARLI.values())].clip(-3, 3)
    x = real.reindex(columns=FEATURES)
    return {"auc": auc_with_ci(real["dd_label"].to_numpy(), model.decision_function(x)),
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
        "model": "Logistic regression v1 (L2)", "features": FEATURES,
        "best_params": search.best_params_, "cv_auc_train": round(float(search.best_score_), 3),
        "note_train": "train metrics use out-of-fold scores",
        "coefficients": coefficients(search.best_estimator_),
        "synthetic": {s: evaluate_split(scored, s) for s in ("train", "validation")},
        "screening_cutoff": at_screen,
        "screening_sweep": rows,
        "real_decarli_calibration": evaluate_real_calibration(search),
    }
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "model_logistic_metrics.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    scored[["student_id", "grade", "split", "risk_index", "band"]].round(4) \
        .to_csv(ARTIFACTS / "predictions_logistic_v1.csv", index=False)
    print(json.dumps({k: v for k, v in report.items() if k != "screening_sweep"}, indent=2, default=str))


if __name__ == "__main__":
    main()
