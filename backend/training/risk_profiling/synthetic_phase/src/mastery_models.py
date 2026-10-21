"""Mastery classification (proposal 3.5) - part 2: baseline, single models and the stacking ensemble.

Classes: 0 High Performer, 1 Progressing, 2 Needs Urgent Assistance (mastery_data.py).
Features: the full multimodal + historical set (mastery_data.FEATURE_SETS["historical"]).

Models (proposal 3.5.2-3.5.5)
- accuracy_rule  : baseline, overall-accuracy z >= 1.0 -> Needs Urgent, <= -1.0 -> High Performer
- random_forest  : class_weight balanced; depth and leaf size tuned
- svm            : RBF, features scaled, class_weight balanced; C and gamma tuned
- xgboost        : multi-class, balanced sample weights; depth, learning rate, trees tuned
- stacking       : RF + SVM base learners -> XGBoost meta-learner. Leakage-free (3.5.3): the meta-learner
                   is trained on OUT-OF-FOLD base predictions from 5-fold CV of the training set; the bases
                   are then refitted on the full training set to score new children.
Tuning: stratified 5-fold CV on the TRAIN split only, scored by macro-F1 (proposal 2.2).
Imbalance: class weights / sample weights on training data only (3.6.3).

Evaluation on validation (3.6.2): accuracy, macro precision / recall / F1, per-class recall,
confusion matrix, one-vs-rest ROC-AUC; recall of Needs Urgent Assistance with Wilson 95% CI.

Selection rule: among models that are not reliably worse than the best on BOTH validation macro-F1 and
recall of Needs Urgent Assistance (95% paired-bootstrap CI of each difference includes 0), choose the
simplest (order in COMPLEXITY). The recall criterion is the proposal's own requirement (3.6.2: a model
that misses children needing help is not acceptable); it was added after v1, whose macro-F1-only rule
selected the accuracy rule (Needs Urgent recall 0.43). Both results are reported. The test split is NOT used here; mastery_final.py evaluates the chosen model once.

Class decisions use each model's own predict (SVM: the class-weighted decision function, not Platt
probabilities); probabilities are used only for ROC-AUC. The meta-learner's class weighting
(balanced or none) is tuned, because the base learners already weight classes (v1 weighted twice).

Usage: python mastery_models.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, precision_score, recall_score,
                             roc_auc_score)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_predict, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

sys.path.insert(0, str(Path(__file__).resolve().parent))
from final_evaluation import wilson  # noqa: E402
from mastery_data import FEATURE_SETS, TIERS  # noqa: E402
from model_formula import ARTIFACTS, RESULTS, SEED  # noqa: E402

FEATURES = FEATURE_SETS["historical"]
COMPLEXITY = ["accuracy_rule", "random_forest", "xgboost", "svm", "stacking"]
N_BOOT = 1000


def cv() -> StratifiedKFold:
    return StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)


def impute() -> SimpleImputer:
    return SimpleImputer(strategy="constant", fill_value=0.0, add_indicator=True)


def rf() -> Pipeline:
    return Pipeline([("impute", impute()), ("rf", RandomForestClassifier(
        n_estimators=300, class_weight="balanced", random_state=SEED, n_jobs=-1))])


def svm() -> Pipeline:
    return Pipeline([("impute", impute()), ("scale", StandardScaler()),
                     ("svm", SVC(kernel="rbf", class_weight="balanced", probability=True, random_state=SEED))])


def xgb() -> Pipeline:
    return Pipeline([("impute", impute()), ("xgb", XGBClassifier(
        objective="multi:softprob", eval_metric="mlogloss", subsample=0.8, colsample_bytree=0.8,
        random_state=SEED, n_jobs=1, tree_method="hist"))])


GRIDS = {
    "random_forest": {"rf__max_depth": [5, 10, None], "rf__min_samples_leaf": [5, 20]},
    "svm": {"svm__C": [0.1, 1, 10], "svm__gamma": ["scale", 0.05]},
    "xgboost": {"xgb__max_depth": [2, 3, 4], "xgb__learning_rate": [0.05, 0.1], "xgb__n_estimators": [100, 300]},
}
META_GRID = {"max_depth": [1, 2, 3], "n_estimators": [50, 150], "learning_rate": [0.05, 0.1]}
META_WEIGHTING = ["balanced", None]


def accuracy_rule(x: pd.DataFrame) -> np.ndarray:
    z = x["z_overall_acc"].to_numpy()
    return np.where(z >= 1.0, 2, np.where(z <= -1.0, 0, 1))


def tune(name: str, x: pd.DataFrame, y: np.ndarray) -> GridSearchCV:
    maker = {"random_forest": rf, "svm": svm, "xgboost": xgb}[name]
    search = GridSearchCV(maker(), GRIDS[name], scoring="f1_macro", cv=cv(), n_jobs=-1)
    fit_params = {"xgb__sample_weight": compute_sample_weight("balanced", y)} if name == "xgboost" else {}
    return search.fit(x, y, **fit_params)


class StackedMastery(BaseEstimator, ClassifierMixin):
    """RF + SVM base learners -> XGBoost meta-learner on out-of-fold probabilities (proposal 3.5.3)."""

    def __init__(self, rf_params: dict | None = None, svm_params: dict | None = None, meta_params: dict | None = None,
                 meta_weighting: str | None = "balanced"):
        self.rf_params = rf_params
        self.svm_params = svm_params
        self.meta_params = meta_params
        self.meta_weighting = meta_weighting

    def _bases(self):
        return [rf().set_params(**(self.rf_params or {})), svm().set_params(**(self.svm_params or {}))]

    def _meta(self):
        return XGBClassifier(objective="multi:softprob", eval_metric="mlogloss", random_state=SEED, n_jobs=1,
                             tree_method="hist", **(self.meta_params or {}))

    def oof_probabilities(self, x, y) -> np.ndarray:
        return np.hstack([cross_val_predict(b, x, y, cv=cv(), method="predict_proba") for b in self._bases()])

    def fit(self, x, y, meta_x: np.ndarray | None = None):
        meta_x = self.oof_probabilities(x, y) if meta_x is None else meta_x
        weight = compute_sample_weight(self.meta_weighting, y) if self.meta_weighting else None
        self.meta_ = self._meta().fit(meta_x, y, sample_weight=weight)
        self.bases_ = [b.fit(x, y) for b in self._bases()]
        self.classes_ = np.unique(y)
        return self

    def predict_proba(self, x) -> np.ndarray:
        return self.meta_.predict_proba(np.hstack([b.predict_proba(x) for b in self.bases_]))

    def predict(self, x) -> np.ndarray:
        return self.predict_proba(x).argmax(axis=1)


def tune_stacking(x: pd.DataFrame, y: np.ndarray, rf_params: dict, svm_params: dict) -> tuple[StackedMastery, dict]:
    base = StackedMastery(rf_params, svm_params)
    meta_x = base.oof_probabilities(x, y)
    best = None
    for weighting in META_WEIGHTING:
        search = GridSearchCV(XGBClassifier(objective="multi:softprob", eval_metric="mlogloss", random_state=SEED,
                                            n_jobs=1, tree_method="hist"), META_GRID, scoring="f1_macro", cv=cv())
        weight = compute_sample_weight(weighting, y) if weighting else None
        search.fit(meta_x, y, sample_weight=weight)
        if best is None or search.best_score_ > best[1].best_score_:
            best = (weighting, search)
    weighting, search = best
    model = StackedMastery(rf_params, svm_params, search.best_params_, weighting).fit(x, y, meta_x=meta_x)
    # The meta-learner's CV score on OOF inputs is optimistic; nested CV refits the whole stack per fold.
    nested = cross_val_score(StackedMastery(rf_params, svm_params, search.best_params_, weighting), x, y,
                             cv=cv(), scoring="f1_macro")
    return model, {"meta_params": search.best_params_, "meta_weighting": weighting,
                   "cv_macro_f1_meta_optimistic": round(float(search.best_score_), 3),
                   "nested_cv_macro_f1": round(float(nested.mean()), 3)}


def metrics(y: np.ndarray, pred: np.ndarray, proba: np.ndarray | None) -> dict:
    urgent = int(((pred == 2) & (y == 2)).sum())
    out = {
        "accuracy": round(float(accuracy_score(y, pred)), 3),
        "macro_precision": round(float(precision_score(y, pred, average="macro", zero_division=0)), 3),
        "macro_recall": round(float(recall_score(y, pred, average="macro")), 3),
        "macro_f1": round(float(f1_score(y, pred, average="macro")), 3),
        "recall_per_class": dict(zip(TIERS, np.round(recall_score(y, pred, average=None), 3).tolist())),
        "needs_urgent_recall_ci95": wilson(urgent, int((y == 2).sum())),
        "confusion_matrix": {"rows_true_cols_pred": TIERS, "values": confusion_matrix(y, pred, labels=[0, 1, 2]).tolist()},
    }
    if proba is not None:
        out["roc_auc_ovr_macro"] = round(float(roc_auc_score(y, proba, multi_class="ovr", average="macro")), 3)
    return out


def urgent_recall(y: np.ndarray, pred: np.ndarray) -> float:
    return float(((pred == 2) & (y == 2)).sum() / max((y == 2).sum(), 1))


SCORES = {"macro_f1": lambda y, p: f1_score(y, p, average="macro"), "urgent_recall": urgent_recall}


def paired_bootstrap(y: np.ndarray, preds: dict) -> dict:
    rng = np.random.default_rng(SEED)
    groups = [np.where(y == c)[0] for c in np.unique(y)]
    rows = {k: [] for k in SCORES}
    for _ in range(N_BOOT):
        idx = np.concatenate([rng.choice(g, len(g)) for g in groups])
        for k, f in SCORES.items():
            rows[k].append({m: f(y[idx], p[idx]) for m, p in preds.items()})
    return {k: pd.DataFrame(v) for k, v in rows.items()}


def select(y: np.ndarray, preds: dict, criteria=("macro_f1", "urgent_recall")) -> tuple[str, dict]:
    boots = paired_bootstrap(y, preds)
    out = {"criteria": list(criteria), "best": {}, "models": {m: {} for m in COMPLEXITY}}
    for k in SCORES:
        value = {m: SCORES[k](y, p) for m, p in preds.items()}
        best = max(value, key=value.get)
        out["best"][k] = best
        for m in COMPLEXITY:
            lo, hi = np.percentile(boots[k][best] - boots[k][m], [2.5, 97.5])
            out["models"][m][k] = {"value": round(value[m], 3), "diff_from_best": round(value[best] - value[m], 3),
                                   "diff_ci95": [round(float(lo), 3), round(float(hi), 3)],
                                   "not_reliably_worse": bool(lo <= 0) or m == best}
    ok = {m: all(out["models"][m][k]["not_reliably_worse"] for k in criteria) for m in COMPLEXITY}
    return next(m for m in COMPLEXITY if ok[m]), out


def main() -> None:
    data = pd.read_csv(ARTIFACTS / "mastery_features_v1.csv")
    train, val = (data[data["split"] == s] for s in ("train", "validation"))
    xt, yt = train[FEATURES], train["mastery"].to_numpy()
    xv, yv = val[FEATURES], val["mastery"].to_numpy()

    models, tuning = {}, {}
    for name in ("random_forest", "svm", "xgboost"):
        search = tune(name, xt, yt)
        models[name] = search.best_estimator_
        tuning[name] = {"best_params": search.best_params_, "cv_macro_f1": round(float(search.best_score_), 3)}
        print(name, tuning[name])
    rf_params = {k: v for k, v in tuning["random_forest"]["best_params"].items()}
    svm_params = {k: v for k, v in tuning["svm"]["best_params"].items()}
    models["stacking"], tuning["stacking"] = tune_stacking(xt, yt, rf_params, svm_params)
    print("stacking", tuning["stacking"])

    preds = {"accuracy_rule": accuracy_rule(xv)}
    probas = {"accuracy_rule": None}
    for name, model in models.items():
        probas[name] = model.predict_proba(xv)
        preds[name] = np.asarray(model.predict(xv))
    validation = {m: metrics(yv, preds[m], probas[m]) for m in COMPLEXITY}
    chosen, selection = select(yv, preds)
    selection["macro_f1_only_rule_would_select"] = select(yv, preds, ("macro_f1",))[0]

    report = {
        "task": "3-class mastery: " + ", ".join(TIERS), "features": FEATURES,
        "class_counts": {s: d["mastery"].value_counts().sort_index().tolist() for s, d in data.groupby("split")},
        "tuning": tuning, "validation": validation, "selection": selection, "selected": chosen,
        "note": "Test split not used here (see mastery_final.py). v1 of this script scored the accuracy rule "
                "(no trained parameters) on test once; nothing from that run was used for model fitting.",
    }
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "mastery_models.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    for name, model in models.items():
        joblib.dump(model, ARTIFACTS / f"mastery_{name}_v1.joblib")
    summary = {m: {k: v[k] for k in ("macro_f1", "accuracy", "recall_per_class")} for m, v in validation.items()}
    print(json.dumps({"validation": summary, "selection": selection, "selected": chosen}, indent=2))


if __name__ == "__main__":
    main()
