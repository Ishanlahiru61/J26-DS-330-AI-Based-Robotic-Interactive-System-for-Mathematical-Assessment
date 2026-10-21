"""Mastery classification (proposal 3.8) - part 4: SHAP, LIME and explanation reliability.

Model: the selected mastery model (SVM, mastery_models.py). Explanations target the probability of
"Needs Urgent Assistance" (class 2), the class the proposal prioritises.

1. SHAP (primary, global + local): model-agnostic permutation explainer with a 100-child training
   background; global importance = mean |SHAP| over 150 validation children.
2. LIME (complementary, local): LimeTabularExplainer for one correctly classified validation child per tier.
3. SHAP-LIME agreement: Spearman correlation of the two explanations for the same child.
4. Stability across folds (3.8.1): the model is refitted on each of 5 CV training folds and global SHAP
   importance recomputed on the same children; mean pairwise Spearman of the rankings.
5. Direction check: Spearman of each feature's z-score with its own SHAP value (positive = worse value
   pushes toward Needs Urgent Assistance).

Explanations are associations with the model's output, not causes (proposal 1.2.5).
Usage: python mastery_explain.py   (run mastery_models.py first)
"""

from __future__ import annotations

import itertools
import json
import sys
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap
from lime.lime_tabular import LimeTabularExplainer
from scipy.stats import spearmanr
from sklearn.base import clone

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mastery_data import TIERS  # noqa: E402
from mastery_models import FEATURES, cv  # noqa: E402
from model_formula import ARTIFACTS, RESULTS, SEED  # noqa: E402

URGENT = 2
N_BACKGROUND, N_EXPLAIN, N_STABILITY = 100, 150, 80


def urgent_probability(model):
    def f(x: np.ndarray) -> np.ndarray:
        return model.predict_proba(pd.DataFrame(x, columns=FEATURES))[:, URGENT]
    return f


def shap_values(model, background: pd.DataFrame, x: pd.DataFrame) -> pd.DataFrame:
    explainer = shap.explainers.Permutation(urgent_probability(model), shap.maskers.Independent(background.to_numpy()),
                                            seed=SEED)
    values = explainer(x.to_numpy(), max_evals=2 * len(FEATURES) + 1).values
    return pd.DataFrame(values, columns=FEATURES, index=x.index)


def lime_explanation(model, train: pd.DataFrame, row: pd.Series) -> dict:
    """LIME contributions for P(Needs Urgent): local slope x the child's standardised value.

    Without discretisation LIME returns slopes on standardised features; multiplying by the child's
    standardised value turns them into contributions comparable with SHAP.
    """
    filled = train[FEATURES].fillna(0.0)                    # LIME needs complete data; 0 = as expected (imputer)
    explainer = LimeTabularExplainer(filled.to_numpy(), feature_names=FEATURES, class_names=TIERS,
                                     discretize_continuous=False, sample_around_instance=True, random_state=SEED)

    def proba(x: np.ndarray) -> np.ndarray:
        return model.predict_proba(pd.DataFrame(x, columns=FEATURES))

    value = row[FEATURES].fillna(0.0).to_numpy(dtype=float)
    exp = explainer.explain_instance(value, proba, labels=[URGENT], num_features=len(FEATURES), num_samples=5000)
    scaled = (value - explainer.scaler.mean_) / explainer.scaler.scale_
    slopes = dict(exp.local_exp[URGENT])
    return {FEATURES[i]: float(slopes.get(i, 0.0) * scaled[i]) for i in range(len(FEATURES))}


def stability(model, train: pd.DataFrame, background: pd.DataFrame, x: pd.DataFrame) -> dict:
    rankings = []
    for fold, (idx, _) in enumerate(cv().split(train[FEATURES], train["mastery"])):
        fold_model = clone(model).fit(train.iloc[idx][FEATURES], train.iloc[idx]["mastery"])
        rankings.append(shap_values(fold_model, background, x).abs().mean())
    rhos = [spearmanr(a, b)[0] for a, b in itertools.combinations(rankings, 2)]
    top5 = [set(r.sort_values(ascending=False).index[:5]) for r in rankings]
    return {"mean_pairwise_spearman": round(float(np.mean(rhos)), 3), "min_pairwise_spearman": round(float(np.min(rhos)), 3),
            "top5_features_in_all_folds": sorted(set.intersection(*top5))}


def main() -> None:
    warnings.filterwarnings("ignore")
    model = joblib.load(ARTIFACTS / "mastery_svm_v1.joblib")
    data = pd.read_csv(ARTIFACTS / "mastery_features_v1.csv")
    train, val = data[data["split"] == "train"], data[data["split"] == "validation"]
    background = train[FEATURES].sample(N_BACKGROUND, random_state=SEED)
    x = val[FEATURES].sample(N_EXPLAIN, random_state=SEED)

    values = shap_values(model, background, x)
    global_importance = values.abs().mean().sort_values(ascending=False)
    direction = {f: round(float(spearmanr(x[f], values[f], nan_policy="omit")[0]), 3) for f in FEATURES}

    pred = model.predict(val[FEATURES])
    examples = {}
    for tier in range(3):
        hit = val[(val["mastery"] == tier) & (pred == tier)]
        if hit.empty:
            continue
        row = hit.iloc[0]
        local_shap = shap_values(model, background, row[FEATURES].to_frame().T.astype(float)).iloc[0]
        lime = lime_explanation(model, train, row)
        lime_series = pd.Series(lime).reindex(FEATURES).fillna(0.0)
        examples[TIERS[tier]] = {
            "student_id": row["student_id"], "grade": int(row["grade"]),
            "p_needs_urgent": round(float(model.predict_proba(row[FEATURES].to_frame().T.astype(float))[0, URGENT]), 3),
            "shap_top5": local_shap.reindex(local_shap.abs().sort_values(ascending=False).index[:5]).round(3).to_dict(),
            "lime_top5": lime_series.reindex(lime_series.abs().sort_values(ascending=False).index[:5]).round(3).to_dict(),
            "shap_lime_spearman": round(float(spearmanr(local_shap, lime_series)[0]), 3),
        }
        print(TIERS[tier], examples[TIERS[tier]])

    report = {
        "model": "SVM (selected mastery model)", "target": "P(Needs Urgent Assistance)",
        "shap_global_mean_abs": global_importance.round(4).to_dict(),
        "direction_check_spearman": direction,
        "examples": examples,
        "stability_across_folds": stability(model, train, background, x.iloc[:N_STABILITY]),
        "note": "Associations with the model output, not causes (proposal 1.2.5). Synthetic data.",
    }
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "mastery_explanations.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("shap_global_mean_abs", "stability_across_folds")}, indent=2))


if __name__ == "__main__":
    main()
