"""Step 4d - SHAP explanations of the XGBoost model.

SHAP (TreeExplainer, exact for tree models) splits each child's XGBoost score (log-odds)
into one contribution per input column: score = base value + sum of contributions.

1. Global: mean |SHAP| per feature and per skill on validation children. Accuracy and speed
   of the same task are grouped into one skill (they share credit; the child report will
   speak in skills, not features).
2. Agreement with the evidence: Spearman correlation between the XGBoost ranking (mean |SHAP|
   of F1-F8) and the meta-analysis weights (feature_evidence.md).
3. Checks: direction (worse z -> more risk) and additivity (contributions sum to the score).
4. One worked example: a validation DD child in the high band, contributions by skill.

Per-child SHAP values go to artifacts/ (not in Git); summaries to results/.
Usage: python explain_shap.py   (run model_xgboost.py first)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))
from model_formula import ARTIFACTS, RESULTS, WEIGHTS  # noqa: E402
from model_svm import FEATURES  # noqa: E402
from model_xgboost import MODEL_PATH  # noqa: E402

SKILLS = {  # feature -> skill (task taxonomy: T05, T10-T14, T02, T04)
    "z_f1_symbolic_comparison_acc": "number_comparison",
    "z_f2_symbolic_comparison_rt": "number_comparison",
    "z_f3_arithmetic_acc": "arithmetic",
    "z_f4_arithmetic_rt": "arithmetic",
    "z_f5_counting_rt": "counting",
    "z_f6_counting_acc": "counting",
    "z_f7_nonsymbolic_comparison_acc": "quantity_comparison",
    "z_f8_nonsymbolic_comparison_rt": "quantity_comparison",
}


def column_names(model) -> list[str]:
    """Names after imputation: F1-F8, then 'missing_<feature>' indicators."""
    names = model.named_steps["impute"].get_feature_names_out(FEATURES)
    return [n.replace("missingindicator_", "missing_") for n in names]


def shap_values(model, x: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    transformed = model.named_steps["impute"].transform(x)
    explainer = shap.TreeExplainer(model.named_steps["xgb"])
    values = explainer.shap_values(transformed)
    values = pd.DataFrame(values, columns=column_names(model), index=x.index).astype(float)  # float32 -> float
    return values, float(explainer.expected_value)


def by_skill(values: pd.DataFrame) -> pd.DataFrame:
    """Sum contributions per skill; a missing indicator belongs to its feature's skill."""
    skill_of = {c: SKILLS[c.removeprefix("missing_")] for c in values.columns}
    return values.T.groupby(skill_of).sum().T


def global_importance(values: pd.DataFrame) -> dict:
    per_feature = values.abs().mean().sort_values(ascending=False)
    per_skill = by_skill(values).abs().mean().sort_values(ascending=False)
    return {"per_feature": per_feature.round(4).to_dict(), "per_skill": per_skill.round(4).to_dict()}


def evidence_agreement(values: pd.DataFrame) -> dict:
    importance = values[FEATURES].abs().mean()
    weights = pd.Series(WEIGHTS)[FEATURES]
    rho, p = spearmanr(importance, weights)
    return {"spearman_rho": round(float(rho), 3), "p_value": round(float(p), 3), "n_features": len(FEATURES),
            "ranking_xgboost": importance.sort_values(ascending=False).index.tolist(),
            "ranking_evidence": weights.sort_values(ascending=False).index.tolist()}


def direction_check(x: pd.DataFrame, values: pd.DataFrame) -> dict:
    """Spearman correlation of each feature's z with its own SHAP value (> 0 = worse means more risk)."""
    out = {}
    for f in FEATURES:
        ok = x[f].notna()
        out[f] = round(float(spearmanr(x.loc[ok, f], values.loc[ok, f])[0]), 3)
    return out


def example_child(data: pd.DataFrame, values: pd.DataFrame, base: float, preds: pd.DataFrame) -> dict | None:
    cand = data[(data["split"] == "validation") & (data["dd_label"] == 1)]
    cand = cand.merge(preds[["student_id", "band", "risk_index"]], on="student_id")
    cand = cand[cand["band"] == "high"].sort_values("risk_index", ascending=False)
    if cand.empty:
        return None
    child = cand.iloc[0]
    idx = data.index[data["student_id"] == child["student_id"]][0]
    skills = by_skill(values.loc[[idx]]).iloc[0].sort_values(ascending=False)
    # A skill whose features are all missing was not assessed (e.g. symbolic comparison in Grade 1):
    # its SHAP value only reflects the imputed "as expected" value and must not explain the child.
    assessed = {s for f, s in SKILLS.items() if pd.notna(child[f])}
    not_assessed = sorted(set(SKILLS.values()) - assessed)
    return {"student_id": child["student_id"], "grade": int(child["grade"]), "band": child["band"],
            "probability": round(float(child["risk_index"]), 3), "base_value_log_odds": round(base, 3),
            "z_scores": {f: (None if pd.isna(child[f]) else round(float(child[f]), 2)) for f in FEATURES},
            "skills_not_assessed": not_assessed,
            "contribution_by_skill_log_odds": skills.drop(not_assessed).round(3).to_dict(),
            "contribution_by_feature_log_odds": values.loc[idx].round(3).to_dict()}


def main() -> None:
    model = joblib.load(MODEL_PATH)
    features = pd.read_csv(ARTIFACTS / "features_v1.csv")
    labels = pd.read_csv(ARTIFACTS / "synthetic_trials_v1_children.csv").set_index("child_id")["dd_label"]
    features["dd_label"] = features["student_id"].map(labels)
    preds = pd.read_csv(ARTIFACTS / "predictions_xgboost_v1.csv")

    values, base = shap_values(model, features[FEATURES])
    margin = model.named_steps["xgb"].predict(model.named_steps["impute"].transform(features[FEATURES]),
                                              output_margin=True)
    additivity_error = float(np.abs(base + values.sum(axis=1).to_numpy() - margin).max())

    val = (features["split"] == "validation") & (features["sufficient_data_area"] == 1)
    report = {
        "model": "XGBoost v1", "explainer": f"shap.TreeExplainer {shap.__version__}", "units": "log-odds",
        "children_explained": int(val.sum()), "base_value_log_odds": round(base, 3),
        "additivity_max_error": round(additivity_error, 6),
        "global_importance_validation": global_importance(values[val]),
        "agreement_with_evidence_weights": evidence_agreement(values[val]),
        "direction_check_spearman": direction_check(features.loc[val, FEATURES], values[val]),
        "example_child": example_child(features, values, base, preds),
    }
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "shap_xgboost_summary.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    out = pd.concat([features[["student_id", "grade", "split"]], values.add_prefix("shap_"),
                     by_skill(values).add_prefix("skill_")], axis=1)
    out.round(4).to_csv(ARTIFACTS / "shap_values_xgboost_v1.csv", index=False)
    print(json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
    main()
