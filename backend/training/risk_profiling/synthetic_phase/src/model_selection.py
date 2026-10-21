"""Step 4f - model selection.

Compares the five models (4a-4e) on the same children and chooses one with a written rule:

    Among the models whose AUC is not reliably worse than the best model, choose the simplest
    and most explainable one (order in COMPLEXITY).

"Not reliably worse" = the 95% paired-bootstrap CI of (AUC best - AUC model) includes 0. The test is
paired (the same children are resampled for both models) and stratified (DD and typical children
resampled separately). Primary data: validation children with sufficient data. Secondary: train
(out-of-fold for trained models) + validation, for more DD children. The test split is not used.

Writes results/model_selection.json and config/selected_model.json.
Usage: python model_selection.py   (run the five model scripts first)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from model_formula import ARTIFACTS, MODULE, PERCENTILES, RESULTS, SEED  # noqa: E402

# Simplest / most explainable first (reasons in docs/modelling.md, 4f).
COMPLEXITY = ["formula", "logistic", "xgboost", "random_forest", "svm"]
EXPLANATION = {
    "formula": "exact: weight x z per feature, weights from published evidence; no training",
    "logistic": "exact: learned weight x z per feature",
    "xgboost": "SHAP (exact for trees), 100 depth-1 trees",
    "random_forest": "approximate (permutation importance); 500 trees",
    "svm": "none per child (RBF kernel)",
}
N_BOOT = 2000


def load_scores() -> pd.DataFrame:
    labels = pd.read_csv(ARTIFACTS / "synthetic_trials_v1_children.csv").set_index("child_id")["dd_label"]
    base = pd.read_csv(ARTIFACTS / "features_v1.csv")[["student_id", "grade", "split", "sufficient_data_area"]]
    base["dd_label"] = base["student_id"].map(labels)
    for name in COMPLEXITY:
        pred = pd.read_csv(ARTIFACTS / f"predictions_{name}_v1.csv")[["student_id", "risk_index"]]
        base = base.merge(pred.rename(columns={"risk_index": name}), on="student_id")
    return base[(base["split"] != "test") & (base["sufficient_data_area"] == 1)].reset_index(drop=True)


def paired_bootstrap(y: np.ndarray, scores: pd.DataFrame, n_boot: int = N_BOOT) -> pd.DataFrame:
    """AUC of every model on the same stratified resamples of children."""
    rng = np.random.default_rng(SEED)
    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
    rows = []
    for _ in range(n_boot):
        idx = np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))])
        rows.append({m: roc_auc_score(y[idx], scores[m].to_numpy()[idx]) for m in scores.columns})
    return pd.DataFrame(rows)


def compare(data: pd.DataFrame) -> dict:
    y = data["dd_label"].to_numpy()
    scores = data[COMPLEXITY]
    auc = {m: float(roc_auc_score(y, scores[m])) for m in COMPLEXITY}
    best = max(auc, key=auc.get)
    boots = paired_bootstrap(y, scores)
    out = {"n": int(len(y)), "n_dd": int(y.sum()), "best_auc_model": best, "models": {}}
    for m in COMPLEXITY:
        diff = boots[best] - boots[m]
        lo, hi = np.percentile(diff, [2.5, 97.5])
        out["models"][m] = {
            "auc": round(auc[m], 3),
            "auc_ci95": [round(float(v), 3) for v in np.percentile(boots[m], [2.5, 97.5])],
            "diff_from_best": round(auc[best] - auc[m], 3),
            "diff_ci95": [round(float(lo), 3), round(float(hi), 3)],
            "not_reliably_worse": bool(lo <= 0 <= hi) or m == best,
        }
    return out


def select(primary: dict) -> str:
    """Simplest model that is not reliably worse than the best (primary comparison)."""
    return next(m for m in COMPLEXITY if primary["models"][m]["not_reliably_worse"])


def real_decarli() -> dict:
    out = {}
    for m in COMPLEXITY:
        report = json.loads((RESULTS / f"model_{m}_metrics.json").read_text(encoding="utf-8"))
        real = report.get("real_decarli_calibration") or {}
        if real:
            out[m] = {"auc": real["auc"]["auc"], "ci95": real["auc"]["ci95"]}
    return out


def main() -> None:
    data = load_scores()
    primary = compare(data[data["split"] == "validation"])
    secondary = compare(data)
    chosen = select(primary)

    screening = json.loads((RESULTS / "sensitivity_analysis.json").read_text(encoding="utf-8"))
    report = {
        "rule": "Among models not reliably worse than the best (95% paired-bootstrap CI of the AUC difference "
                "includes 0), choose the simplest and most explainable (order: " + " > ".join(COMPLEXITY) + ")",
        "selected": chosen,
        "validation": primary,
        "train_oof_plus_validation": secondary,
        "real_decarli_calibration": real_decarli(),
        "explanation": EXPLANATION,
        "note": "Synthetic data; test split and the 14 real Decarli test children not used (Step 6).",
    }
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "model_selection.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    selected = {
        "validated": False,
        "status": "UNVALIDATED: chosen on synthetic data; confirm in Step 6 and re-estimate on pilot data",
        "model": chosen,
        "script": f"src/model_{chosen}.py",
        "bands_percentiles": PERCENTILES,
        "screening_flag_worst_pct": 100 - screening["screening_percentile_formula"] if chosen == "formula" else None,
        "cutoffs_file": "results/formula_cutoffs.json" if chosen == "formula" else None,
        "insufficient_rule": "sufficient_data_area == 0 -> 'insufficient'",
    }
    (MODULE / "config" / "selected_model.json").write_text(json.dumps(selected, indent=2), encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("selected", "validation", "train_oof_plus_validation",
                                             "real_decarli_calibration")}, indent=2))


if __name__ == "__main__":
    main()
