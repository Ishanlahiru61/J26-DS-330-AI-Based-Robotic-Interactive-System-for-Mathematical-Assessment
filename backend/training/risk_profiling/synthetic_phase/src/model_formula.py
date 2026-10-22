"""Step 4a - evidence-weighted risk formula (baseline model).

R = sum(w_i * z_i) / sum(w_i) over the features the child has (register E3, D1b),
with weights from the meta-analysis grading (feature_evidence.md, D3). Nothing is fitted
to labels. Risk bands come from same-grade train children: >= 84th percentile = moderate,
>= 95th = high (register F1-F3). Children without sufficient data get "insufficient".

Usage: python model_formula.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

MODULE = Path(__file__).resolve().parents[1]
REPO = Path(__file__).resolve().parents[5]
ARTIFACTS = MODULE / "artifacts"
RESULTS = MODULE / "results"
SEED = 42

# Evidence weights (feature_evidence.md, Section 4: grading rule A/B).
WEIGHTS = {
    "z_f1_symbolic_comparison_acc": 0.90,
    "z_f3_arithmetic_acc": 0.87,
    "z_f5_counting_rt": 0.77,
    "z_f2_symbolic_comparison_rt": 0.75,
    "z_f7_nonsymbolic_comparison_acc": 0.69,
    "z_f6_counting_acc": 0.35,
    "z_f4_arithmetic_rt": 0.19,
    "z_f8_nonsymbolic_comparison_rt": 0.12,
}
PERCENTILES = {"moderate": 84, "high": 95}  # <= 16th and <= 5th percentile of performance (F1, F2)

# Real Decarli columns (final_dataset.md) -> formula features; F3/F4 were not measured.
DECARLI = {
    "z_number_sense_accuracy": "z_f1_symbolic_comparison_acc",
    "z_number_sense_rt": "z_f2_symbolic_comparison_rt",
    "z_enumeration_4_7_rt": "z_f5_counting_rt",
    "z_enumeration_4_7_accuracy": "z_f6_counting_acc",
    "z_nonsymbolic_comparison_accuracy": "z_f7_nonsymbolic_comparison_acc",
    "z_nonsymbolic_comparison_rt": "z_f8_nonsymbolic_comparison_rt",
}


def risk_index(features: pd.DataFrame) -> pd.DataFrame:
    z = features.reindex(columns=list(WEIGHTS))
    w = pd.Series(WEIGHTS)
    available = z.notna()
    weight_sum = available.mul(w, axis=1).sum(axis=1)
    out = pd.DataFrame(index=features.index)
    out["risk_index"] = z.fillna(0).mul(w, axis=1).sum(axis=1) / weight_sum.replace(0, np.nan)
    out["weight_coverage"] = weight_sum / w.sum()
    return out


def fit_cutoffs(scored: pd.DataFrame) -> dict:
    train = scored[scored["split"] == "train"]
    return {int(g): {band: float(np.nanpercentile(d["risk_index"], p)) for band, p in PERCENTILES.items()}
            for g, d in train.groupby("grade")}


def band(row: pd.Series, cutoffs: dict) -> str:
    if row.get("sufficient_data_area", 1) == 0 or pd.isna(row["risk_index"]):
        return "insufficient"
    c = cutoffs[int(row["grade"])]
    if row["risk_index"] >= c["high"]:
        return "high"
    return "moderate" if row["risk_index"] >= c["moderate"] else "low"


def auc_with_ci(y: np.ndarray, score: np.ndarray, n_boot: int = 2000) -> dict:
    rng = np.random.default_rng(SEED)
    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
    boots = []
    for _ in range(n_boot):
        idx = np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))])
        boots.append(roc_auc_score(y[idx], score[idx]))
    return {"auc": round(float(roc_auc_score(y, score)), 3),
            "ci95": [round(float(np.percentile(boots, 2.5)), 3), round(float(np.percentile(boots, 97.5)), 3)],
            "n": int(len(y)), "n_dd": int(y.sum())}


def screening_metrics(y: np.ndarray, flagged: np.ndarray) -> dict:
    tp, fn = int((flagged & (y == 1)).sum()), int((~flagged & (y == 1)).sum())
    tn, fp = int((~flagged & (y == 0)).sum()), int((flagged & (y == 0)).sum())
    return {"sensitivity": round(tp / (tp + fn), 3) if tp + fn else None,
            "specificity": round(tn / (tn + fp), 3) if tn + fp else None,
            "flagged_share": round(float(flagged.mean()), 3), "tp": tp, "fn": fn, "fp": fp, "tn": tn}


def evaluate(scored: pd.DataFrame) -> dict:
    report = {}
    for split in ("train", "validation"):
        d = scored[(scored["split"] == split) & (scored["band"] != "insufficient")]
        y = d["dd_label"].to_numpy()
        entry = {"auc": auc_with_ci(y, d["risk_index"].to_numpy()),
                 "moderate_or_high": screening_metrics(y, d["band"].isin(["moderate", "high"]).to_numpy()),
                 "high": screening_metrics(y, (d["band"] == "high").to_numpy()),
                 "insufficient_children": int(((scored["split"] == split) & (scored["band"] == "insufficient")).sum()),
                 "auc_by_grade": {}}
        for g, dg in d.groupby("grade"):
            if dg["dd_label"].sum() >= 2:
                entry["auc_by_grade"][int(g)] = round(float(roc_auc_score(dg["dd_label"], dg["risk_index"])), 3)
        report[split] = entry
    return report


def evaluate_real_calibration() -> dict | None:
    """AUC on the 50 real Decarli calibration children (no fitting involved; test set untouched)."""
    path = REPO / "data" / "private" / "risk_profiling" / "real_decarli_calibration_z.csv"
    if not path.exists():
        return None
    real = pd.read_csv(path).rename(columns=DECARLI)
    real[list(DECARLI.values())] = real[list(DECARLI.values())].clip(-3, 3)  # E5
    scored = risk_index(real)
    y = real["dd_label"].to_numpy()
    return {"auc": auc_with_ci(y, scored["risk_index"].to_numpy()),
            "features_available": list(DECARLI.values()),
            "note": "Real Italian children aged 7-14, 50% DD; F3/F4 not measured. Cut-offs not applied (norms differ)."}


def main() -> None:
    features = pd.read_csv(ARTIFACTS / "features_v1.csv")
    labels = pd.read_csv(ARTIFACTS / "synthetic_trials_v1_children.csv").set_index("child_id")["dd_label"]
    scored = pd.concat([features, risk_index(features)], axis=1)
    scored["dd_label"] = scored["student_id"].map(labels)
    cutoffs = fit_cutoffs(scored)
    scored["band"] = scored.apply(band, axis=1, cutoffs=cutoffs)

    report = {"model": "evidence-weighted formula v1", "weights": WEIGHTS,
              "synthetic": evaluate(scored), "real_decarli_calibration": evaluate_real_calibration()}
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "model_formula_metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (RESULTS / "formula_cutoffs.json").write_text(json.dumps(
        {"validated": False, "status": "UNVALIDATED: fitted on synthetic train children; replace with pilot norms",
         "percentiles": PERCENTILES, "cutoffs_by_grade": cutoffs}, indent=2), encoding="utf-8")
    scored[["student_id", "grade", "split", "risk_index", "weight_coverage", "band"]].round(4) \
        .to_csv(ARTIFACTS / "predictions_formula_v1.csv", index=False)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
