"""Step 3 - feature engineering: preprocessed child records -> one feature row per child.

Usage:
    python build_features.py            # synthetic trials-v1 records in artifacts/

Features are z-scores against same-grade children of the same age (regression-based
norming on age_months within grade), fitted on the TRAIN split only, positive = worse,
capped at +/-3. See docs/feature_engineering.md.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

MODULE = Path(__file__).resolve().parents[1]
ARTIFACTS = MODULE / "artifacts"
RESULTS = MODULE / "results"
Z_CAP = 3.0          # register E5
MIN_NORM_N = 30      # minimum children per grade to fit a norm

AREAS = {
    "number_sense": "area_number_sense",
    "number_relationships": "area_number_relationships",
    "arithmetic_calculation": "area_arithmetic",
    "multiplicative_reasoning": "area_multiplicative",
}
# Feature -> True if a higher raw value is worse (E4).
HIGHER_IS_WORSE = {f"{a}_{m}": m == "rt" for a in AREAS.values() for m in ("acc", "rt")}
HIGHER_IS_WORSE.update({
    "f1_symbolic_comparison_acc": False, "f2_symbolic_comparison_rt": True,
    "f3_arithmetic_acc": False, "f4_arithmetic_rt": True,
    "f5_counting_rt": True, "f6_counting_acc": False,
    "f7_nonsymbolic_comparison_acc": False, "f8_nonsymbolic_comparison_rt": True,
    "f9_subitizing_acc": False, "f9_subitizing_rt": True,
    "f10_relative_error": True, "f11_attempts": True, "f12_rt_variability": True,
})
TASK_FEATURES = {  # feature -> (task types, measure)
    "f1_symbolic_comparison_acc": (["T05"], "acc"), "f2_symbolic_comparison_rt": (["T05"], "rt"),
    "f3_arithmetic_acc": (["T10", "T11", "T12", "T13", "T14"], "acc"),
    "f4_arithmetic_rt": (["T10", "T11", "T12", "T13", "T14"], "rt"),
    "f5_counting_rt": (["T02"], "rt"), "f6_counting_acc": (["T02"], "acc"),
    "f7_nonsymbolic_comparison_acc": (["T04"], "acc"), "f8_nonsymbolic_comparison_rt": (["T04"], "rt"),
    "f9_subitizing_acc": (["T01"], "acc"), "f9_subitizing_rt": (["T01"], "rt"),
}


def _pooled(rows: pd.DataFrame, measure: str) -> float:
    """Pool several task types: accuracy by summed counts (Laplace, P8); RT weighted by n_rt."""
    if rows.empty:
        return np.nan
    if measure == "acc":
        return (rows["n_correct"].sum() + 1) / (rows["n_questions"].sum() + 2)
    rt = rows.dropna(subset=["median_log_rt"])
    return np.average(rt["median_log_rt"], weights=rt["n_rt"]) if len(rt) else np.nan


def raw_features(children: pd.DataFrame, by_task: pd.DataFrame, by_area: pd.DataFrame) -> pd.DataFrame:
    """E1: one row per child with raw (not yet normed) feature values."""
    out = children[["student_id", "grade", "age_months", "split", "sufficient_data",
                    "sufficient_data_area"]].set_index("student_id").copy()
    area = by_area.assign(name=by_area["c2_area"].map(AREAS)).set_index(["student_id", "name"])
    for name in AREAS.values():
        acc = area.xs(name, level="name")["accuracy_smoothed"] if name in area.index.get_level_values(1) else None
        rt = area.xs(name, level="name")["median_log_rt"] if name in area.index.get_level_values(1) else None
        out[f"{name}_acc"] = acc
        out[f"{name}_rt"] = rt

    tasks = {sid: g for sid, g in by_task.groupby("student_id")}
    for feature, (types, measure) in TASK_FEATURES.items():
        out[feature] = [_pooled(tasks[s][tasks[s]["task_type"].isin(types)], measure) if s in tasks else np.nan
                        for s in out.index]
    risk = by_task[by_task["risk_use"] == "risk"]
    out["f10_relative_error"] = risk.groupby("student_id").apply(
        lambda g: np.average(g["mean_relative_error"].fillna(0), weights=g["n_questions"]))
    out["f11_attempts"] = risk.groupby("student_id").apply(
        lambda g: np.average(g["mean_attempts"], weights=g["n_questions"]))
    sd = risk.dropna(subset=["sd_log_rt"])
    out["f12_rt_variability"] = sd.groupby("student_id").apply(
        lambda g: np.average(g["sd_log_rt"], weights=g["n_rt"]))
    return out.reset_index()


def fit_norms(raw: pd.DataFrame, worse: dict | None = None) -> dict:
    """E2-E3: per grade and feature, x = a + b * age_months + residual, on the TRAIN split only.

    worse: feature -> True if a higher raw value is worse (default HIGHER_IS_WORSE).
    """
    worse = HIGHER_IS_WORSE if worse is None else worse
    train = raw[raw["split"] == "train"]
    norms = {}
    for grade, g in train.groupby("grade"):
        norms[int(grade)] = {}
        for feature in [f for f in worse if f in raw.columns]:
            data = g[["age_months", feature]].dropna()
            if len(data) < MIN_NORM_N or data[feature].std() == 0:
                continue
            b, a = np.polyfit(data["age_months"], data[feature], 1)
            resid = data[feature] - (a + b * data["age_months"])
            norms[int(grade)][feature] = {"intercept": float(a), "age_slope": float(b),
                                          "sd": float(resid.std(ddof=2)), "n": int(len(data))}
    return norms


def apply_norms(raw: pd.DataFrame, norms: dict, worse: dict | None = None) -> pd.DataFrame:
    """E2, E4, E5, E6: z = sign * (x - expected for grade and age) / sd, capped; missing stays missing."""
    worse = HIGHER_IS_WORSE if worse is None else worse
    out = raw[["student_id", "grade", "age_months", "split", "sufficient_data", "sufficient_data_area"]].copy()
    for feature, worse_high in worse.items():
        if feature not in raw.columns:
            continue
        z = pd.Series(np.nan, index=raw.index)
        for grade, idx in raw.groupby("grade").groups.items():
            norm = norms.get(int(grade), {}).get(feature)
            if norm is None:
                continue
            expected = norm["intercept"] + norm["age_slope"] * raw.loc[idx, "age_months"]
            z[idx] = (raw.loc[idx, feature] - expected) / norm["sd"]
        out[f"z_{feature}"] = (z if worse_high else -z).clip(-Z_CAP, Z_CAP)
    z_cols = [c for c in out.columns if c.startswith("z_")]
    out["n_features_available"] = out[z_cols].notna().sum(axis=1)
    return out


def hedges_g(worse: np.ndarray, better: np.ndarray) -> float:
    n1, n2 = len(worse), len(better)
    if n1 < 2 or n2 < 2:
        return float("nan")
    pooled = math.sqrt(((n1 - 1) * worse.var(ddof=1) + (n2 - 1) * better.var(ddof=1)) / (n1 + n2 - 2))
    return (worse.mean() - better.mean()) / pooled * (1 - 3 / (4 * (n1 + n2) - 9))


def check_features(features: pd.DataFrame, labels: pd.Series) -> pd.DataFrame:
    rows = []
    data = features.assign(dd=features["student_id"].map(labels))
    for col in [c for c in features.columns if c.startswith("z_")]:
        row = {"feature": col, "missing_share": round(float(features[col].isna().mean()), 3)}
        for split in ("train", "validation", "test"):
            d = data[(data["split"] == split)].dropna(subset=[col])
            row[f"g_{split}"] = round(hedges_g(d.loc[d.dd == 1, col].to_numpy(),
                                               d.loc[d.dd == 0, col].to_numpy()), 3)
        typical_train = data[(data["split"] == "train") & (data["dd"] == 0)][col].dropna()
        row["typical_train_mean"] = round(float(typical_train.mean()), 3)
        row["share_capped"] = round(float((features[col].abs() == Z_CAP).mean()), 4)
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    children = pd.read_csv(ARTIFACTS / "preprocessed_children_v1.csv")
    by_task = pd.read_csv(ARTIFACTS / "preprocessed_child_task_v1.csv")
    by_area = pd.read_csv(ARTIFACTS / "preprocessed_child_area_v1.csv")
    raw = raw_features(children, by_task, by_area)
    norms = fit_norms(raw)
    features = apply_norms(raw, norms)

    features.round(4).to_csv(ARTIFACTS / "features_v1.csv", index=False)
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "feature_norms_v1.json").write_text(json.dumps(norms, indent=1), encoding="utf-8")
    labels = pd.read_csv(ARTIFACTS / "synthetic_trials_v1_children.csv").set_index("child_id")["dd_label"]
    check = check_features(features, labels)
    check.to_csv(RESULTS / "feature_check.csv", index=False)
    print(f"{len(features)} children x {sum(c.startswith('z_') for c in features.columns)} features")
    print(check.to_string(index=False))


if __name__ == "__main__":
    main()
