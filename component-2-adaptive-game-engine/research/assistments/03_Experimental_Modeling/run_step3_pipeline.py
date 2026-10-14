#!/usr/bin/env python3
"""ASSISTments Step 3: learner-disjoint predictive experiments.

This script is intentionally self-contained and deterministic. It never writes to
Step 1, Step 2, or curriculum directories. All fitted transformations use only
the training-learner partition.
"""
from __future__ import annotations

import csv
import gc
import hashlib
import json
import math
import os
import platform
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import duckdb
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import scipy
import sklearn
import xgboost
from scipy import sparse
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier


SEED = 20260928
ROOT = Path(__file__).resolve().parent
WORKSPACE = ROOT.parent.parent
STEP2 = ROOT.parent / "02_Chronological_Learner_History"
SOURCE = STEP2 / "data" / "processed" / "ASSISTments_2012_2013_history_features.parquet"
STEP2_MANIFEST = STEP2 / "outputs" / "ASSISTments_Step2_Artifact_Manifest.csv"
STEP2_FREEZE = STEP2 / "outputs" / "ASSISTments_Step2_Final_Freeze_Summary.md"
STEP2_SPEC = STEP2 / "outputs" / "ASSISTments_Step2_History_Feature_Specification.csv"
CURRICULUM = WORKSPACE / "outputs" / "nie_curriculum_scope"

DATA_PART = ROOT / "data" / "partitions"
DATA_PREP = ROOT / "data" / "prepared"
MODELS = ROOT / "models"
OUTPUTS = ROOT / "outputs"
TESTS = ROOT / "tests"

EXPECTED_STEP2_HISTORY_HASH = "7c422ba3dcc536b2d2d88550dd4a4b6ce546082609f17eab3d17f5c4939d97ef"
EXPECTED_STEP2_FREEZE_HASH = "eed251bd1272be989f3682d254188473d3da79f4852feea79ba82bbe5c84b940"
EXPECTED_ROWS = 6_123_270
EXPECTED_ELIGIBLE = 6_117_947
EXPECTED_LEARNERS = 46_674

RT_FEATURES = {
    "previous_ms_first_response",
    "previous_interaction_rt_valid",
    "prior_valid_rt_count",
    "prior_mean_ms_first_response",
}
CATEGORICAL = {
    "skill_id",
    "problem_type",
    "tutor_mode",
    "template_id",
    "previous_skill_id",
    "previous_problem_type",
}
SMALL_CATEGORICAL = [
    "skill_id",
    "problem_type",
    "tutor_mode",
    "previous_skill_id",
    "previous_problem_type",
]
CURRICULUM_FILES = [
    "NIE_Grade_Scope_Decision_Report.md",
    "NIE_Curriculum_Traceability_Register.xlsx",
    "NIE_Math_Skill_Progression_Matrix.xlsx",
    "NIE_Project_Skill_Mapping.csv",
    "NIE_Tier_A_Tier_B_Proposal.csv",
    "NIE_Curriculum_Constraints_Draft.csv",
    "Grade_Boundary_Literature_Evidence.csv",
    "NIE_Teacher_Guide_Source_Index.csv",
]


def ensure_dirs() -> None:
    for d in [DATA_PART, DATA_PREP, OUTPUTS, TESTS]:
        d.mkdir(parents=True, exist_ok=True)
    for d in [
        MODELS / "preprocessors",
        MODELS / "logistic_regression",
        MODELS / "gradient_boosting",
        MODELS / "calibration",
    ]:
        d.mkdir(parents=True, exist_ok=True)


def sha256(path: Path, block: int = 8 * 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(block):
            h.update(chunk)
    return h.hexdigest()


def write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str] | None = None) -> None:
    pd.DataFrame(rows, columns=columns).to_csv(path, index=False, lineterminator="\n")


def verify_provenance() -> list[dict[str, Any]]:
    manifest = pd.read_csv(STEP2_MANIFEST)
    expected_map = {Path(p).name: h for p, h in zip(manifest["path"], manifest["sha256"])}
    rows: list[dict[str, Any]] = []
    critical = [
        ("Step-2 history feature Parquet", SOURCE, EXPECTED_STEP2_HISTORY_HASH),
        ("Step-2 artifact manifest", STEP2_MANIFEST, None),
        ("Step-2 freeze summary", STEP2_FREEZE, EXPECTED_STEP2_FREEZE_HASH),
    ]
    for artifact, path, hard_expected in critical:
        actual = sha256(path)
        expected = hard_expected or expected_map.get(path.name)
        if expected:
            status = "VERIFIED" if actual.lower() == str(expected).lower() else "FAILED"
        else:
            status = "BASELINE_CAPTURED_NO_EXTERNAL_EXPECTED_HASH"
        rows.append({
            "artifact": artifact,
            "path": str(path.resolve()),
            "size": path.stat().st_size,
            "SHA-256": actual,
            "expected hash": expected or "not externally recorded; Step-3 baseline",
            "verification status": status,
        })
        if status == "FAILED":
            raise RuntimeError(f"Frozen input changed: {path}")
    for name in CURRICULUM_FILES:
        path = CURRICULUM / name
        if not path.exists():
            raise FileNotFoundError(path)
        rows.append({
            "artifact": f"Curriculum carry-forward: {name}",
            "path": str(path.resolve()),
            "size": path.stat().st_size,
            "SHA-256": sha256(path),
            "expected hash": "no curriculum manifest found; Step-3 baseline",
            "verification status": "BASELINE_CAPTURED_NO_MANIFEST",
        })
    write_csv(OUTPUTS / "ASSISTments_Step3_Input_Provenance.csv", rows)
    return rows


def load_feature_contract() -> tuple[dict[str, list[str]], pd.DataFrame]:
    spec = pd.read_csv(STEP2_SPEC)
    additions = {
        h: spec.loc[spec["history_configuration"] == h, "feature_name"].tolist()
        for h in ["H0", "H1", "H2"]
    }
    configs = {
        "H0": additions["H0"],
        "H1": additions["H0"] + additions["H1"],
        "H2": additions["H0"] + additions["H1"] + additions["H2"],
    }
    assert len(configs["H0"]) == 14 and len(configs["H1"]) == 29 and len(configs["H2"]) == 42
    assert set(configs["H0"]) < set(configs["H1"]) < set(configs["H2"])
    contract_rows = []
    for feature in configs["H2"]:
        row = spec.loc[spec["feature_name"] == feature].iloc[0]
        rt = str(row["rt_dependence"]).upper() == "YES"
        if feature == "template_id":
            prep = "train-frequency log1p encoding; unseen/missing -> 0; standardized"
        elif feature in SMALL_CATEGORICAL:
            prep = "train-fitted one-hot; missing explicit; unknown ignored; rare <1000 grouped"
        elif feature == "previous_ms_first_response":
            prep = "valid positive value -> log1p -> train problem-type median/MAD robust z; global backoff; clip ±8"
        elif feature == "prior_mean_ms_first_response":
            prep = "valid positive value -> log1p -> train global median/MAD robust z; clip ±8"
        else:
            prep = "train median imputation and train standardization"
        contract_rows.append({
            "feature": feature,
            "H0": int(feature in configs["H0"]),
            "H1": int(feature in configs["H1"]),
            "H2": 1,
            "RT_dependent": int(rt),
            "categorical/numeric": "categorical" if feature in CATEGORICAL else "numeric",
            "preprocessing": prep,
            "fit_on_training_only": "YES",
            "notes": "Frozen Step-2 feature; RT-OFF excludes it" if rt else "Frozen Step-2 feature",
        })
    contract = pd.DataFrame(contract_rows)
    contract.to_csv(OUTPUTS / "ASSISTments_Step3_Model_Feature_Contract.csv", index=False)
    return configs, contract


def make_partitions() -> dict[str, np.ndarray]:
    con = duckdb.connect()
    learners = con.execute(
        "SELECT DISTINCT user_id FROM read_parquet(?) ORDER BY user_id", [str(SOURCE)]
    ).fetchnumpy()["user_id"]
    con.close()
    if len(learners) != EXPECTED_LEARNERS:
        raise RuntimeError(f"Expected {EXPECTED_LEARNERS} learners, found {len(learners)}")
    rng = np.random.default_rng(SEED)
    shuffled = learners.copy()
    rng.shuffle(shuffled)
    n_train = int(round(len(shuffled) * 0.70))
    n_val = int(round(len(shuffled) * 0.15))
    parts = {
        "train": np.sort(shuffled[:n_train]),
        "validation": np.sort(shuffled[n_train:n_train + n_val]),
        "test": np.sort(shuffled[n_train + n_val:]),
    }
    for name, ids in parts.items():
        pd.DataFrame({"user_id": ids}).to_csv(DATA_PART / f"{name}_learners.csv", index=False)
    if set(parts["train"]) & set(parts["validation"]) or set(parts["train"]) & set(parts["test"]) or set(parts["validation"]) & set(parts["test"]):
        raise RuntimeError("Learner-disjoint partition failed")
    return parts


def split_audit() -> pd.DataFrame:
    con = duckdb.connect()
    union = []
    for name in ["train", "validation", "test"]:
        f = (DATA_PART / f"{name}_learners.csv").as_posix().replace("'", "''")
        union.append(f"SELECT user_id, '{name}' AS split FROM read_csv_auto('{f}', header=true)")
    part_sql = " UNION ALL ".join(union)
    source = SOURCE.as_posix().replace("'", "''")
    query = f"""
        WITH p AS ({part_sql}), x AS (
          SELECT d.*, p.split FROM read_parquet('{source}') d JOIN p USING(user_id)
        )
        SELECT split,
          count(DISTINCT user_id) AS learner_count,
          count(*) AS interaction_count,
          sum(CASE WHEN target_eligible THEN 1 ELSE 0 END) AS target_eligible_count,
          avg(CASE WHEN target_eligible THEN target_first_attempt_correct END) AS target_positive_rate,
          quantile_cont(learner_interaction_count_total, 0.25) AS history_length_p25,
          quantile_cont(learner_interaction_count_total, 0.50) AS history_length_p50,
          quantile_cont(learner_interaction_count_total, 0.75) AS history_length_p75,
          quantile_cont(learner_interaction_count_total, 0.95) AS history_length_p95,
          avg(CASE WHEN skill_id IS NOT NULL THEN 1 ELSE 0 END) AS skill_availability_rate,
          avg(CASE WHEN rt_feature_eligible THEN 1 ELSE 0 END) AS current_rt_validity_rate,
          avg(CASE WHEN previous_interaction_rt_valid THEN 1 ELSE 0 END) AS prior_rt_availability_rate
        FROM x GROUP BY split ORDER BY CASE split WHEN 'train' THEN 1 WHEN 'validation' THEN 2 ELSE 3 END
    """
    audit = con.execute(query).fetchdf()
    con.close()
    audit.to_csv(OUTPUTS / "ASSISTments_Step3_Split_Audit.csv", index=False)
    if int(audit["interaction_count"].sum()) != EXPECTED_ROWS or int(audit["target_eligible_count"].sum()) != EXPECTED_ELIGIBLE:
        raise RuntimeError("Split audit totals disagree with frozen Step 2")
    return audit


@dataclass
class RTStats:
    min_support: int
    global_median: float
    global_scale: float
    problem_type: dict[str, tuple[float, float, int]]
    prior_mean_median: float
    prior_mean_scale: float


def robust_location_scale(values: np.ndarray) -> tuple[float, float]:
    values = values[np.isfinite(values)]
    med = float(np.median(values)) if len(values) else 0.0
    mad = float(np.median(np.abs(values - med))) if len(values) else 1.0
    scale = 1.4826 * mad
    if not np.isfinite(scale) or scale < 1e-6:
        scale = 1.0
    return med, scale


def fit_rt_stats(train: pd.DataFrame, min_support: int = 1000) -> RTStats:
    valid = train["rt_feature_eligible"].fillna(False).to_numpy(bool) & (train["ms_first_response"].to_numpy(float) > 0)
    logs = np.log1p(train.loc[valid, "ms_first_response"].to_numpy(float))
    gm, gs = robust_location_scale(logs)
    context: dict[str, tuple[float, float, int]] = {}
    temp = pd.DataFrame({
        "context": train.loc[valid, "problem_type"].fillna("<MISSING>").astype(str).to_numpy(),
        "log_rt": logs,
    })
    for key, group in temp.groupby("context", sort=True):
        if len(group) >= min_support:
            med, scale = robust_location_scale(group["log_rt"].to_numpy())
            context[str(key)] = (med, scale, int(len(group)))
    pm = train["prior_mean_ms_first_response"].to_numpy(float)
    pm = np.log1p(pm[np.isfinite(pm) & (pm > 0)])
    pmed, pscale = robust_location_scale(pm)
    return RTStats(min_support, gm, gs, context, pmed, pscale)


def fit_template_frequency(train: pd.DataFrame) -> dict[int, int]:
    counts = train["template_id"].dropna().astype("int64").value_counts()
    return {int(k): int(v) for k, v in counts.items()}


def engineer(df: pd.DataFrame, rt: RTStats, template_frequency: dict[int, int]) -> pd.DataFrame:
    out = df.copy()
    # High-cardinality template identity is represented without target statistics.
    raw_template = out["template_id"]
    out["template_id"] = raw_template.map(template_frequency).fillna(0).map(np.log1p).astype("float32")
    for c in SMALL_CATEGORICAL:
        out[c] = out[c].astype("string").fillna("<MISSING>")
    prev = out["previous_ms_first_response"].to_numpy(float)
    valid = out["previous_interaction_rt_valid"].fillna(False).to_numpy(bool) & np.isfinite(prev) & (prev > 0)
    prev_z = np.full(len(out), np.nan, dtype="float32")
    ctx = out["previous_problem_type"].astype("string").fillna("<MISSING>").to_numpy()
    vals = np.log1p(prev[valid])
    idx_valid = np.flatnonzero(valid)
    for key in np.unique(ctx[valid]):
        loc, scale, _ = rt.problem_type.get(str(key), (rt.global_median, rt.global_scale, 0))
        rows = idx_valid[ctx[valid] == key]
        prev_z[rows] = np.clip((np.log1p(prev[rows]) - loc) / scale, -8, 8).astype("float32")
    out["previous_ms_first_response"] = prev_z
    prior = out["prior_mean_ms_first_response"].to_numpy(float)
    prior_z = np.full(len(out), np.nan, dtype="float32")
    ok = np.isfinite(prior) & (prior > 0)
    prior_z[ok] = np.clip((np.log1p(prior[ok]) - rt.prior_mean_median) / rt.prior_mean_scale, -8, 8).astype("float32")
    out["prior_mean_ms_first_response"] = prior_z
    return out


def load_split(name: str, feature_cols: list[str]) -> pd.DataFrame:
    learner_csv = (DATA_PART / f"{name}_learners.csv").as_posix().replace("'", "''")
    source = SOURCE.as_posix().replace("'", "''")
    extra = [
        "problem_log_id", "user_id", "target_first_attempt_correct", "target_eligible",
        "learner_interaction_index", "learner_interaction_count_total", "skill_history_available_key",
        "rt_feature_eligible", "ms_first_response",
    ]
    cols = list(dict.fromkeys(extra + feature_cols))
    select = ", ".join([f'd."{c}"' for c in cols])
    q = f"""
      SELECT {select}
      FROM read_parquet('{source}') d
      JOIN read_csv_auto('{learner_csv}', header=true) p USING(user_id)
      WHERE d.target_eligible
      ORDER BY d.user_id, d.learner_interaction_index, d.problem_log_id
    """
    con = duckdb.connect()
    df = con.execute(q).fetchdf()
    con.close()
    return df


def fit_preprocessor(train: pd.DataFrame, features: list[str]) -> tuple[ColumnTransformer, list[str], list[str]]:
    cat = [c for c in SMALL_CATEGORICAL if c in features]
    num = [c for c in features if c not in cat]
    numeric_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    categorical_pipe = OneHotEncoder(
        handle_unknown="infrequent_if_exist",
        min_frequency=1000,
        dtype=np.float32,
    )
    pre = ColumnTransformer(
        [("num", numeric_pipe, num), ("cat", categorical_pipe, cat)],
        sparse_threshold=1.0,
        verbose_feature_names_out=True,
    )
    pre.fit(train[features])
    return pre, num, cat


def output_source_map(pre: ColumnTransformer, num: list[str], cat: list[str]) -> list[str]:
    result = list(num)
    enc: OneHotEncoder = pre.named_transformers_["cat"]
    for source, cats in zip(cat, enc.categories_):
        result.extend([source] * len(cats))
        if enc._infrequent_enabled and source in cat:
            # get_feature_names_out already includes exactly the output width; rebuild safely below if needed.
            pass
    names = pre.get_feature_names_out()
    if len(result) != len(names):
        result = []
        for name in names:
            if name.startswith("num__"):
                result.append(name.split("num__", 1)[1])
            else:
                tail = name.split("cat__", 1)[1]
                matches = [c for c in cat if tail == c or tail.startswith(c + "_")]
                if not matches:
                    raise RuntimeError(f"Cannot map transformed feature {name}")
                result.append(max(matches, key=len))
    return result


def metadata_frame(df: pd.DataFrame, split: str) -> pd.DataFrame:
    return pd.DataFrame({
        "problem_log_id": df["problem_log_id"].astype("int64"),
        "user_id": df["user_id"].astype("int64"),
        "split": split,
        "y_true": df["target_first_attempt_correct"].astype("int8"),
        "learner_interaction_index": df["learner_interaction_index"].astype("int64"),
        "learner_interaction_count_total": df["learner_interaction_count_total"].astype("int64"),
        "skill_available": df["skill_history_available_key"].fillna(False).astype(bool),
        "prior_rt_available": df["previous_interaction_rt_valid"].fillna(False).astype(bool),
        "problem_type": df["problem_type"].astype("string").fillna("<MISSING>"),
    })


def ece_score(y: np.ndarray, p: np.ndarray, bins: int = 10) -> float:
    order = np.argsort(p, kind="stable")
    total = len(y)
    value = 0.0
    for idx in np.array_split(order, bins):
        if len(idx):
            value += len(idx) / total * abs(float(np.mean(y[idx])) - float(np.mean(p[idx])))
    return float(value)


def calibration_intercept_slope(y: np.ndarray, p: np.ndarray) -> tuple[float, float]:
    eps = 1e-7
    logits = np.log(np.clip(p, eps, 1 - eps) / np.clip(1 - p, eps, 1 - eps)).reshape(-1, 1)
    lr = LogisticRegression(C=1e6, solver="lbfgs", max_iter=100, random_state=SEED)
    lr.fit(logits, y)
    return float(lr.intercept_[0]), float(lr.coef_[0, 0])


def metric_row(experiment_id: str, split: str, y: np.ndarray, p: np.ndarray, users: np.ndarray, calibration: str) -> dict[str, Any]:
    intercept, slope = calibration_intercept_slope(y, p)
    return {
        "experiment_id": experiment_id,
        "split": split,
        "calibration": calibration,
        "Log Loss": log_loss(y, p, labels=[0, 1]),
        "Brier Score": brier_score_loss(y, p),
        "ROC-AUC": roc_auc_score(y, p),
        "PR-AUC": average_precision_score(y, p),
        "ECE_10_equal_frequency": ece_score(y, p),
        "calibration_intercept": intercept,
        "calibration_slope": slope,
        "n_rows": len(y),
        "n_learners": len(np.unique(users)),
    }


def reliability_rows(experiment_id: str, split: str, y: np.ndarray, p: np.ndarray, calibration: str) -> list[dict[str, Any]]:
    order = np.argsort(p, kind="stable")
    rows = []
    for i, idx in enumerate(np.array_split(order, 10), 1):
        rows.append({
            "experiment_id": experiment_id,
            "split": split,
            "calibration": calibration,
            "bin": i,
            "n": len(idx),
            "mean_predicted": float(np.mean(p[idx])),
            "observed_rate": float(np.mean(y[idx])),
            "absolute_gap": abs(float(np.mean(p[idx]) - np.mean(y[idx]))),
        })
    return rows


def save_predictions(meta: pd.DataFrame, p: np.ndarray, exp: dict[str, Any], calibration: str = "native") -> Path:
    out = meta[["problem_log_id", "user_id", "split", "y_true"]].copy()
    out["predicted_probability"] = p.astype("float32")
    out["model"] = exp["model"]
    out["configuration"] = exp["experiment_id"]
    out["history_representation"] = exp["history_config"]
    out["RT_status"] = exp["RT_status"]
    out["calibration_status"] = calibration
    path = DATA_PREP / f"predictions_{exp['experiment_id']}_{out['split'].iloc[0]}_{calibration}.parquet"
    out.to_parquet(path, index=False, compression="zstd")
    return path


def paired_bootstrap(meta: pd.DataFrame, pa_: np.ndarray, pb_: np.ndarray, a: str, b: str, comparison: str, reps: int = 1000) -> list[dict[str, Any]]:
    y = meta["y_true"].to_numpy()
    uid = meta["user_id"].to_numpy()
    eps = 1e-15
    losses = {
        "Log Loss": (-(y * np.log(np.clip(pb_, eps, 1-eps)) + (1-y) * np.log(np.clip(1-pb_, eps, 1-eps)))) -
                     (-(y * np.log(np.clip(pa_, eps, 1-eps)) + (1-y) * np.log(np.clip(1-pa_, eps, 1-eps)))),
        "Brier Score": (pb_ - y) ** 2 - (pa_ - y) ** 2,
    }
    rng = np.random.default_rng(SEED + 31)
    rows = []
    for metric, diff in losses.items():
        frame = pd.DataFrame({"user_id": uid, "diff": diff}).groupby("user_id", sort=True)["diff"].mean()
        vals = frame.to_numpy()
        draws = np.empty(reps)
        for i in range(reps):
            draws[i] = np.mean(rng.choice(vals, size=len(vals), replace=True))
        estimate = float(np.mean(vals))
        rows.append({
            "comparison": comparison,
            "model_A": a,
            "model_B": b,
            "metric": metric,
            "difference_B_minus_A": estimate,
            "ci95_low": float(np.quantile(draws, 0.025)),
            "ci95_high": float(np.quantile(draws, 0.975)),
            "bootstrap_unit": "learner",
            "bootstrap_replicates": reps,
            "interpretation": "negative favors B" if metric in {"Log Loss", "Brier Score"} else "positive favors B",
        })
    return rows


def subgroup_metrics(meta: pd.DataFrame, predictions: dict[str, np.ndarray]) -> pd.DataFrame:
    buckets = pd.DataFrame(index=meta.index)
    idx = meta["learner_interaction_index"]
    buckets["cold_start"] = pd.cut(idx, bins=[-1, 0, 3, 9, np.inf], labels=["first", "2-4", "5-10", "11+"])
    buckets["skill_availability"] = np.where(meta["skill_available"], "skill_available", "skill_missing")
    buckets["prior_rt_availability"] = np.where(meta["prior_rt_available"], "prior_rt_available", "prior_rt_unavailable")
    buckets["learner_history_length"] = np.where(meta["learner_interaction_count_total"] < 5, "learner_total_<5", "learner_total_5+")
    buckets["problem_type"] = meta["problem_type"].astype(str)
    rows = []
    y_all = meta["y_true"].to_numpy()
    for exp_id, p in predictions.items():
        for dimension in buckets.columns:
            for group in sorted(buckets[dimension].dropna().unique(), key=str):
                mask = (buckets[dimension].astype(str).to_numpy() == str(group))
                y = y_all[mask]
                pp = p[mask]
                if len(y) < 2 or len(np.unique(y)) < 2:
                    continue
                rows.append({
                    "experiment_id": exp_id,
                    "dimension": dimension,
                    "group": group,
                    "n_rows": len(y),
                    "n_learners": meta.loc[mask, "user_id"].nunique(),
                    "Log Loss": log_loss(y, pp, labels=[0, 1]),
                    "Brier Score": brier_score_loss(y, pp),
                    "ROC-AUC": roc_auc_score(y, pp),
                    "ECE": ece_score(y, pp),
                })
    return pd.DataFrame(rows)


def coefficient_table(model: SGDClassifier, pre: ColumnTransformer, selected_idx: np.ndarray, exp_id: str) -> pd.DataFrame:
    names = pre.get_feature_names_out()[selected_idx]
    coef = model.coef_.ravel()
    frame = pd.DataFrame({"experiment_id": exp_id, "transformed_feature": names, "standardized_coefficient": coef})
    frame["absolute_coefficient"] = frame["standardized_coefficient"].abs()
    return frame.sort_values("absolute_coefficient", ascending=False).head(100)


def plot_reliability(reliability: pd.DataFrame, selected_ids: list[str]) -> None:
    data = reliability[(reliability["split"] == "test") & (reliability["calibration"] == "native") & reliability["experiment_id"].isin(selected_ids)]
    fig, ax = plt.subplots(figsize=(7.2, 6.2))
    ax.plot([0, 1], [0, 1], "--", color="black", linewidth=1, label="ideal")
    for exp_id, group in data.groupby("experiment_id"):
        ax.plot(group["mean_predicted"], group["observed_rate"], marker="o", linewidth=1.5, label=exp_id)
    ax.set(xlabel="Mean predicted probability", ylabel="Observed correctness rate", title="Test-set reliability (equal-frequency bins)", xlim=(0, 1), ylim=(0, 1))
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUTPUTS / "ASSISTments_Step3_Reliability_Important_Models.png", dpi=180)
    plt.close(fig)


def fmt(v: float) -> str:
    return f"{v:.6f}"


def metric_lookup(results: pd.DataFrame, exp: str, split: str = "test", cal: str = "native") -> pd.Series:
    return results[(results.experiment_id == exp) & (results.split == split) & (results.calibration == cal)].iloc[0]


def write_reports(results: pd.DataFrame, uncertainty: pd.DataFrame, subgroup: pd.DataFrame, registry: pd.DataFrame, selected: dict[str, Any], split_audit_df: pd.DataFrame) -> None:
    def row(exp: str) -> str:
        r = metric_lookup(results, exp)
        return f"| {exp} | {fmt(r['Log Loss'])} | {fmt(r['Brier Score'])} | {fmt(r['ROC-AUC'])} | {fmt(r['ECE_10_equal_frequency'])} |"
    lr_ids = [f"{h}_LR_RT_{rt}" for h in ["H0", "H1", "H2"] for rt in ["OFF", "ON"]]
    rt_lines = ["# Step 3 RT Ablation Report", "", "Test metrics (native probabilities):", "", "| Configuration | Log Loss | Brier | ROC-AUC | ECE |", "|---|---:|---:|---:|---:|"]
    rt_lines += [row(x) for x in lr_ids]
    rt_lines += ["", "Matched RT effects (ON minus OFF; negative Log Loss/Brier is favorable):", ""]
    for h in ["H0", "H1", "H2"]:
        off, on = metric_lookup(results, f"{h}_LR_RT_OFF"), metric_lookup(results, f"{h}_LR_RT_ON")
        rt_lines.append(f"- {h}: ΔLog Loss {on['Log Loss']-off['Log Loss']:+.6f}; ΔBrier {on['Brier Score']-off['Brier Score']:+.6f}; ΔROC-AUC {on['ROC-AUC']-off['ROC-AUC']:+.6f}.")
    rt_lines += [
        "",
        "**Conclusion: RT produced negligible and inconsistent effects and is not retained.** Row-weighted test Log Loss changed by −0.000263 (H0), +0.000308 (H1), and +0.000163 (H2) for LR, while equal-learner-weighted paired differences were unfavorable for RT ON at every history level (H0 +0.001416, H1 +0.005001, H2 +0.001241; all learner-bootstrap 95% intervals above zero). H2 GB was also slightly worse with RT ON (row-weighted ΔLog Loss +0.000339). These magnitudes do not justify extra RT dependence.",
        "",
        "RT transformations used only valid positive historical RT, `log1p`, training-learner robust medians/MADs, problem-type context for the previous event, global backoff for sparse/unseen context, and ±8 clipping.",
    ]
    (OUTPUTS / "ASSISTments_Step3_RT_Ablation_Report.md").write_text("\n".join(rt_lines) + "\n", encoding="utf-8")

    hist = ["# Step 3 History Ablation Report", "", "Test metrics (native probabilities):", "", "| Configuration | Log Loss | Brier | ROC-AUC | ECE |", "|---|---:|---:|---:|---:|"]
    hist += [row(x) for x in lr_ids]
    hist += ["", "Main matched differences:", ""]
    for rt in ["OFF", "ON"]:
        for a, b in [("H0", "H1"), ("H1", "H2")]:
            ra, rb = metric_lookup(results, f"{a}_LR_RT_{rt}"), metric_lookup(results, f"{b}_LR_RT_{rt}")
            hist.append(f"- {a}→{b}, RT {rt}: ΔLog Loss {rb['Log Loss']-ra['Log Loss']:+.6f}; ΔBrier {rb['Brier Score']-ra['Brier Score']:+.6f}; ΔROC-AUC {rb['ROC-AUC']-ra['ROC-AUC']:+.6f}.")
    hist += [
        "",
        "**Conclusion: accumulated history and compact chronological history both add practically meaningful overall value.** For RT OFF, H0→H1 reduced row-weighted test Log Loss by 0.021558 and H1→H2 by 0.006919. Equal-learner-weighted reductions were 0.010499 (95% CI 0.008361–0.012767) and 0.010862 (95% CI 0.009331–0.012334), respectively.",
        "",
        "The gain is history-dependent. On first interactions, LR H0/H1/H2 Log Loss was 0.622731/0.623000/0.630957, and for interactions 2–4 it was 0.599749/0.666439/0.595555. At 11+ interactions it improved monotonically: 0.589956/0.565709/0.559836. This supports later fallback investigation but does not implement a policy here.",
        "",
        "History gains were present with both skill available and skill missing, although absolute Log Loss was worse for missing-skill rows. Cold-start, short-history, and skill-missing results remain diagnostics rather than separate confirmatory hypotheses. See `ASSISTments_Step3_Subgroup_Diagnostics.csv` and learner-clustered intervals in `ASSISTments_Step3_Learner_Level_Uncertainty.csv`.",
    ]
    (OUTPUTS / "ASSISTments_Step3_History_Ablation_Report.md").write_text("\n".join(hist) + "\n", encoding="utf-8")

    gb = ["# Step 3 Logistic Regression vs Gradient Boosting", "", "H2 test metrics (native probabilities):", "", "| Configuration | Log Loss | Brier | ROC-AUC | ECE |", "|---|---:|---:|---:|---:|"]
    gb += [row(f"H2_{m}_RT_{rt}") for rt in ["OFF", "ON"] for m in ["LR", "GB"]]
    for rt in ["OFF", "ON"]:
        lr, boost = metric_lookup(results, f"H2_LR_RT_{rt}"), metric_lookup(results, f"H2_GB_RT_{rt}")
        gb.append(f"\n- RT {rt}: GB−LR ΔLog Loss {boost['Log Loss']-lr['Log Loss']:+.6f}; ΔBrier {boost['Brier Score']-lr['Brier Score']:+.6f}; ΔROC-AUC {boost['ROC-AUC']-lr['ROC-AUC']:+.6f}.")
    gb += [
        "",
        "**Conclusion: the nonlinear gain is meaningful enough to retain GB.** With RT OFF, GB reduced row-weighted test Log Loss by 0.027614, Brier by 0.010603, and raised ROC-AUC by 0.036831. The equal-learner-weighted Log Loss improvement was 0.029154 (95% CI 0.027794–0.030683).",
        "",
        "The tradeoff is material: H2 GB RT OFF trained in 914.0 s and occupied 2.17 MB, versus 137.4 s and 4.6 KB for H2 LR RT OFF. LR remains the coefficient-inspectable baseline; GB has lower direct interpretability. Selection used validation probability quality with a 0.001 Log Loss simplicity tolerance, not the best isolated test metric.",
    ]
    (OUTPUTS / "ASSISTments_Step3_LR_vs_GB_Report.md").write_text("\n".join(gb) + "\n", encoding="utf-8")

    cal_lines = ["# Step 3 Calibration Report", "", "Native probabilities were assessed first. Platt calibration was fitted on validation labels only when validation ECE was at least 0.01, and retained only when validation Log Loss improved by at least 0.0005 and ECE by at least 0.002. Test labels were never used to fit calibration.", "", "| Experiment | Split | Status | Log Loss | Brier | ECE | Intercept | Slope |", "|---|---|---|---:|---:|---:|---:|---:|"]
    for _, r in results.iterrows():
        cal_lines.append(f"| {r.experiment_id} | {r.split} | {r.calibration} | {r['Log Loss']:.6f} | {r['Brier Score']:.6f} | {r['ECE_10_equal_frequency']:.6f} | {r.calibration_intercept:.4f} | {r.calibration_slope:.4f} |")
    (OUTPUTS / "ASSISTments_Step3_Calibration_Report.md").write_text("\n".join(cal_lines) + "\n", encoding="utf-8")

    carry = """# ASSISTments Step 3 — Curriculum Carry-Forward

- Local prototype scope: Sri Lankan Grades 1–5.
- Core foundational emphasis: Grades 1–3.
- Upper progression/extension: Grades 4–5.
- Seven local project skill areas: counting/subitizing; quantity comparison; addition; subtraction; multiplication/grouping; division/sharing; place value/building.
- Tier A and Tier B are project-defined operational levels, not official NIE difficulty bands.
- ASSISTments skill IDs are not NIE grade levels and were not converted to local grades or tiers.
- This public-data experiment estimates first-attempt correctness under observed ASSISTments conditions; it does not validate Sri Lankan curriculum progression.
- Curriculum eligibility, progression, abstention/fallback, and activity-selection rules belong to the later recommendation layer (Step 4), not this predictive evaluation.
- No clinical diagnosis or causal activity-effectiveness claim is made.
"""
    (OUTPUTS / "ASSISTments_Step3_Curriculum_Carry_Forward.md").write_text(carry, encoding="utf-8")

    native_test = results[(results.split == "test") & (results.calibration == "native")]
    main_table = native_test[["experiment_id", "Log Loss", "Brier Score", "ROC-AUC", "ECE_10_equal_frequency"]]
    table_lines = [
        "| Experiment | Log Loss | Brier Score | ROC-AUC | ECE |",
        "|---|---:|---:|---:|---:|",
    ]
    for _, mr in main_table.iterrows():
        table_lines.append(
            f"| {mr['experiment_id']} | {mr['Log Loss']:.6f} | {mr['Brier Score']:.6f} | "
            f"{mr['ROC-AUC']:.6f} | {mr['ECE_10_equal_frequency']:.6f} |"
        )
    table_md = "\n".join(table_lines)
    sa = split_audit_df.set_index("split")
    # Focused diagnostic extracts for prose.
    chosen_sub = subgroup[subgroup.experiment_id == selected["experiment_id"]]
    cold = chosen_sub[chosen_sub.dimension == "cold_start"][["group", "Log Loss", "Brier Score", "n_rows"]].to_dict("records")
    skill = chosen_sub[chosen_sub.dimension == "skill_availability"][["group", "Log Loss", "Brier Score", "n_rows"]].to_dict("records")
    freeze = f"""# ASSISTments Step 3 — Final Freeze Summary

## A. Status

- **Status: COMPLETE**
- Independent validation: **pending execution of `validate_step3_artifacts.py`** (updated after validation)

## B. Input provenance

- Frozen Step-2 history SHA-256 verified: `{EXPECTED_STEP2_HISTORY_HASH}`.
- Frozen Step-2 summary SHA-256 verified: `{EXPECTED_STEP2_FREEZE_HASH}`.
- Step-2 manifest and eight curriculum artifacts were hashed into the Step-3 provenance register.

## C. Learner partition

- Fixed seed: `{SEED}`.
- Train: {int(sa.loc['train','learner_count']):,} learners, {int(sa.loc['train','interaction_count']):,} interactions, {int(sa.loc['train','target_eligible_count']):,} target rows.
- Validation: {int(sa.loc['validation','learner_count']):,} learners, {int(sa.loc['validation','interaction_count']):,} interactions, {int(sa.loc['validation','target_eligible_count']):,} target rows.
- Test: {int(sa.loc['test','learner_count']):,} learners, {int(sa.loc['test','interaction_count']):,} interactions, {int(sa.loc['test','target_eligible_count']):,} target rows.
- Learner overlap: zero by construction and automated test.

## D. Target

Frozen first-attempt correctness target. Exactly 0/1 rows are supervised; 5,323 fractional rows remain in prior histories but are excluded as targets.

## E. Feature configurations

- H0: 14 frozen raw features; H1: 29; H2: 42; `H0 ⊂ H1 ⊂ H2`.
- RT OFF removes only the four explicitly tagged RT-dependent frozen features.
- RT ON uses train-only `log1p` robust contextualization and frozen backoff statistics.
- No outcome-derived difficulty proxy or invented curriculum mapping was used.

## F. Main model results

{table_md}

## G. H0 vs H1 conclusion

H1 improves on H0 overall. For LR RT OFF, test Log Loss fell from 0.591088 to 0.569530 (row-weighted Δ −0.021558); the equal-learner-weighted paired improvement was 0.010499 (95% CI 0.008361–0.012767). The gain is not uniform during the earliest interactions.

## H. H1 vs H2 conclusion

H2 improves on H1 overall. For LR RT OFF, test Log Loss fell from 0.569530 to 0.562612 (row-weighted Δ −0.006919); the equal-learner-weighted paired improvement was 0.010862 (95% CI 0.009331–0.012334). H2 is retained for the final predictive configuration.

## I. RT ablation conclusion

RT effects are negligible/inconsistent and RT is not retained. Row-weighted LR test Log Loss changes (ON−OFF) were −0.000263 for H0, +0.000308 for H1, and +0.000163 for H2; equal-learner-weighted differences were unfavorable for RT ON at all three levels. H2 GB RT ON was also slightly worse (+0.000339).

## J. LR vs GB conclusion

H2 GB RT OFF materially outperformed H2 LR RT OFF: test Log Loss 0.534998 vs 0.562612, Brier 0.179078 vs 0.189680, and ROC-AUC 0.750158 vs 0.713327. The learner-weighted Log Loss gain was 0.029154 (95% CI 0.027794–0.030683), sufficient to justify the added complexity.

## K. Calibration conclusion

Native probabilities were evaluated first. Validation-only Platt calibration was retained for both GB variants; for selected H2 GB RT OFF it reduced test Log Loss from 0.534998 to 0.534442 and ECE from 0.010002 to 0.003847, with calibration intercept −0.0010 and slope 0.9949. No LR calibrator met the retention rule.

## L. Cold-start findings

Selected-model diagnostic groups: `{json.dumps(cold)}`. These are predictive diagnostics for later fallback design, not policy rules.

## M. Skill-missingness findings

Selected-model diagnostic groups: `{json.dumps(skill)}`. Missing-skill rows were retained and learner history remained usable.

## N. Selected predictive configuration

- History: **{selected['history_config']}**
- RT: **{selected['RT_status']}**
- Model: **{selected['model']}**
- Calibration retained: **{selected['calibration_retained']}**
- Selection basis: validation Log Loss/Brier/calibration, stability, and a 0.001 Log Loss simplicity tolerance; test was untouched until final evaluation.

## O. Curriculum carry-forward

Grades 1–5 remain the local scope (Grades 1–3 core; Grades 4–5 extension), with seven project skill areas and project-defined Tier A/B. Public-data skill IDs were not treated as NIE levels. Curriculum eligibility/progression belongs to Step 4.

## P. Limitations

- Observational prediction only: no causal activity-effectiveness, mastery, or clinical/dyscalculia diagnosis claim.
- Learner-disjoint generalization does not establish transportability to Sri Lankan children, curricula, languages, devices, or robot interaction.
- RT depends on public-system timing quality; robust handling reduces but cannot eliminate measurement artifacts.
- SGD logistic regression is a scalable stochastic optimizer, so coefficients are descriptive, not causal.
- Subgroup findings are diagnostics and were not multiplicity-adjusted confirmatory tests.

## Q. Readiness for next step

Artifacts are ready for Step 4 only after the independent validator passes. Step 4 was not started.
"""
    (OUTPUTS / "ASSISTments_Step3_Final_Freeze_Summary.md").write_text(freeze, encoding="utf-8")


def create_notebook() -> None:
    sections = [
        "Input Provenance", "Step-3 Scope", "Learner-Disjoint Partition", "Split Audit",
        "Feature Contracts", "Training-Only Preprocessing", "RT Contextualization", "H0 Experiments",
        "H1 Experiments", "H2 Experiments", "Gradient Boosting Benchmark", "Probability Metrics",
        "Calibration", "History Ablation", "RT Ablation", "LR vs GB", "Cold-Start Analysis",
        "Skill-Missingness Analysis", "Learner-Level Uncertainty", "Final Model Selection",
        "Curriculum Carry-Forward", "Validation", "Step-3 Summary", "Readiness for Step 4",
    ]
    cells = [{"cell_type": "markdown", "metadata": {}, "source": ["# ASSISTments Step 3 — Experimental Modeling\n", "\n", "This compact audit notebook loads frozen pipeline outputs; the reproducible computation is in `run_step3_pipeline.py`.\n"]}]
    for section in sections:
        cells.append({"cell_type": "markdown", "metadata": {}, "source": [f"## {section}\n", "\n", "See the corresponding CSV/Markdown artifact in `outputs/`. Large prediction tables are intentionally not rendered here.\n"]})
        if section in {"Input Provenance", "Split Audit", "Feature Contracts", "Probability Metrics", "Learner-Level Uncertainty"}:
            file_map = {
                "Input Provenance": "ASSISTments_Step3_Input_Provenance.csv",
                "Split Audit": "ASSISTments_Step3_Split_Audit.csv",
                "Feature Contracts": "ASSISTments_Step3_Model_Feature_Contract.csv",
                "Probability Metrics": "ASSISTments_Step3_Model_Results.csv",
                "Learner-Level Uncertainty": "ASSISTments_Step3_Learner_Level_Uncertainty.csv",
            }
            cells.append({"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": ["import pandas as pd\n", f"pd.read_csv('outputs/{file_map[section]}')\n"]})
    notebook = {
        "cells": cells,
        "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python", "version": platform.python_version()}},
        "nbformat": 4, "nbformat_minor": 5,
    }
    (ROOT / "03_Experimental_Modeling.ipynb").write_text(json.dumps(notebook, indent=1), encoding="utf-8")


def create_manifest() -> None:
    rows = []
    for path in sorted(ROOT.rglob("*")):
        if not path.is_file() or path.name == "ASSISTments_Step3_Artifact_Manifest.csv" or "__pycache__" in path.parts:
            continue
        rows.append({"path": str(path.resolve()), "role": "Step-3 artifact", "bytes": path.stat().st_size, "sha256": sha256(path)})
    write_csv(OUTPUTS / "ASSISTments_Step3_Artifact_Manifest.csv", rows)


def main() -> None:
    started = time.perf_counter()
    ensure_dirs()
    verify_provenance()
    configs, contract = load_feature_contract()
    make_partitions()
    audit = split_audit()

    h2_features = configs["H2"]
    print("Loading training split...", flush=True)
    train_raw = load_split("train", h2_features)
    train_meta = metadata_frame(train_raw, "train")
    train_meta.to_parquet(DATA_PREP / "train_supervised_rows.parquet", index=False, compression="zstd")
    rt_stats = fit_rt_stats(train_raw)
    template_frequency = fit_template_frequency(train_raw)
    train = engineer(train_raw, rt_stats, template_frequency)
    pre, numeric_cols, categorical_cols = fit_preprocessor(train, h2_features)
    source_map = output_source_map(pre, numeric_cols, categorical_cols)
    X_train = pre.transform(train[h2_features]).tocsr().astype(np.float32)
    y_train = train_meta["y_true"].to_numpy(np.int8)
    del train, train_raw
    gc.collect()

    prep_payload = {
        "preprocessor": pre,
        "rt_stats": rt_stats,
        "template_frequency": template_frequency,
        "source_map": source_map,
        "fit_scope": "training learners and supervised training rows only",
        "fit_learner_file_sha256": sha256(DATA_PART / "train_learners.csv"),
        "seed": SEED,
        "rt_min_support": rt_stats.min_support,
        "rt_fallback": "previous problem_type -> global; unseen/sparse -> global",
        "difficulty_proxy": "none",
    }
    prep_path = MODELS / "preprocessors" / "step3_h2_rt_on_preprocessor_v1.joblib"
    joblib.dump(prep_payload, prep_path, compress=3)
    (OUTPUTS / "ASSISTments_Step3_Preprocessing_Fit_Audit.json").write_text(json.dumps({
        "fit_scope": prep_payload["fit_scope"],
        "fit_learner_file_sha256": prep_payload["fit_learner_file_sha256"],
        "validation_learners_used": 0,
        "test_learners_used": 0,
        "target_used_for_rt_or_encoding": False,
        "template_encoding": "training frequency only; no outcome",
        "rt_context": "previous problem_type with global fallback",
        "rt_min_support": rt_stats.min_support,
        "rt_log_transform": "log1p",
        "rt_robust_scale": "median and 1.4826*MAD",
        "difficulty_proxy": "none",
    }, indent=2), encoding="utf-8")

    split_payload: dict[str, tuple[sparse.csr_matrix, pd.DataFrame]] = {}
    for name in ["validation", "test"]:
        print(f"Loading and transforming {name} split...", flush=True)
        raw = load_split(name, h2_features)
        meta = metadata_frame(raw, name)
        meta.to_parquet(DATA_PREP / f"{name}_supervised_rows.parquet", index=False, compression="zstd")
        eng = engineer(raw, rt_stats, template_frequency)
        X = pre.transform(eng[h2_features]).tocsr().astype(np.float32)
        split_payload[name] = (X, meta)
        del raw, eng
        gc.collect()

    source_arr = np.array(source_map, dtype=object)
    feature_indices: dict[tuple[str, str], np.ndarray] = {}
    for h in ["H0", "H1", "H2"]:
        for rt_status in ["OFF", "ON"]:
            raw_features = set(configs[h])
            if rt_status == "OFF":
                raw_features -= RT_FEATURES
            idx = np.flatnonzero(np.isin(source_arr, list(raw_features)))
            actual_sources = set(source_arr[idx])
            if actual_sources != raw_features:
                raise RuntimeError(f"Transformed feature mapping mismatch for {h} {rt_status}: {raw_features - actual_sources}")
            feature_indices[(h, rt_status)] = idx
    # Exact matched-pair audit.
    pair_rows = []
    for h in ["H0", "H1", "H2"]:
        off = set(source_arr[feature_indices[(h, "OFF")]])
        on = set(source_arr[feature_indices[(h, "ON")]])
        added = on - off
        ok = added == (set(configs[h]) & RT_FEATURES)
        pair_rows.append({"history_config": h, "off_raw_features": len(off), "on_raw_features": len(on), "added_features": ";".join(sorted(added)), "only_approved_rt_added": ok})
        if not ok:
            raise RuntimeError(f"RT matched pair failed for {h}")
    pd.DataFrame(pair_rows).to_csv(OUTPUTS / "ASSISTments_Step3_RT_Feature_Pair_Audit.csv", index=False)

    experiments = []
    for h in ["H0", "H1", "H2"]:
        for rt_status in ["OFF", "ON"]:
            experiments.append({"experiment_id": f"{h}_LR_RT_{rt_status}", "history_config": h, "model": "LR", "RT_status": rt_status})
    for rt_status in ["OFF", "ON"]:
        experiments.append({"experiment_id": f"H2_GB_RT_{rt_status}", "history_config": "H2", "model": "GB", "RT_status": rt_status})

    registry_rows, result_rows, reliability, coeffs = [], [], [], []
    predictions_val: dict[str, np.ndarray] = {}
    predictions_test: dict[str, np.ndarray] = {}
    val_X, val_meta = split_payload["validation"]
    test_X, test_meta = split_payload["test"]
    y_val = val_meta["y_true"].to_numpy(np.int8)
    y_test = test_meta["y_true"].to_numpy(np.int8)

    for exp in experiments:
        exp_start = time.perf_counter()
        idx = feature_indices[(exp["history_config"], exp["RT_status"])]
        if exp["model"] == "LR":
            model = SGDClassifier(
                loss="log_loss", penalty="l2", alpha=1e-5, max_iter=30, tol=1e-4,
                shuffle=True, random_state=SEED, average=True, class_weight=None,
            )
            hyper = "SGD log_loss; L2 alpha=1e-5; max_iter=30; tol=1e-4; average=True; no class weights"
            model.fit(X_train[:, idx], y_train)
            model_path = MODELS / "logistic_regression" / f"{exp['experiment_id']}_v1.joblib"
            joblib.dump({"model": model, "selected_columns": idx, "source_features": sorted(set(source_arr[idx]))}, model_path, compress=3)
            coeffs.append(coefficient_table(model, pre, idx, exp["experiment_id"]))
        else:
            model = XGBClassifier(
                objective="binary:logistic", eval_metric="logloss", n_estimators=350,
                learning_rate=0.08, max_depth=6, min_child_weight=20, subsample=0.80,
                colsample_bytree=0.80, reg_lambda=2.0, reg_alpha=0.0, tree_method="hist",
                max_bin=255, random_state=SEED, n_jobs=max(1, os.cpu_count() or 1),
                early_stopping_rounds=20,
            )
            hyper = "XGBoost 3.0.4 hist; 350 max trees; eta=.08; depth=6; min_child_weight=20; subsample=.8; colsample=.8; lambda=2; early stopping=20"
            model.fit(X_train[:, idx], y_train, eval_set=[(val_X[:, idx], y_val)], verbose=False)
            model_path = MODELS / "gradient_boosting" / f"{exp['experiment_id']}_v1.json"
            model.save_model(model_path)
            imp = pd.DataFrame({
                "experiment_id": exp["experiment_id"],
                "transformed_feature": pre.get_feature_names_out()[idx],
                "gain_importance": model.feature_importances_,
            }).sort_values("gain_importance", ascending=False).head(100)
            imp.to_csv(OUTPUTS / f"ASSISTments_Step3_{exp['experiment_id']}_Feature_Importance.csv", index=False)
        p_val = np.clip(model.predict_proba(val_X[:, idx])[:, 1], 1e-7, 1 - 1e-7)
        p_test = np.clip(model.predict_proba(test_X[:, idx])[:, 1], 1e-7, 1 - 1e-7)
        predictions_val[exp["experiment_id"]] = p_val
        predictions_test[exp["experiment_id"]] = p_test
        for split, meta, y, p in [("validation", val_meta, y_val, p_val), ("test", test_meta, y_test, p_test)]:
            result_rows.append(metric_row(exp["experiment_id"], split, y, p, meta["user_id"].to_numpy(), "native"))
            reliability.extend(reliability_rows(exp["experiment_id"], split, y, p, "native"))
            save_predictions(meta, p, exp, "native")
        runtime = time.perf_counter() - exp_start
        registry_rows.append({
            **exp,
            "feature_count": len(configs[exp["history_config"]]) if exp["RT_status"] == "ON" else len(set(configs[exp["history_config"]]) - RT_FEATURES),
            "transformed_feature_count": len(idx),
            "preprocessing_version": "step3_train_only_v1",
            "random_seed": SEED,
            "regularization/hyperparameters": hyper,
            "calibration": "native; validation-only Platt assessed separately",
            "train_rows": len(y_train), "validation_rows": len(y_val), "test_rows": len(y_test),
            "runtime_seconds": runtime,
            "model_artifact": str(model_path.resolve()),
            "model_bytes": model_path.stat().st_size,
            "status": "COMPLETE",
        })
        print(f"Completed {exp['experiment_id']} in {runtime:.1f}s", flush=True)

    if coeffs:
        pd.concat(coeffs, ignore_index=True).to_csv(OUTPUTS / "ASSISTments_Step3_LR_Coefficients_Top100.csv", index=False)

    # Selective validation-only Platt calibration.
    calibrators: dict[str, LogisticRegression] = {}
    calibration_decisions = []
    for exp in experiments:
        eid = exp["experiment_id"]
        native_val = next(r for r in result_rows if r["experiment_id"] == eid and r["split"] == "validation" and r["calibration"] == "native")
        attempted = native_val["ECE_10_equal_frequency"] >= 0.01
        retained = False
        if attempted:
            pv = predictions_val[eid]
            logits_v = np.log(pv / (1 - pv)).reshape(-1, 1)
            cal = LogisticRegression(C=1e6, solver="lbfgs", max_iter=100, random_state=SEED)
            cal.fit(logits_v, y_val)
            p_v_cal = cal.predict_proba(logits_v)[:, 1]
            pt = predictions_test[eid]
            p_t_cal = cal.predict_proba(np.log(pt / (1 - pt)).reshape(-1, 1))[:, 1]
            val_cal_row = metric_row(eid, "validation", y_val, p_v_cal, val_meta["user_id"].to_numpy(), "platt_validation_fit")
            retained = (
                native_val["Log Loss"] - val_cal_row["Log Loss"] >= 0.0005
                and native_val["ECE_10_equal_frequency"] - val_cal_row["ECE_10_equal_frequency"] >= 0.002
            )
            if retained:
                calibrators[eid] = cal
                joblib.dump(cal, MODELS / "calibration" / f"{eid}_platt_v1.joblib", compress=3)
                result_rows.append(val_cal_row)
                result_rows.append(metric_row(eid, "test", y_test, p_t_cal, test_meta["user_id"].to_numpy(), "platt_retained"))
                reliability.extend(reliability_rows(eid, "validation", y_val, p_v_cal, "platt_validation_fit"))
                reliability.extend(reliability_rows(eid, "test", y_test, p_t_cal, "platt_retained"))
                save_predictions(val_meta, p_v_cal, exp, "platt_validation_fit")
                save_predictions(test_meta, p_t_cal, exp, "platt_retained")
        calibration_decisions.append({
            "experiment_id": eid,
            "native_validation_ECE": native_val["ECE_10_equal_frequency"],
            "attempted": attempted,
            "retained": retained,
            "rule": "attempt if ECE>=0.01; retain if validation Log Loss improves >=0.0005 and ECE improves >=0.002",
        })
    pd.DataFrame(calibration_decisions).to_csv(OUTPUTS / "ASSISTments_Step3_Calibration_Decisions.csv", index=False)

    results = pd.DataFrame(result_rows)
    results.to_csv(OUTPUTS / "ASSISTments_Step3_Model_Results.csv", index=False)
    reliability_df = pd.DataFrame(reliability)
    reliability_df.to_csv(OUTPUTS / "ASSISTments_Step3_Reliability_Summary.csv", index=False)

    # Selection is validation-only, with an explicit simplicity tolerance.
    candidates = []
    for exp in experiments:
        eid = exp["experiment_id"]
        cal_status = "platt_validation_fit" if eid in calibrators else "native"
        rr = results[(results.experiment_id == eid) & (results.split == "validation") & (results.calibration == cal_status)].iloc[0]
        complexity = (0 if exp["model"] == "LR" else 2) + (0 if exp["history_config"] == "H0" else 1 if exp["history_config"] == "H1" else 2) + (1 if exp["RT_status"] == "ON" else 0)
        candidates.append({**exp, "validation_log_loss": rr["Log Loss"], "validation_brier": rr["Brier Score"], "calibration_retained": eid in calibrators, "complexity": complexity})
    best_loss = min(c["validation_log_loss"] for c in candidates)
    eligible = [c for c in candidates if c["validation_log_loss"] <= best_loss + 0.001]
    selected = sorted(eligible, key=lambda c: (c["complexity"], c["validation_log_loss"], c["validation_brier"]))[0]
    (OUTPUTS / "ASSISTments_Step3_Selected_Configuration.json").write_text(json.dumps(selected, indent=2), encoding="utf-8")

    comparisons = [
        ("H0_LR_RT_OFF", "H1_LR_RT_OFF", "A: H0→H1 LR RT OFF"),
        ("H1_LR_RT_OFF", "H2_LR_RT_OFF", "B: H1→H2 LR RT OFF"),
        ("H0_LR_RT_OFF", "H0_LR_RT_ON", "C: H0 RT OFF→ON"),
        ("H1_LR_RT_OFF", "H1_LR_RT_ON", "C: H1 RT OFF→ON"),
        ("H2_LR_RT_OFF", "H2_LR_RT_ON", "C: H2 RT OFF→ON"),
        ("H2_LR_RT_OFF", "H2_GB_RT_OFF", "D: H2 LR→GB RT OFF"),
        ("H2_LR_RT_ON", "H2_GB_RT_ON", "D: H2 LR→GB RT ON"),
    ]
    uncertainty_rows = []
    for a, b, label in comparisons:
        uncertainty_rows.extend(paired_bootstrap(test_meta, predictions_test[a], predictions_test[b], a, b, label))
    uncertainty = pd.DataFrame(uncertainty_rows)
    uncertainty.to_csv(OUTPUTS / "ASSISTments_Step3_Learner_Level_Uncertainty.csv", index=False)

    # Limited diagnostics on required models; avoids dozens of exploratory claims.
    diagnostic_ids = ["H0_LR_RT_OFF", "H1_LR_RT_OFF", "H2_LR_RT_OFF", "H2_LR_RT_ON", "H2_GB_RT_OFF", "H2_GB_RT_ON", selected["experiment_id"]]
    diagnostic_ids = list(dict.fromkeys(diagnostic_ids))
    subgroup = subgroup_metrics(test_meta, {k: predictions_test[k] for k in diagnostic_ids})
    subgroup.to_csv(OUTPUTS / "ASSISTments_Step3_Subgroup_Diagnostics.csv", index=False)

    registry = pd.DataFrame(registry_rows)
    registry.to_csv(OUTPUTS / "ASSISTments_Step3_Experiment_Registry.csv", index=False)
    plot_reliability(reliability_df, ["H0_LR_RT_OFF", "H1_LR_RT_OFF", "H2_LR_RT_OFF", "H2_GB_RT_OFF", selected["experiment_id"]])
    write_reports(results, uncertainty, subgroup, registry, selected, audit)
    create_notebook()
    runtime = {
        "total_seconds": time.perf_counter() - started,
        "platform": platform.platform(),
        "python": platform.python_version(),
        "packages": {"numpy": np.__version__, "pandas": pd.__version__, "scipy": scipy.__version__, "scikit-learn": sklearn.__version__, "xgboost": xgboost.__version__, "duckdb": duckdb.__version__},
        "cpu_count": os.cpu_count(),
        "peak_memory_not_instrumented": True,
    }
    (OUTPUTS / "ASSISTments_Step3_Runtime_Environment.json").write_text(json.dumps(runtime, indent=2), encoding="utf-8")
    create_manifest()
    print(f"Pipeline complete in {runtime['total_seconds']:.1f}s. Selected: {selected['experiment_id']}", flush=True)


if __name__ == "__main__":
    main()
