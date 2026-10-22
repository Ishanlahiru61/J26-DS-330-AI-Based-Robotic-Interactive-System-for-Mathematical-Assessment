"""Build the final child-level datasets for the risk-profiling model.

Outputs
- data/synthetic_children_v1.csv
  SYNTHETIC training set: one row per simulated child, feature z-scores + DD label.
- data/private/risk_profiling/real_decarli_{calibration,test}_z.csv
  REAL children of Decarli et al. (2020) on the same z-score scale (not in Git).
- backend/training/risk_profiling/synthetic_phase/results/generator_parameters.json
- backend/training/risk_profiling/synthetic_phase/results/synthetic_check.csv

Every parameter below is traced in docs/evidence_register.md (rows G1-G8, H1).
Run build_real_labeled_participant_dataset.py first.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

MODULE = Path(__file__).resolve().parents[1]
REPO = Path(__file__).resolve().parents[5]
PRIVATE = REPO / "data" / "private" / "risk_profiling"
SYNTHETIC = MODULE / "data"
RESULTS = MODULE / "results"

SEED = 42
N_CHILDREN = 10_000
# G3: midpoint of 3-7% [Haberstroh & Schulte-Korne 2019]; equals the <=5th
# percentile DD criterion of [D].
PREVALENCE = 0.05

# id, column name, Decarli column (None = not measured in [D]),
# True if a higher raw value means worse, generator effect size, source.
# G1 rule: the most precise significant pooled estimate; if none is
# significant, the [M] point estimate (the only pooled estimate available).
FEATURES = [
    ("F1", "z_number_sense_accuracy", "Accuracy_SymbolicComp", False, 0.90, "[M] g=0.90, p<.001"),
    ("F2", "z_number_sense_rt", "RTs_SymbolicComp", True, 0.75, "[S] g=0.75 (CI 0.51-0.99); [M] 1.03 n.s."),
    ("F3", "z_arithmetic_accuracy", None, False, 0.87, "[M] g=0.87 fact retrieval, p<.01"),
    ("F4", "z_arithmetic_rt", None, True, 0.38, "[M] g=0.38 n.s. (only pooled estimate)"),
    ("F5", "z_enumeration_4_7_rt", "MeanRT_47", True, 0.77, "[M] g=0.77, p<.001"),
    ("F6", "z_enumeration_4_7_accuracy", "Mean_ACC_47", False, 0.69, "[M] g=0.69 n.s. (only pooled estimate)"),
    ("F7", "z_nonsymbolic_comparison_accuracy", "Mean_ACC_ANS", False, 0.69, "[M] g=0.69, p<.001"),
    ("F8", "z_nonsymbolic_comparison_rt", "Mean_RTs_ANS", True, 0.24, "[S] g=0.24 (CI 0.13-0.36); [M] 0.27 n.s."),
]
NAMES = [f[1] for f in FEATURES]


def to_z(real: pd.DataFrame, reference: pd.DataFrame) -> pd.DataFrame:
    """z = (x - control mean) / control SD of the calibration split, worse = positive (E1)."""
    out = pd.DataFrame(index=real.index)
    for _, name, column, higher_is_worse, _, _ in FEATURES:
        if column is None:
            out[name] = np.nan
            continue
        controls = reference.loc[reference["dd_label"] == 0, column]
        z = (real[column] - controls.mean()) / controls.std(ddof=1)
        out[name] = z if higher_is_worse else -z
    return out


def pooled_within_group_correlation(z: pd.DataFrame, labels: pd.Series) -> pd.DataFrame:
    """G2: correlation after removing group means, so the DD effect does not inflate it."""
    residuals = z - z.groupby(labels).transform("mean")
    return residuals.corr()


def riley_minimum_n(parameters: int, prevalence: float) -> dict:
    """Riley et al. (2020) BMJ 368:m441, binary outcome, criteria 1-3.

    Anticipated Cox-Snell R2 = 15% of its maximum, the conservative default the
    authors recommend when no previous model exists.
    """
    log_l_null = prevalence * math.log(prevalence) + (1 - prevalence) * math.log(1 - prevalence)
    max_r2 = 1 - math.exp(2 * log_l_null)
    r2 = 0.15 * max_r2
    shrinkage = 0.9
    c1 = parameters / ((shrinkage - 1) * math.log(1 - r2 / shrinkage))
    s_vh = r2 / (r2 + 0.05 * max_r2)
    c2 = parameters / ((s_vh - 1) * math.log(1 - r2 / s_vh))
    c3 = (1.96 / 0.05) ** 2 * prevalence * (1 - prevalence)
    return {
        "anticipated_r2_cox_snell": round(r2, 4),
        "criterion_1_shrinkage": math.ceil(c1),
        "criterion_2_optimism": math.ceil(c2),
        "criterion_3_intercept": math.ceil(c3),
        "minimum_n": math.ceil(max(c1, c2, c3)),
    }


def hedges_g(worse: np.ndarray, better: np.ndarray) -> float:
    n1, n2 = len(worse), len(better)
    pooled = math.sqrt(((n1 - 1) * worse.var(ddof=1) + (n2 - 1) * better.var(ddof=1)) / (n1 + n2 - 2))
    correction = 1 - 3 / (4 * (n1 + n2) - 9)
    return (worse.mean() - better.mean()) / pooled * correction


def main() -> None:
    real = pd.read_csv(PRIVATE / "component3_real_labeled_participant_dataset.csv")
    splits = pd.read_csv(PRIVATE / "component3_real_participant_split_assignments.csv")
    real = real.merge(splits, on="student_id", validate="one_to_one")
    real["dd_label"] = (real["diagnostic_group"] == "DD").astype(int)
    # H1: train + validation children calibrate the generator; test children stay unseen.
    real["role"] = np.where(real["split"] == "test", "test", "calibration")
    calibration = real[real["role"] == "calibration"]

    z_real = to_z(real, calibration)
    measured = [name for _, name, column, *_ in FEATURES if column is not None]
    correlation = pooled_within_group_correlation(
        z_real.loc[calibration.index, measured], calibration["dd_label"]
    )

    # G4: features never measured together get within-group correlation 0.
    full = pd.DataFrame(np.eye(len(NAMES)), index=NAMES, columns=NAMES)
    full.loc[measured, measured] = correlation.values
    if np.linalg.eigvalsh(full.values).min() <= 0:
        raise ValueError("Correlation matrix is not positive definite.")

    rng = np.random.default_rng(SEED)
    n_dd = round(PREVALENCE * N_CHILDREN)
    labels = np.zeros(N_CHILDREN, dtype=int)
    labels[:n_dd] = 1
    rng.shuffle(labels)
    effects = np.array([f[4] for f in FEATURES])
    # G1/G5: controls ~ N(0, R); DD children ~ N(effect sizes, R), equal variance.
    z_syn = rng.multivariate_normal(np.zeros(len(NAMES)), full.values, size=N_CHILDREN)
    z_syn += labels[:, None] * effects

    synthetic = pd.DataFrame(z_syn, columns=NAMES).round(4)
    synthetic.insert(0, "child_id", [f"SYN-{i:05d}" for i in range(1, N_CHILDREN + 1)])
    synthetic.insert(1, "is_synthetic", True)
    synthetic["dd_label"] = labels
    SYNTHETIC.mkdir(parents=True, exist_ok=True)
    synthetic.to_csv(SYNTHETIC / "synthetic_children_v1.csv", index=False)

    for role in ("calibration", "test"):
        rows = real["role"] == role
        frame = z_real.loc[rows].round(4)
        frame.insert(0, "child_id", "DEC-" + real.loc[rows, "student_id"].astype(str))
        frame.insert(1, "is_synthetic", False)
        frame["dd_label"] = real.loc[rows, "dd_label"].values
        frame.to_csv(PRIVATE / f"real_decarli_{role}_z.csv", index=False)

    check = []
    for fid, name, column, _, effect, _ in FEATURES:
        g_syn = hedges_g(synthetic.loc[labels == 1, name].values, synthetic.loc[labels == 0, name].values)
        row = {"feature": fid, "column": name, "target_g": effect, "synthetic_g": round(g_syn, 3)}
        if column is not None:
            cal = z_real.loc[calibration.index, name]
            row["real_calibration_g"] = round(hedges_g(
                cal[calibration["dd_label"] == 1].values, cal[calibration["dd_label"] == 0].values), 3)
        check.append(row)
    RESULTS.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(check).to_csv(RESULTS / "synthetic_check.csv", index=False)

    syn_corr = pooled_within_group_correlation(synthetic[measured], synthetic["dd_label"])
    parameters = {
        "dataset_version": "synthetic_children_v1",
        "seed": SEED,
        "n_children": N_CHILDREN,
        "n_dd": n_dd,
        "prevalence": PREVALENCE,
        "sample_size_riley_2020": riley_minimum_n(len(NAMES), PREVALENCE),
        "features": [
            {"id": f[0], "column": f[1], "decarli_column": f[2], "higher_raw_is_worse": f[3],
             "generator_effect_size": f[4], "source": f[5]}
            for f in FEATURES
        ],
        "within_group_correlation": full.round(3).to_dict(),
        "max_abs_correlation_error_synthetic_vs_target": round(
            float((syn_corr - correlation).abs().to_numpy().max()), 3),
        "calibration_children": {"n": int(len(calibration)), "dd": int(calibration["dd_label"].sum())},
        "test_children": {"n": int((real["role"] == "test").sum()),
                          "dd": int(real.loc[real["role"] == "test", "dd_label"].sum())},
    }
    (RESULTS / "generator_parameters.json").write_text(json.dumps(parameters, indent=2), encoding="utf-8")
    print(pd.DataFrame(check).to_string(index=False))
    print(json.dumps({k: parameters[k] for k in ("n_children", "n_dd", "sample_size_riley_2020",
                                                 "max_abs_correlation_error_synthetic_vs_target",
                                                 "calibration_children", "test_children")}, indent=2))
    print(correlation.round(2))


if __name__ == "__main__":
    main()
