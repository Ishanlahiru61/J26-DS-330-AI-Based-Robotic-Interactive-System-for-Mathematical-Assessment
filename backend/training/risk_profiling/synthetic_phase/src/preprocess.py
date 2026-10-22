"""Step 2 - preprocessing: trials -> one clean record per child (per task type and per C2 area).

Usage:
    python preprocess.py                      # synthetic trials-v1 from artifacts/
    python preprocess.py <trials.csv> <out_dir> [labels.csv]

Each rule names its evidence-register row; see docs/preprocessing.md.
Labels (dd_label) are used only to balance the train/validation/test split.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dataset_schema import TASK_TYPES, add_derived_columns, validate_trials  # noqa: E402

MODULE = Path(__file__).resolve().parents[1]
ARTIFACTS = MODULE / "artifacts"
REPORT = MODULE / "results" / "preprocessing_report.json"
SEED = 42
SPLIT = {"train": 0.70, "validation": 0.15, "test": 0.15}  # learner-level split as in [P-C2] p. 23

RISK_TASKS = {t for t, v in TASK_TYPES.items() if v[4] == "risk"}
# Risk task types a child of each grade should have (task_taxonomy.md Section 3).
EXPECTED_RISK_TASKS = {g: {t for t in RISK_TASKS if g in TASK_TYPES[t][3]} for g in (1, 2, 3)}
# C2 areas a child of each grade should have risk evidence in (multiplication is Grade 3 only, C3c).
EXPECTED_AREAS = {
    1: {"number_sense", "number_relationships", "arithmetic_calculation"},
    2: {"number_sense", "number_relationships", "arithmetic_calculation"},
    3: {"number_sense", "number_relationships", "arithmetic_calculation", "multiplicative_reasoning"},
}


def clean_trials(trials: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Steps 1-5: validate, derive, drop unusable rows, prepare response times."""
    problems = validate_trials(trials)
    if problems:
        raise ValueError("invalid trials table:\n- " + "\n- ".join(problems))
    df = add_derived_columns(trials)
    report = {"trials_in": len(df)}

    invalid = df["valid_trial"] == 0                                    # B1
    unassigned = df["task_type"].isna() & ~invalid                     # C7
    out_of_curriculum = (df["in_curriculum"] == 0) & ~invalid & ~unassigned  # C3b, C3c
    report["removed_invalid_trial"] = int(invalid.sum())
    report["removed_unassigned_task_type"] = int(unassigned.sum())
    report["removed_not_in_curriculum"] = int(out_of_curriculum.sum())
    df = df[~(invalid | unassigned | out_of_curriculum)].copy()
    report["trials_kept"] = len(df)

    # Step 5 (I3): RT only where usable, only for correct answers, on a log scale (as C2: log(1 + RT)).
    df["rt_usable"] = (df["rt_valid"] == 1) & (df["accuracy"] == 1)
    df["log_rt"] = np.where(df["rt_usable"], np.log1p(df["response_time_s"]), np.nan)
    report["rt_excluded_censored_or_demo"] = int((df["rt_valid"] == 0).sum())
    report["rt_used_correct_answers"] = int(df["rt_usable"].sum())
    df["behaviour_known"] = df["behavior_label"].notna() & (df["behavior_label"] != "uncertain")  # I2
    df["finger_counting"] = (df["behavior_label"] == "finger_counting").astype(int)
    expected = df["expected_answer"].where(df["expected_answer"] > 0)
    df["relative_error"] = df["error_value"] / expected
    return df, report


def pool(df: pd.DataFrame, by: str) -> pd.DataFrame:
    """Steps 6-7: pool all sessions of a child for each task type (or area)."""
    keys = ["student_id", "grade", by]
    grouped = df.groupby(keys, observed=True)
    out = grouped.agg(
        age_months=("age_months", "max"),
        n_sessions=("session_index", "nunique"),
        n_questions=("accuracy", "size"),
        n_correct=("accuracy", "sum"),
        mean_relative_error=("relative_error", "mean"),
        mean_attempts=("attempts", "mean"),
        n_rt=("rt_usable", "sum"),
        median_log_rt=("log_rt", "median"),
        sd_log_rt=("log_rt", "std"),
    ).reset_index()
    out["accuracy_raw"] = out["n_correct"] / out["n_questions"]
    out["accuracy_smoothed"] = (out["n_correct"] + 1) / (out["n_questions"] + 2)  # Laplace correction
    known = df[df["behaviour_known"]].groupby(keys, observed=True)["finger_counting"].mean()
    out = out.merge(known.rename("finger_counting_rate").reset_index(), on=keys, how="left")
    return out


def assign_split(children: pd.DataFrame, labels: pd.Series | None) -> pd.Series:
    """Step 10: child-level split, stratified by grade (and by label when given)."""
    rng = np.random.default_rng(SEED)
    strata = children["grade"].astype(str)
    if labels is not None:
        strata = strata + "_" + children["student_id"].map(labels).astype(str)
    split = pd.Series(index=children.index, dtype=object)
    for _, idx in children.groupby(strata).groups.items():
        idx = rng.permutation(np.asarray(idx))
        n_train = round(len(idx) * SPLIT["train"])
        n_val = round(len(idx) * SPLIT["validation"])
        split[idx[:n_train]] = "train"
        split[idx[n_train:n_train + n_val]] = "validation"
        split[idx[n_train + n_val:]] = "test"
    return split


def preprocess(trials: pd.DataFrame, labels: pd.Series | None = None):
    df, report = clean_trials(trials)
    by_task = pool(df, "task_type")
    by_task["risk_use"] = by_task["task_type"].map(lambda t: TASK_TYPES[t][4])
    by_area = pool(df[df["task_type"].isin(RISK_TASKS)], "c2_area")  # G11 remedy: more questions per score

    # Step 9: child-level record and data sufficiency (docs/screening-boundary.md).
    children = df.groupby("student_id").agg(
        grade=("grade", "first"), age_months=("age_months", "max"),
        n_sessions=("session_index", "nunique"), n_questions=("accuracy", "size"),
    ).reset_index()
    present = by_task[by_task["risk_use"] == "risk"].groupby("student_id")["task_type"].agg(set)
    children["missing_risk_tasks"] = [
        ",".join(sorted(EXPECTED_RISK_TASKS[g] - present.get(s, set())))
        for s, g in zip(children["student_id"], children["grade"])
    ]
    children["sufficient_data"] = (children["missing_risk_tasks"] == "").astype(int)
    areas_present = by_area.groupby("student_id")["c2_area"].agg(set)
    children["missing_areas"] = [
        ",".join(sorted(EXPECTED_AREAS[g] - areas_present.get(s, set())))
        for s, g in zip(children["student_id"], children["grade"])
    ]
    children["sufficient_data_area"] = (children["missing_areas"] == "").astype(int)
    children["split"] = assign_split(children, labels)
    split_map = children.set_index("student_id")["split"]
    by_task["split"] = by_task["student_id"].map(split_map)
    by_area["split"] = by_area["student_id"].map(split_map)

    report.update({
        "children": len(children),
        "children_sufficient_data_all_task_types": int(children["sufficient_data"].sum()),
        "children_sufficient_data_all_areas": int(children["sufficient_data_area"].sum()),
        "median_questions_per_child_task": float(by_task["n_questions"].median()),
        "median_questions_per_child_area": float(by_area["n_questions"].median()),
        "split_children": children["split"].value_counts().to_dict(),
    })
    if labels is not None:
        report["split_dd"] = children.assign(dd=children["student_id"].map(labels)) \
            .groupby("split")["dd"].sum().astype(int).to_dict()
    return children, by_task, by_area, report


def main() -> None:
    if len(sys.argv) > 1:
        trials_path, out_dir = Path(sys.argv[1]), Path(sys.argv[2])
        labels_path = Path(sys.argv[3]) if len(sys.argv) > 3 else None
    else:
        trials_path, out_dir = ARTIFACTS / "synthetic_trials_v1.csv", ARTIFACTS
        labels_path = ARTIFACTS / "synthetic_trials_v1_children.csv"
    labels = None
    if labels_path is not None:
        labels = pd.read_csv(labels_path).set_index("child_id")["dd_label"]
    children, by_task, by_area, report = preprocess(pd.read_csv(trials_path), labels)
    out_dir.mkdir(parents=True, exist_ok=True)
    children.to_csv(out_dir / "preprocessed_children_v1.csv", index=False)
    by_task.round(4).to_csv(out_dir / "preprocessed_child_task_v1.csv", index=False)
    by_area.round(4).to_csv(out_dir / "preprocessed_child_area_v1.csv", index=False)
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
