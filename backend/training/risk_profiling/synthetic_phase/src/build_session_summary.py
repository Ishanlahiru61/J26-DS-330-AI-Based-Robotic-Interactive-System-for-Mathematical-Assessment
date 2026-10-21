"""Turn cube_assessment_trials into student_performance_summary (schema v1).

Usage:
    python build_session_summary.py <trials.csv> <summary.csv>

One output row per student x session x task type. Definitions: docs/dataset_schema.md.
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
THRESHOLDS = json.loads((MODULE / "config" / "mastery_thresholds.json").read_text(encoding="utf-8"))

KEYS = ["data_origin", "student_id", "session_id", "session_index", "age_months", "grade", "task_type"]


def _session_features(group: pd.DataFrame) -> pd.Series:
    valid = group[group["valid_trial"] == 1]
    rt = group.loc[group["rt_valid"] == 1, "response_time_s"]
    rt_correct = group.loc[(group["rt_valid"] == 1) & (group["accuracy"] == 1), "response_time_s"]
    expected = valid["expected_answer"].where(valid["expected_answer"] > 0)
    behaviour = group["behavior_label"].where(group["behavior_label"] != "uncertain")  # I2: unknown = missing
    n_valid = len(valid)
    demo = group["robot_demo_triggered"]
    demo_logged = group["robot_demo_source"] == "component1"
    accuracy_rate = valid["accuracy"].mean() if n_valid else np.nan
    return pd.Series({
        "c2_activity": group["c2_activity"].iloc[0],
        "c2_area": group["c2_area"].iloc[0],
        "in_curriculum": int(group["in_curriculum"].iloc[0]),
        "n_trials": len(group),
        "n_valid_trials": n_valid,
        "accuracy_rate": accuracy_rate,
        "error_rate": 1 - accuracy_rate if n_valid else np.nan,
        "mean_relative_error": (valid["error_value"] / expected).mean() if n_valid else np.nan,
        "median_rt_correct_s": rt_correct.median() if len(rt_correct) else np.nan,
        "rt_cov": rt.std(ddof=1) / rt.mean() if len(rt) >= 2 else np.nan,
        "mean_attempts": valid["attempts"].mean() if n_valid else np.nan,
        # B1c: logged and derived demo values are never mixed in one rate.
        "robot_demo_rate_component1": demo.loc[demo_logged].mean() if demo_logged.any() else np.nan,
        "robot_demo_rate_derived": demo.loc[~demo_logged].mean() if (~demo_logged).any() else np.nan,
        "n_robot_demo_derived": int((~demo_logged).sum()),
        "finger_counting_rate": (behaviour == "finger_counting").sum() / behaviour.notna().sum()
        if behaviour.notna().any() else np.nan,
    })


def _mastery(row: pd.Series) -> str | None:
    if row["n_valid_trials"] < THRESHOLDS["min_valid_trials"] or pd.isna(row["accuracy_rate"]):
        return None
    for key in ("successful", "shows_difficulty", "needs_further_help"):
        if row["accuracy_rate"] >= THRESHOLDS["levels"][key]["min_accuracy_rate"]:
            return THRESHOLDS["levels"][key]["label_en"]
    return None


def _improvement_slope(summary: pd.DataFrame) -> pd.Series:
    """Slope of accuracy_rate over session_index, using this and all earlier sessions (A4, F13)."""
    slopes = pd.Series(np.nan, index=summary.index)
    for _, group in summary.groupby(["student_id", "task_type"]):
        group = group.sort_values("session_index")
        for i in range(1, len(group)):
            history = group.iloc[: i + 1].dropna(subset=["accuracy_rate"])
            if history["session_index"].nunique() >= 2:
                slopes[group.index[i]] = np.polyfit(history["session_index"], history["accuracy_rate"], 1)[0]
    return slopes


def build_summary(trials: pd.DataFrame) -> pd.DataFrame:
    problems = validate_trials(trials)
    if problems:
        raise ValueError("invalid trials table:\n- " + "\n- ".join(problems))
    trials = add_derived_columns(trials)
    trials["task_type"] = trials["task_type"].fillna("UNASSIGNED")
    summary = (
        trials.groupby(KEYS, observed=True, dropna=False)[list(trials.columns)]
        .apply(_session_features)
        .reset_index()
    )
    summary["risk_use"] = summary["task_type"].map(lambda t: TASK_TYPES[t][4] if t in TASK_TYPES else "none")
    summary["age_grade_mismatch"] = pd.NA  # B2c: open until the age-grade rule (C1) is sourced
    summary["improvement_slope"] = _improvement_slope(summary)
    summary["mastery_level"] = summary.apply(_mastery, axis=1)
    summary["mastery_thresholds_version"] = THRESHOLDS["version"]
    return summary.sort_values(["student_id", "session_index", "task_type"]).reset_index(drop=True)


def main() -> None:
    source, target = Path(sys.argv[1]), Path(sys.argv[2])
    summary = build_summary(pd.read_csv(source))
    target.parent.mkdir(parents=True, exist_ok=True)
    summary.round(4).to_csv(target, index=False)
    print(f"{len(summary)} summary rows -> {target}")


if __name__ == "__main__":
    main()
