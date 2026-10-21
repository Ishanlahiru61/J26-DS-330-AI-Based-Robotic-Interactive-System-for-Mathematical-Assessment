"""Column definitions and task taxonomy for the two operational datasets (schema v1).

- cube_assessment_trials: one row per answered question.
- student_performance_summary: one row per student x session x task type.

Every column names the component that supplies it and its evidence-register row.
See docs/dataset_schema.md and docs/task_taxonomy.md.
"""

from __future__ import annotations

import pandas as pd

SCHEMA_VERSION = "v1"

C2_ACTIVITIES = (
    "counting", "comparison", "ordering", "place_value",
    "addition", "subtraction", "multiplication", "division",
)
C2_AREAS = (
    "number_sense", "number_relationships", "arithmetic_calculation", "multiplicative_reasoning",
)
REPRESENTATIONS = ("non_symbolic", "symbolic", "verbal", "mixed")
BEHAVIOR_LABELS = ("finger_counting", "other_hand_movement", "uncertain")
DATA_ORIGINS = ("real", "demo", "synthetic")

# task type -> (math_domain, task_form, operation, grades where taught, risk use)
# Grades from [NIE-G1/G2/G3]; risk use from task_taxonomy.md Section 2.
TASK_TYPES = {
    "T01": ("counting", "enumeration", "none", (1, 2, 3), "control"),
    "T02": ("counting", "enumeration", "none", (1, 2, 3), "risk"),
    "T03": ("counting", "sequence", "none", (1, 2, 3), "record"),
    "T04": ("number_sense", "comparison", "none", (1, 2, 3), "risk"),
    "T05": ("number_sense", "comparison", "none", (2, 3), "risk"),
    "T06": ("number_sense", "ordering", "none", (2, 3), "record"),
    "T07": ("number_sense", "sequence", "none", (1, 2, 3), "record"),
    "T08": ("number_sense", "transcoding", "none", (1, 2, 3), "record"),
    "T09": ("place_value", "transcoding", "none", (2, 3), "record"),
    "T10": ("arithmetic", "calculation", "addition", (1, 2, 3), "risk"),
    "T11": ("arithmetic", "calculation", "addition", (2, 3), "risk"),
    "T12": ("arithmetic", "calculation", "subtraction", (1, 2, 3), "risk"),
    "T13": ("arithmetic", "calculation", "subtraction", (2, 3), "risk"),
    "T14": ("arithmetic", "calculation", "multiplication", (3,), "risk"),
    "T15": ("arithmetic", "calculation", "division", (3,), "record"),
    "T16": ("arithmetic", "word_problem", "none", (2, 3), "record"),
}

# Robot demonstration fires after > 15 s hesitation ([P-C1] p. 29, team_interface I3):
# response times at or above this are censored and excluded from RT features.
RT_CENSOR_S = 15.0

# name -> (dtype, required, source component, evidence row)
TRIAL_COLUMNS = {
    "data_origin": ("category", True, "this component", "G5"),
    "student_id": ("string", True, "student_profiles (pseudonymous)", "B1, B2b"),
    "session_id": ("string", True, "orchestrator", "B1"),
    "session_index": ("Int64", True, "orchestrator (1 = first session)", "A4, B4"),
    "question_id": ("string", True, "C2 task_id", "B1"),
    "interaction_position": ("Int64", True, "C2", "B1"),
    "age_months": ("Int64", True, "student_profiles (derived from DOB)", "B2, B2b"),
    "grade": ("Int64", True, "student_profiles", "B2, C2"),
    "c2_activity": ("category", True, "C2", "B3"),
    "c2_area": ("category", True, "C2", "B3"),
    "difficulty_tier": ("category", True, "C2", "B3"),
    "representation": ("category", False, "C2 (requested)", "C7"),
    "expected_answer": ("Float64", True, "C1", "B1"),
    "given_answer": ("Float64", False, "C1 detected_answer", "B1"),
    "accuracy": ("Int64", True, "C1 is_correct, first attempt", "B1"),
    "error_value": ("Float64", False, "C1 |expected - given|", "B1"),
    "response_time_s": ("Float64", False, "C1 response_time_ms / 1000", "B1"),
    "attempts": ("Int64", True, "C1 attempt_count", "B1"),
    "robot_demo_triggered": ("Int64", False, "C1 (optional; fallback rule when missing)", "I3, B1c"),
    "valid_trial": ("Int64", True, "C1 validation_status", "B1"),
    "behavior_label": ("category", False, "C3 (unknown = missing, I2)", "I1, I2"),
}

# Columns this component derives from the trial columns.
DERIVED_TRIAL_COLUMNS = (
    "task_type", "math_domain", "task_form", "operation", "in_curriculum",
    "robot_demo_source", "rt_valid",
)


def fill_robot_demo(trials: pd.DataFrame) -> pd.DataFrame:
    """B1c: keep C1's logged value when present; otherwise derive it from C1's rule.

    C1's rule ([P-C1] p. 29): the robot demonstrates after a wrong answer or a hesitation
    of more than 15 s. A derived value is an estimate, not an observation, so
    `robot_demo_source` always records which one a row holds ("component1" or "derived").
    """
    out = trials.copy()
    logged = out["robot_demo_triggered"].notna()
    rule = (out["accuracy"] == 0) | (out["response_time_s"].fillna(0) >= RT_CENSOR_S)
    out["robot_demo_source"] = ["component1" if x else "derived" for x in logged]
    out["robot_demo_triggered"] = out["robot_demo_triggered"].where(logged, rule.astype(int)).astype(int)
    return out


def assign_task_type(row: pd.Series) -> str | None:
    """Map a C2 activity to a taxonomy task type (task_taxonomy.md Section 4).

    Returns None when the information needed is missing (e.g. comparison without
    a representation), so the trial is kept but not used for risk.
    """
    activity, tier = row["c2_activity"], row["difficulty_tier"]
    if activity == "counting":
        if pd.isna(row["expected_answer"]):
            return None
        return "T01" if row["expected_answer"] <= 3 else "T02"  # subitizing range 1-3 [D]
    if activity == "comparison":
        return {"non_symbolic": "T04", "symbolic": "T05"}.get(row["representation"])
    if activity == "ordering":
        return "T06"
    if activity == "place_value":
        return "T08" if tier == "A" else "T09"  # assumption: Tier A = build a number, Tier B = tens/ones
    if activity == "addition":
        return "T10" if tier == "A" else "T11"
    if activity == "subtraction":
        return "T12" if tier == "A" else "T13"
    if activity == "multiplication":
        return "T14"
    if activity == "division":
        return "T15"
    return None


def validate_trials(trials: pd.DataFrame) -> list[str]:
    """Return a list of problems; an empty list means the table is valid."""
    problems = []
    missing = [c for c in TRIAL_COLUMNS if c not in trials.columns]
    if missing:
        return [f"missing columns: {missing}"]
    for name, (_, required, _, _) in TRIAL_COLUMNS.items():
        if required and trials[name].isna().any():
            problems.append(f"{name}: {int(trials[name].isna().sum())} empty values in a required column")
    checks = {
        "data_origin": DATA_ORIGINS,
        "c2_activity": C2_ACTIVITIES,
        "c2_area": C2_AREAS,
        "difficulty_tier": ("A", "B"),
        "representation": REPRESENTATIONS,
        "behavior_label": BEHAVIOR_LABELS,
        "grade": (1, 2, 3),
        "accuracy": (0, 1),
        "robot_demo_triggered": (0, 1),
        "valid_trial": (0, 1),
    }
    for name, allowed in checks.items():
        bad = trials[name].dropna()[~trials[name].dropna().isin(allowed)]
        if len(bad):
            problems.append(f"{name}: values not allowed {sorted(set(bad.astype(str)))}")
    if (trials["attempts"] < 1).any():
        problems.append("attempts: must be >= 1")
    if (trials["response_time_s"] < 0).any():
        problems.append("response_time_s: must be >= 0")
    if trials.duplicated(["student_id", "session_id", "question_id"]).any():
        problems.append("duplicate (student_id, session_id, question_id)")
    return problems


def add_derived_columns(trials: pd.DataFrame) -> pd.DataFrame:
    out = fill_robot_demo(trials)
    out["task_type"] = out.apply(assign_task_type, axis=1)
    info = out["task_type"].map(TASK_TYPES)
    out["math_domain"] = info.map(lambda v: v[0] if isinstance(v, tuple) else None)
    out["task_form"] = info.map(lambda v: v[1] if isinstance(v, tuple) else None)
    out["operation"] = info.map(lambda v: v[2] if isinstance(v, tuple) else None)
    # C3c / C3b: a task not taught at the child's grade is not evidence of difficulty.
    out["in_curriculum"] = [
        int(isinstance(v, tuple) and g in v[3]) for v, g in zip(info, out["grade"])
    ]
    # I3: RT after a robot demo or at the 15 s cut-off is not a free response time.
    out["rt_valid"] = (
        (out["valid_trial"] == 1)
        & (out["robot_demo_triggered"] == 0)
        & out["response_time_s"].notna()
        & (out["response_time_s"] < RT_CENSOR_S)
    ).astype(int)
    return out
