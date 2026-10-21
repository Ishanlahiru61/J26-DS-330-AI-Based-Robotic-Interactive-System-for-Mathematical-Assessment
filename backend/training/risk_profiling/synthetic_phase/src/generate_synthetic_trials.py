"""Generate synthetic cube_assessment_trials from synthetic_children_v1 (trials-v1).

Each synthetic child keeps its evidence-based z-scores and DD label
(synthetic_children_v1, register G1-G4). Those z-scores decide how the child
answers each question; config/simulation_parameters.json gives the absolute
levels (mostly PLACEHOLDERS) and the session design (C2, NIE, pilot plan).

Rule (G8): a task type without a published DD effect gets an independent
child-level z-score with no DD shift, so no effect is simulated without evidence.

Outputs (artifacts/ is ignored by Git; rebuild with this script):
- artifacts/synthetic_trials_v1.csv, artifacts/synthetic_summary_v1.csv
- data/synthetic_trials_v1_sample.csv (first 30 children)
- results/synthetic_trials_check.csv (DD-vs-typical effect per task type)
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_session_summary import build_summary  # noqa: E402
from dataset_schema import TASK_TYPES, add_derived_columns, validate_trials  # noqa: E402

MODULE = Path(__file__).resolve().parents[1]
REPO = Path(__file__).resolve().parents[5]
CHILDREN = MODULE / "data" / "synthetic_children_v1.csv"
ARTIFACTS = MODULE / "artifacts"
SAMPLE = MODULE / "data" / "synthetic_trials_v1_sample.csv"
CHECK = MODULE / "results" / "synthetic_trials_check.csv"
P = json.loads((MODULE / "config" / "simulation_parameters.json").read_text(encoding="utf-8"))

ACTIVITIES_BY_GRADE = {  # nie_curriculum_map.md Section 0
    1: ["counting", "comparison", "place_value", "addition", "subtraction"],
    2: ["counting", "comparison", "ordering", "place_value", "addition", "subtraction"],
    3: ["counting", "comparison", "ordering", "place_value", "addition", "subtraction",
        "multiplication", "division"],
}
AREA = {
    "counting": "number_sense", "place_value": "number_sense",
    "comparison": "number_relationships", "ordering": "number_relationships",
    "addition": "arithmetic_calculation", "subtraction": "arithmetic_calculation",
    "multiplication": "multiplicative_reasoning", "division": "multiplicative_reasoning",
}
# Task type -> (accuracy z column, RT z column) in synthetic_children_v1. None = no evidence (G8).
FEATURE_FOR_TASK = {
    "T02": ("z_enumeration_4_7_accuracy", "z_enumeration_4_7_rt"),
    "T04": ("z_nonsymbolic_comparison_accuracy", "z_nonsymbolic_comparison_rt"),
    "T05": ("z_number_sense_accuracy", "z_number_sense_rt"),
    "T10": ("z_arithmetic_accuracy", "z_arithmetic_rt"),
    "T11": ("z_arithmetic_accuracy", "z_arithmetic_rt"),
    "T12": ("z_arithmetic_accuracy", "z_arithmetic_rt"),
    "T13": ("z_arithmetic_accuracy", "z_arithmetic_rt"),
    "T14": ("z_arithmetic_accuracy", "z_arithmetic_rt"),  # fact-retrieval extension (taxonomy T14)
}


def select_children(rng: np.random.Generator) -> pd.DataFrame:
    children = pd.read_csv(CHILDREN)
    n, n_dd = P["n_children"]["value"], P["n_children"]["dd"]
    dd = children[children["dd_label"] == 1].sample(n_dd, random_state=P["seed"])
    typical = children[children["dd_label"] == 0].sample(n - n_dd, random_state=P["seed"])
    chosen = pd.concat([dd, typical]).sample(frac=1, random_state=P["seed"]).reset_index(drop=True)
    chosen["grade"] = rng.permutation(np.resize(P["grades"]["value"], n))
    low = chosen["grade"].map(lambda g: P["age_months_by_grade"][str(g)][0])
    high = chosen["grade"].map(lambda g: P["age_months_by_grade"][str(g)][1])
    chosen["age_months"] = rng.integers(low, high + 1)
    return chosen


def expected_answer(activity: str, tier: str, grade: int, rng: np.random.Generator) -> int:
    top = P["answer_range_max"][str(grade)]
    if activity == "counting":  # set size, independent of tier: subitizing (1-3) or counting (4+)
        if rng.random() < P["counting_subitizing_share"]["value"]:
            return int(rng.integers(1, 4))
        return int(rng.integers(4, P["answer_range_max"]["counting_max"][str(grade)] + 1))
    if activity in ("multiplication", "division"):
        return int(rng.integers(2, 21))
    return int(rng.integers(1, top + 1))


def build_questions(children: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    rows = []
    q = P["questions_per_session"]
    for child in children.itertuples():
        grade = int(child.grade)
        b_share = P["tier_B_share"]["grade_1" if grade == 1 else "grade_2_3"]
        sym_share = P["comparison_symbolic_share"]["grade_1" if grade == 1 else "grade_2_3"]
        for session in range(1, P["sessions_per_child"]["value"] + 1):
            for position in range(1, int(rng.integers(q["min"], q["max"] + 1)) + 1):
                activity = str(rng.choice(ACTIVITIES_BY_GRADE[grade]))
                tier = "B" if rng.random() < b_share else "A"
                representation = None
                if activity == "comparison":
                    representation = "symbolic" if rng.random() < sym_share else "non_symbolic"
                rows.append({
                    "data_origin": "synthetic",
                    "student_id": child.child_id,
                    "session_id": f"{child.child_id}-S{session}",
                    "session_index": session,
                    "question_id": f"Q{position:02d}",
                    "interaction_position": position,
                    "age_months": int(child.age_months) + (session - 1) * P["months_between_sessions"]["value"],
                    "grade": grade,
                    "c2_activity": activity,
                    "c2_area": AREA[activity],
                    "difficulty_tier": tier,
                    "representation": representation,
                    "expected_answer": float(expected_answer(activity, tier, grade, rng)),
                })
    return pd.DataFrame(rows)


def simulate_answers(questions: pd.DataFrame, children: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    derived = add_derived_columns(questions.assign(
        given_answer=np.nan, accuracy=0, error_value=np.nan, response_time_s=np.nan,
        attempts=1, robot_demo_triggered=0, valid_trial=1, behavior_label=None,
    ))
    task = derived["task_type"].to_numpy()
    by_child = children.set_index("child_id")

    # Independent child-level z for task types without evidence (G8), fixed across sessions.
    no_evidence = {t for t in TASK_TYPES if t not in FEATURE_FOR_TASK}
    independent = {(c, t): rng.standard_normal(2) for c in children["child_id"] for t in no_evidence}

    z_acc = np.empty(len(derived))
    z_rt = np.empty(len(derived))
    for i, (child, t) in enumerate(zip(derived["student_id"], task)):
        if t in FEATURE_FOR_TASK:
            acc_col, rt_col = FEATURE_FOR_TASK[t]
            z_acc[i], z_rt[i] = by_child.at[child, acc_col], by_child.at[child, rt_col]
        else:
            z_acc[i], z_rt[i] = independent[(child, t)]

    tier = derived["difficulty_tier"].to_numpy()
    p0 = np.where(tier == "A", P["typical_p_correct"]["A"], P["typical_p_correct"]["B"])
    logit = np.log(p0 / (1 - p0)) - P["logit_per_z"]["value"] * z_acc  # z positive = worse
    accuracy = (rng.random(len(derived)) < 1 / (1 + np.exp(-logit))).astype(int)

    rt0 = np.where(tier == "A", P["typical_rt_s"]["A"], P["typical_rt_s"]["B"])
    log_rt = (np.log(rt0) + P["log_rt_per_z"]["value"] * z_rt
              + rng.normal(0, P["log_rt_trial_sd"]["value"], len(derived)))
    rt = np.minimum(np.round(np.exp(log_rt), 2), P["rt_censor_s"]["value"])

    off = P["wrong_answer_offsets"]
    offsets = rng.choice(off["values"], size=len(derived), p=off["weights"])
    expected = derived["expected_answer"].to_numpy()
    given = np.where(accuracy == 1, expected, np.maximum(0, expected + offsets))
    wrong = accuracy == 0

    out = questions.copy()
    out["given_answer"] = given
    out["accuracy"] = accuracy
    out["error_value"] = np.abs(expected - given)
    out["response_time_s"] = rt
    out["attempts"] = np.where(wrong, rng.choice(P["wrong_attempts"]["values"], size=len(out)), 1)
    # C1: the robot demonstrates after a wrong answer or a hesitation of 15 s or more ([P-C1] p. 29).
    out["robot_demo_triggered"] = (wrong | (rt >= P["rt_censor_s"]["value"])).astype(int)
    out["valid_trial"] = (rng.random(len(out)) < P["valid_trial_prob"]["value"]).astype(int)
    probs = P["behaviour_probs"]
    labels = ["finger_counting", "other_hand_movement", "uncertain"]
    out["behavior_label"] = rng.choice(labels, size=len(out), p=[probs[k] for k in labels])
    return out


def hedges_g(worse: np.ndarray, better: np.ndarray) -> float:
    n1, n2 = len(worse), len(better)
    pooled = math.sqrt(((n1 - 1) * worse.var(ddof=1) + (n2 - 1) * better.var(ddof=1)) / (n1 + n2 - 2))
    return (worse.mean() - better.mean()) / pooled * (1 - 3 / (4 * (n1 + n2) - 9))


def check_effects(summary: pd.DataFrame, children: pd.DataFrame) -> pd.DataFrame:
    """Child-level DD-vs-typical g on pooled error rate and RT, per task type (positive = DD worse)."""
    labels = children.set_index("child_id")["dd_label"]
    rows = []
    for t, group in summary.groupby("task_type"):
        child = group.groupby("student_id").agg(error_rate=("error_rate", "mean"),
                                                median_rt=("median_rt_correct_s", "mean"))
        child["dd"] = child.index.map(labels)
        row = {"task_type": t, "risk_use": TASK_TYPES.get(t, ("",) * 5)[4], "n_children": len(child)}
        for col in ("error_rate", "median_rt"):
            data = child.dropna(subset=[col])
            if data["dd"].sum() >= 2:
                row[f"g_{col}"] = round(hedges_g(data.loc[data.dd == 1, col].to_numpy(),
                                                  data.loc[data.dd == 0, col].to_numpy()), 3)
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    rng = np.random.default_rng(P["seed"])
    children = select_children(rng)
    trials = simulate_answers(build_questions(children, rng), children, rng)
    problems = validate_trials(trials)
    if problems:
        raise ValueError(problems)

    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    trials.to_csv(ARTIFACTS / "synthetic_trials_v1.csv", index=False)
    first = set(children["child_id"].head(30))
    trials[trials["student_id"].isin(first)].to_csv(SAMPLE, index=False)

    summary = build_summary(trials)
    summary.round(4).to_csv(ARTIFACTS / "synthetic_summary_v1.csv", index=False)
    children[["child_id", "grade", "age_months", "dd_label"]].to_csv(
        ARTIFACTS / "synthetic_trials_v1_children.csv", index=False)

    check = check_effects(summary, children)
    CHECK.parent.mkdir(parents=True, exist_ok=True)
    check.to_csv(CHECK, index=False)
    print(f"{len(trials)} trials, {len(summary)} summary rows, {len(children)} children "
          f"({int(children['dd_label'].sum())} DD)")
    print(check.to_string(index=False))


if __name__ == "__main__":
    main()
