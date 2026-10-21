"""Mastery classification (proposal 3.5) - part 1: labels and multimodal features.

Labels (proposal 3.5.1): three tiers from a predefined rubric that is INDEPENDENT of the model's
features, so the model cannot simply re-learn its own label rule. In the synthetic phase the rubric
is applied to each child's TRUE (hidden) ability, standing in for an independent teacher assessment;
the models only see the noisy session data. In the pilot the label is the teacher's rating on the
NIE 3-level mastery scale.

    ability = mean of the child's true accuracy z-scores (number sense, arithmetic, counting,
              quantity comparison), standardised over the 3,000 children; positive = worse
    Needs Urgent Assistance : ability >= 1.0   (<= 16th percentile, register F1)  NIE "Needs further help"
    High Performer          : ability <= -1.0  (>= 84th percentile, same distance) NIE "Does it successfully"
    Progressing             : otherwise                                            NIE "Shows difficulty"

Features (proposal 2.2, 3.3, 3.4), all z-scores against same grade and age (train norms), positive = worse:
    accuracy_only   overall accuracy
    performance     + topic-wise (C2 area) accuracy, error severity
    multimodal      + topic-wise speed, attempts, speed variability, robot demo rate, finger-counting rate
    historical      + session-to-session trends in accuracy, speed and attempts

Usage: python mastery_data.py   (writes artifacts/mastery_features_v1.csv, results/mastery_labels.json)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_features import apply_norms, fit_norms  # noqa: E402
from model_formula import ARTIFACTS, MODULE, RESULTS  # noqa: E402
from preprocess import clean_trials  # noqa: E402

TIERS = ["High Performer", "Progressing", "Needs Urgent Assistance"]  # class 0, 1, 2
TRUE_ACCURACY = ["z_number_sense_accuracy", "z_arithmetic_accuracy", "z_enumeration_4_7_accuracy",
                 "z_nonsymbolic_comparison_accuracy"]
CUT = 1.0  # register F1

NEW_RAW = {  # raw feature -> True if a higher value is worse
    "overall_acc": False, "robot_demo_rate": True, "finger_counting_rate": True,
    "acc_trend": False, "rt_trend": True, "attempts_trend": True,
}
AREA_ACC = [f"z_area_{a}_acc" for a in ("number_sense", "number_relationships", "arithmetic", "multiplicative")]
AREA_RT = [f"z_area_{a}_rt" for a in ("number_sense", "number_relationships", "arithmetic", "multiplicative")]
FEATURE_SETS = {
    "accuracy_only": ["z_overall_acc"],
    "performance": ["z_overall_acc", *AREA_ACC, "z_f10_relative_error"],
}
FEATURE_SETS["multimodal"] = FEATURE_SETS["performance"] + [
    *AREA_RT, "z_f11_attempts", "z_f12_rt_variability", "z_robot_demo_rate", "z_finger_counting_rate"]
FEATURE_SETS["historical"] = FEATURE_SETS["multimodal"] + ["z_acc_trend", "z_rt_trend", "z_attempts_trend"]


def mastery_labels(children: pd.DataFrame) -> pd.Series:
    ability = children[TRUE_ACCURACY].mean(axis=1)
    ability = (ability - ability.mean()) / ability.std()
    tier = np.where(ability >= CUT, 2, np.where(ability <= -CUT, 0, 1))
    return pd.Series(tier, index=children["child_id"], name="mastery")


def raw_multimodal(trials: pd.DataFrame) -> pd.DataFrame:
    df, _ = clean_trials(trials)
    g = df.groupby("student_id")
    out = pd.DataFrame({
        "overall_acc": (g["accuracy"].sum() + 1) / (g["accuracy"].size() + 2),  # Laplace (P8)
        "robot_demo_rate": g["robot_demo_triggered"].mean(),
    })
    known = df[df["behaviour_known"]]
    out["finger_counting_rate"] = known.groupby("student_id")["finger_counting"].mean()
    s = df.groupby(["student_id", "session_index"]).agg(
        acc=("accuracy", lambda a: (a.sum() + 1) / (a.size + 2)), rt=("log_rt", "median"),
        attempts=("attempts", "mean")).unstack("session_index")
    first, last = s.columns.get_level_values(1).min(), s.columns.get_level_values(1).max()
    for name, col in (("acc_trend", "acc"), ("rt_trend", "rt"), ("attempts_trend", "attempts")):
        out[name] = s[(col, last)] - s[(col, first)]
    return out


def build(trials: pd.DataFrame, features: pd.DataFrame, children: pd.DataFrame) -> pd.DataFrame:
    raw = features[["student_id", "grade", "age_months", "split", "sufficient_data", "sufficient_data_area"]] \
        .merge(raw_multimodal(trials), left_on="student_id", right_index=True, how="left")
    normed = apply_norms(raw, fit_norms(raw, NEW_RAW), NEW_RAW).drop(columns="n_features_available")
    keep = sorted({c for cols in FEATURE_SETS.values() for c in cols} - set(normed.columns))
    data = normed.merge(features[["student_id", *keep]], on="student_id")
    labels = mastery_labels(children)
    data["mastery"] = data["student_id"].map(labels)
    data["mastery_label"] = data["mastery"].map(dict(enumerate(TIERS)))
    data["dd_label"] = data["student_id"].map(children.set_index("child_id")["dd_label"])
    return data[data["sufficient_data_area"] == 1].reset_index(drop=True)


def main() -> None:
    trials = pd.read_csv(ARTIFACTS / "synthetic_trials_v1.csv")
    features = pd.read_csv(ARTIFACTS / "features_v1.csv")
    used = pd.read_csv(ARTIFACTS / "synthetic_trials_v1_children.csv")["child_id"]
    children = pd.read_csv(MODULE / "data" / "synthetic_children_v1.csv")
    children = children[children["child_id"].isin(used)]
    data = build(trials, features, children)
    data.round(4).to_csv(ARTIFACTS / "mastery_features_v1.csv", index=False)

    all_features = FEATURE_SETS["historical"]
    report = {
        "rubric": {"ability": "mean of true accuracy z (4 constructs), standardised; positive = worse",
                   "needs_urgent_assistance": f"ability >= {CUT}", "high_performer": f"ability <= -{CUT}",
                   "source": "register F1; NIE 3-level mastery scale; independent of model features"},
        "children": int(len(data)),
        "class_counts_by_split": {s: d["mastery_label"].value_counts().to_dict() for s, d in data.groupby("split")},
        "dd_share_by_tier": data.groupby("mastery_label")["dd_label"].mean().round(3).to_dict(),
        "feature_sets": FEATURE_SETS,
        "missing_share": data[all_features].isna().mean().round(3).to_dict(),
        "spearman_with_tier_train": data[data["split"] == "train"][all_features + ["mastery"]]
        .corr(method="spearman")["mastery"].drop("mastery").round(3).to_dict(),
    }
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "mastery_labels.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
