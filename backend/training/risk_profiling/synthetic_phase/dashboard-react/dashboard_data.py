"""Data layer for the React dashboard export (synthetic phase); used by export_data.py.

Everything shown is computed by the component's own code: the evidence formula and risk levels
(model_formula, explain_formula), difficulty profiles (profiles), the selected mastery model
(mastery_models, SVM) and the plain-language reports (child_report).

Demo classes are built from the TEST children only (never used to fit or select any model),
split into classes of about 35 per grade (proposal NFR: 30-40 students).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.stats import norm

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))
from child_report import load_template, parent_report, teacher_report  # noqa: E402
from explain_formula import PRACTISE, SKILL_FEATURES, explain_child  # noqa: E402
from mastery_data import TIERS  # noqa: E402
from mastery_explain import shap_values  # noqa: E402
from mastery_models import FEATURES as MASTERY_FEATURES  # noqa: E402
from model_formula import ARTIFACTS, MODULE, RESULTS, SEED  # noqa: E402
from profiles import rule_profiles, screening_flag  # noqa: E402

CLASS_SIZE = 35
SKILLS = list(SKILL_FEATURES)
SKILL_NAMES = {"number_comparison": "Comparing numbers", "quantity_comparison": "Comparing amounts",
               "counting": "Counting", "arithmetic": "Calculation"}
LEVEL_NAMES = {"low": "No concern", "monitor": "Worth monitoring", "moderate": "Closer look",
               "high": "Closer look (priority)", "insufficient": "Not enough data"}
PROFILE_NAMES = {"number_sense": "Mainly number sense", "number_relationships": "Mainly number relationships",
                 "arithmetic": "Mainly calculation", "multiple_areas": "Several areas",
                 "no_single_area": "Mild, no single area"}
FEATURE_NAMES = {
    "z_overall_acc": "Overall accuracy", "z_area_number_sense_acc": "Number sense accuracy",
    "z_area_number_relationships_acc": "Comparison & ordering accuracy", "z_area_arithmetic_acc": "Calculation accuracy",
    "z_area_multiplicative_acc": "× and ÷ accuracy", "z_f10_relative_error": "Size of errors",
    "z_area_number_sense_rt": "Number sense speed", "z_area_number_relationships_rt": "Comparison speed",
    "z_area_arithmetic_rt": "Calculation speed", "z_area_multiplicative_rt": "× and ÷ speed",
    "z_f11_attempts": "Attempts per question", "z_f12_rt_variability": "Uneven speed",
    "z_robot_demo_rate": "Robot demonstrations", "z_finger_counting_rate": "Finger counting",
    "z_acc_trend": "Accuracy trend", "z_rt_trend": "Speed trend", "z_attempts_trend": "Attempts trend",
}


def percentile(z: float | None) -> float | None:
    """z (positive = worse) -> share of same-grade, same-age peers this child does better than."""
    return None if z is None or pd.isna(z) else float(norm.cdf(-z) * 100)


def assign_classes(df: pd.DataFrame) -> pd.Series:
    names = pd.Series(index=df.index, dtype=object)
    for grade, g in df.sort_values("student_id").groupby("grade"):
        for i, idx in enumerate(np.array_split(g.index, max(1, round(len(g) / CLASS_SIZE)))):
            names[idx] = f"Grade {grade} · Demo class {chr(65 + i)}"  # not real classes: test children split ~35 per class
    return names


def load_children() -> pd.DataFrame:
    features = pd.read_csv(ARTIFACTS / "features_v1.csv")
    preds = pd.read_csv(ARTIFACTS / "predictions_formula_v1.csv")[["student_id", "risk_index", "band"]]
    df = features.merge(preds, on="student_id")
    selected = json.loads((MODULE / "config" / "selected_model.json").read_text(encoding="utf-8"))
    df["flagged"] = screening_flag(df, selected["screening_flag_worst_pct"])
    df["profile"] = rule_profiles(df)
    df = df[df["split"] == "test"].copy()

    sessions = pd.read_csv(ARTIFACTS / "preprocessed_children_v1.csv").set_index("student_id")
    explanations = [explain_child(r, r["band"], int(sessions.at[r["student_id"], "n_sessions"]),
                                  int(sessions.at[r["student_id"], "n_questions"]), bool(r["flagged"]), r["profile"])
                    for _, r in df.iterrows()]
    df["explanation"] = explanations
    df["level"] = [e["band"] for e in explanations]
    for skill in SKILLS:
        df[f"pct_{skill}"] = [percentile(e["skills"].get(skill, {}).get("z")) for e in explanations]
        df[f"lvl_{skill}"] = [e["skills"].get(skill, {}).get("level", e["skills"].get(skill, {}).get("status"))
                              for e in explanations]

    mastery = pd.read_csv(ARTIFACTS / "mastery_features_v1.csv").set_index("student_id")
    model = joblib.load(ARTIFACTS / "mastery_svm_v1.joblib")
    have = df["student_id"].isin(mastery.index)
    x = mastery.loc[df.loc[have, "student_id"], MASTERY_FEATURES]
    proba = model.predict_proba(x)
    df["tier"] = "Not enough data"
    df.loc[have, "tier"] = [TIERS[i] for i in model.predict(x)]
    df["p_urgent"] = np.nan
    df.loc[have, "p_urgent"] = proba[:, 2]
    df["class_name"] = assign_classes(df)
    return df.reset_index(drop=True)


def load_sessions(ids: list[str]) -> pd.DataFrame:
    """Per-session accuracy and median response time (correct answers) for the given children."""
    trials = pd.read_csv(ARTIFACTS / "synthetic_trials_v1.csv",
                         usecols=["student_id", "session_index", "accuracy", "response_time_s", "c2_area"])
    trials = trials[trials["student_id"].isin(ids)]
    correct = trials[trials["accuracy"] == 1]
    out = trials.groupby(["student_id", "session_index"])["accuracy"].mean().rename("accuracy").to_frame()
    out["median_rt"] = correct.groupby(["student_id", "session_index"])["response_time_s"].median()
    return out.reset_index()


def mastery_reasons(student_id: str, n_background: int = 60) -> pd.Series | None:
    """SHAP contributions to P(Needs Urgent Assistance) for one child (selected SVM)."""
    mastery = pd.read_csv(ARTIFACTS / "mastery_features_v1.csv")
    row = mastery[mastery["student_id"] == student_id]
    if row.empty:
        return None
    model = joblib.load(ARTIFACTS / "mastery_svm_v1.joblib")
    background = mastery[mastery["split"] == "train"][MASTERY_FEATURES].sample(n_background, random_state=SEED)
    values = shap_values(model, background, row[MASTERY_FEATURES].astype(float)).iloc[0]
    return values.rename(index=FEATURE_NAMES)


def reports(explanation: dict) -> tuple[str, str]:
    t = load_template("en")
    return teacher_report(explanation, t), parent_report(explanation, t)


def results(name: str) -> dict:
    return json.loads((RESULTS / name).read_text(encoding="utf-8"))


def practise_for(explanation: dict) -> list[str]:
    return list(dict.fromkeys(a for s in explanation.get("skills", {}).values() for a in s.get("practise", [])))


__all__ = ["CLASS_SIZE", "FEATURE_NAMES", "LEVEL_NAMES", "PRACTISE", "PROFILE_NAMES", "SKILLS", "SKILL_NAMES",
           "TIERS", "load_children", "load_sessions", "mastery_reasons", "percentile", "practise_for", "reports",
           "results"]
