import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from build_session_summary import build_summary  # noqa: E402
from dataset_schema import add_derived_columns, validate_trials  # noqa: E402


def trial(**overrides):
    row = {
        "data_origin": "demo", "student_id": "S1", "session_id": "S1-1", "session_index": 1,
        "question_id": "Q1", "interaction_position": 1, "age_months": 80, "grade": 1,
        "c2_activity": "addition", "c2_area": "arithmetic_calculation", "difficulty_tier": "A",
        "representation": None, "expected_answer": 5.0, "given_answer": 5.0, "accuracy": 1,
        "error_value": 0.0, "response_time_s": 4.0, "attempts": 1,
        "robot_demo_triggered": 0, "valid_trial": 1, "behavior_label": None,
    }
    row.update(overrides)
    return row


def test_symbolic_comparison_is_not_grade_1_curriculum():
    rows = pd.DataFrame([trial(c2_activity="comparison", c2_area="number_relationships",
                               representation="symbolic", grade=g, question_id=f"Q{g}") for g in (1, 2)])
    out = add_derived_columns(rows)
    assert list(out["task_type"]) == ["T05", "T05"]
    assert list(out["in_curriculum"]) == [0, 1]


def test_multiplication_only_grade_3():
    rows = pd.DataFrame([trial(c2_activity="multiplication", c2_area="multiplicative_reasoning",
                               grade=g, question_id=f"Q{g}") for g in (2, 3)])
    assert list(add_derived_columns(rows)["in_curriculum"]) == [0, 1]


def test_subitizing_separated_from_counting():
    rows = pd.DataFrame([trial(c2_activity="counting", c2_area="number_sense", expected_answer=e,
                               question_id=f"Q{e}") for e in (3, 4)])
    assert list(add_derived_columns(rows)["task_type"]) == ["T01", "T02"]


def test_comparison_without_representation_is_unassigned():
    rows = pd.DataFrame([trial(c2_activity="comparison", c2_area="number_relationships")])
    assert add_derived_columns(rows)["task_type"].isna().all()


def test_rt_after_robot_demo_or_at_cutoff_is_excluded():
    rows = pd.DataFrame([trial(question_id="Q1"), trial(question_id="Q2", robot_demo_triggered=1),
                         trial(question_id="Q3", response_time_s=15.0)])
    assert list(add_derived_columns(rows)["rt_valid"]) == [1, 0, 0]


def test_component1_robot_demo_value_has_priority():
    # C1 logged 0 although the answer is wrong (e.g. robot switched off): C1's value is kept.
    rows = pd.DataFrame([trial(accuracy=0, given_answer=4.0, error_value=1.0, robot_demo_triggered=0)])
    out = add_derived_columns(rows)
    assert out.loc[0, "robot_demo_triggered"] == 0
    assert out.loc[0, "robot_demo_source"] == "component1"


def test_missing_robot_demo_is_derived_from_rule_and_marked():
    rows = pd.DataFrame([
        trial(question_id="Q1", accuracy=0, given_answer=4.0, error_value=1.0, robot_demo_triggered=None),
        trial(question_id="Q2", response_time_s=15.0, robot_demo_triggered=None),
        trial(question_id="Q3", response_time_s=4.0, robot_demo_triggered=None),
    ])
    out = add_derived_columns(rows)
    assert list(out["robot_demo_triggered"]) == [1, 1, 0]
    assert set(out["robot_demo_source"]) == {"derived"}


def test_summary_keeps_logged_and_derived_demo_rates_apart():
    rows = [trial(question_id="Q1", robot_demo_triggered=1, accuracy=0, given_answer=4.0, error_value=1.0),
            trial(question_id="Q2", robot_demo_triggered=None, accuracy=0, given_answer=4.0, error_value=1.0)]
    summary = build_summary(pd.DataFrame(rows))
    assert summary.loc[0, "robot_demo_rate_component1"] == 1.0
    assert summary.loc[0, "robot_demo_rate_derived"] == 1.0
    assert summary.loc[0, "n_robot_demo_derived"] == 1


def test_invalid_values_are_rejected():
    rows = pd.DataFrame([trial(grade=4, accuracy=2)])
    problems = validate_trials(rows)
    assert any("grade" in p for p in problems) and any("accuracy" in p for p in problems)
    with pytest.raises(ValueError):
        build_summary(rows)


def test_summary_rates_and_slope():
    rows = [trial(question_id="Q1", accuracy=1), trial(question_id="Q2", accuracy=0, given_answer=4.0,
                                                       error_value=1.0, attempts=2),
            trial(session_id="S1-2", session_index=2, question_id="Q1", accuracy=1),
            trial(session_id="S1-2", session_index=2, question_id="Q2", accuracy=1)]
    summary = build_summary(pd.DataFrame(rows))
    assert list(summary["accuracy_rate"]) == [0.5, 1.0]
    assert pd.isna(summary.loc[0, "improvement_slope"])
    assert summary.loc[1, "improvement_slope"] == pytest.approx(0.5)
    assert list(summary["mastery_level"]) == ["Shows difficulty", "Does it successfully"]
