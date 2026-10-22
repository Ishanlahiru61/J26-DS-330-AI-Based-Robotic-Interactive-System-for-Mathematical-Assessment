import math
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from preprocess import preprocess  # noqa: E402
from test_operational_datasets import trial  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))


def rows(*items):
    return pd.DataFrame(list(items))


def test_invalid_and_out_of_curriculum_trials_are_removed():
    data = rows(
        trial(question_id="Q1"),
        trial(question_id="Q2", valid_trial=0),
        trial(question_id="Q3", c2_activity="comparison", c2_area="number_relationships",
              representation="symbolic"),  # symbolic comparison in Grade 1: not taught
    )
    _, by_task, _, report = preprocess(data)
    assert report["removed_invalid_trial"] == 1
    assert report["removed_not_in_curriculum"] == 1
    assert list(by_task["task_type"]) == ["T10"]


def test_sessions_pooled_and_accuracy_smoothed():
    data = rows(
        trial(question_id="Q1", accuracy=1),
        trial(question_id="Q2", accuracy=0, given_answer=4.0, error_value=1.0),
        trial(session_id="S1-2", session_index=2, question_id="Q1", accuracy=1),
    )
    _, by_task, _, _ = preprocess(data)
    row = by_task.iloc[0]
    assert row["n_sessions"] == 2 and row["n_questions"] == 3 and row["n_correct"] == 2
    assert row["accuracy_smoothed"] == pytest.approx(3 / 5)  # (2 + 1) / (3 + 2)


def test_rt_uses_correct_uncensored_answers_on_log_scale():
    data = rows(
        trial(question_id="Q1", response_time_s=3.0),
        trial(question_id="Q2", response_time_s=15.0),                       # censored
        trial(question_id="Q3", accuracy=0, given_answer=4.0, error_value=1.0,
              response_time_s=9.0),                                           # wrong answer
    )
    _, by_task, _, _ = preprocess(data)
    assert by_task.iloc[0]["n_rt"] == 1
    assert by_task.iloc[0]["median_log_rt"] == pytest.approx(math.log1p(3.0))


def test_uncertain_behaviour_is_missing_not_zero():
    data = rows(trial(question_id="Q1", behavior_label="finger_counting"),
                trial(question_id="Q2", behavior_label="uncertain"))
    _, by_task, _, _ = preprocess(data)
    assert by_task.iloc[0]["finger_counting_rate"] == 1.0


def test_missing_risk_task_flags_insufficient_data():
    children, _, _, _ = preprocess(rows(trial()))  # Grade 1 child with addition only
    assert children.iloc[0]["sufficient_data"] == 0
    assert set(children.iloc[0]["missing_risk_tasks"].split(",")) == {"T02", "T04", "T12"}
