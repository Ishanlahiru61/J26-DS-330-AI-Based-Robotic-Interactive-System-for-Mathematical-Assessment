import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from child_report import TEMPLATES, join, load_template, parent_report, teacher_report  # noqa: E402
from explain_formula import explain_child  # noqa: E402
from model_formula import WEIGHTS  # noqa: E402

FORBIDDEN = ("dyscalcul", "disorder", "disabilit")


def keys(d: dict, prefix: str = "") -> set[str]:
    out = set()
    for k, v in d.items():
        out |= keys(v, f"{prefix}{k}.") if isinstance(v, dict) else {f"{prefix}{k}"}
    return out


def grade1_high_child():
    row = {"student_id": "SYN-TEST", "grade": 1, **{f: 0.0 for f in WEIGHTS}}
    row.update(z_f1_symbolic_comparison_acc=np.nan, z_f2_symbolic_comparison_rt=np.nan,
               z_f3_arithmetic_acc=2.5, z_f4_arithmetic_rt=2.0, z_f5_counting_rt=-1.5, z_f6_counting_acc=-1.5)
    return explain_child(pd.Series(row), "high", n_sessions=2, n_questions=40)


def test_every_language_file_has_all_english_keys():
    english = keys(load_template("en"))
    for path in TEMPLATES.glob("*.json"):
        missing = english - keys(json.loads(path.read_text(encoding="utf-8")))
        assert not missing, f"{path.name} is missing {sorted(missing)}"


def test_no_diagnostic_words_in_any_template():
    for path in TEMPLATES.glob("*.json"):
        text = path.read_text(encoding="utf-8").lower()
        assert not any(word in text for word in FORBIDDEN), path.name


def test_parent_text_never_mentions_skills_not_assessed():
    t = load_template("en")
    text = parent_report(grade1_high_child(), t, name="Child")
    assert t["skill_names_parent"]["number_comparison"] not in text
    assert "sums" in text                                   # the real difficulty is named


def test_teacher_text_lists_strength_first_and_not_assessed_skill():
    t = load_template("en")
    text = teacher_report(grade1_high_child(), t).splitlines()
    assert text[1].startswith("Strengths: counting")
    assert any(line.startswith("Not assessed at this grade: comparing written numbers") for line in text)
    assert any("Addition and Subtraction" in line for line in text)


def test_insufficient_child_gets_no_result():
    t = load_template("en")
    exp = {"student_id": "SYN-X", "grade": 2, "band": "insufficient", "data": {}, "skills": {}}
    assert "not enough data" in teacher_report(exp, t)
    assert "few more number games" in parent_report(exp, t)


def test_join_uses_semicolons_when_items_contain_and():
    t = {"list_and": "and"}
    assert join(["a", "b", "c"], t) == "a, b and c"
    assert join(["x", "y and z"], t) == "x; and y and z"


def test_flagged_low_band_child_is_monitor_with_profile():
    row = {"student_id": "SYN-TEST", "grade": 2, **{f: 0.0 for f in WEIGHTS}, "z_f3_arithmetic_acc": 1.2}
    exp = explain_child(pd.Series(row), "low", 2, 40, flagged=True, profile="arithmetic")
    assert exp["band"] == "monitor" and exp["profile"] == "arithmetic"
    text = teacher_report(exp, load_template("en"))
    assert "worth monitoring" in text and "Difficulty profile: mainly calculation" in text


def test_unflagged_child_gets_no_profile():
    row = {"student_id": "SYN-TEST", "grade": 2, **{f: 0.0 for f in WEIGHTS}}
    exp = explain_child(pd.Series(row), "low", 2, 40, flagged=False, profile="arithmetic")
    assert exp["band"] == "low" and exp["profile"] is None
