import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from explain_formula import explain_child, level  # noqa: E402
from model_formula import WEIGHTS, risk_index  # noqa: E402


def child(grade=2, **z):
    row = {"student_id": "SYN-TEST", "grade": grade, **{f: 0.0 for f in WEIGHTS}}
    row.update(z)
    return pd.Series(row)


def test_contributions_add_up_to_formula_score():
    row = child(z_f3_arithmetic_acc=2.0, z_f5_counting_rt=1.2, z_f7_nonsymbolic_comparison_acc=-0.5)
    out = explain_child(row, "high")
    total = sum(s["contribution"] for s in out["skills"].values())
    assert total == pytest.approx(risk_index(pd.DataFrame([row]))["risk_index"][0], abs=1e-3)


def test_levels_follow_register_cutoffs():
    assert level(1.7) == "needs_extra_help"
    assert level(1.0) == "needs_practice"
    assert level(0.3) == "as_expected"
    assert level(-1.2) == "strength"


def test_grade1_comparison_is_not_assessed_and_not_a_reason():
    row = child(grade=1, z_f1_symbolic_comparison_acc=np.nan, z_f2_symbolic_comparison_rt=np.nan,
                z_f3_arithmetic_acc=2.5)
    out = explain_child(row, "high")
    assert out["skills"]["number_comparison"] == {"status": "not_assessed"}
    assert out["not_assessed"] == ["number_comparison"]
    assert out["main_reasons"] == ["arithmetic"]


def test_practise_only_uses_c2_activities_of_the_grade():
    out1 = explain_child(child(grade=1, z_f3_arithmetic_acc=3.0, z_f4_arithmetic_rt=3.0), "high")
    out3 = explain_child(child(grade=3, z_f3_arithmetic_acc=3.0, z_f4_arithmetic_rt=3.0), "high")
    assert out1["skills"]["arithmetic"]["practise"] == ["addition", "subtraction"]
    assert "multiplication" in out3["skills"]["arithmetic"]["practise"]


def test_insufficient_data_gets_no_skill_levels():
    out = explain_child(child(), "insufficient")
    assert out["band"] == "insufficient" and out["skills"] == {}
