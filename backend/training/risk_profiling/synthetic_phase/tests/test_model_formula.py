import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from model_formula import WEIGHTS, band, fit_cutoffs, risk_index  # noqa: E402


def test_weights_renormalised_over_available_features():
    # Grade 1 child: no symbolic comparison (F1, F2); every other feature z = 1.
    row = {f: 1.0 for f in WEIGHTS}
    row["z_f1_symbolic_comparison_acc"] = np.nan
    row["z_f2_symbolic_comparison_rt"] = np.nan
    out = risk_index(pd.DataFrame([row]))
    assert out.loc[0, "risk_index"] == pytest.approx(1.0)        # not shrunk toward 0
    missing = WEIGHTS["z_f1_symbolic_comparison_acc"] + WEIGHTS["z_f2_symbolic_comparison_rt"]
    assert out.loc[0, "weight_coverage"] == pytest.approx(1 - missing / sum(WEIGHTS.values()))


def test_higher_weight_feature_moves_index_more():
    a = risk_index(pd.DataFrame([{**{f: 0.0 for f in WEIGHTS}, "z_f1_symbolic_comparison_acc": 2.0}]))
    b = risk_index(pd.DataFrame([{**{f: 0.0 for f in WEIGHTS}, "z_f8_nonsymbolic_comparison_rt": 2.0}]))
    assert a.loc[0, "risk_index"] > b.loc[0, "risk_index"]


def test_bands_use_same_grade_train_percentiles():
    scored = pd.DataFrame({"grade": 1, "split": "train", "risk_index": np.arange(100.0),
                           "sufficient_data_area": 1})
    cutoffs = fit_cutoffs(scored)
    assert band(pd.Series({"grade": 1, "risk_index": 99.0, "sufficient_data_area": 1}), cutoffs) == "high"
    assert band(pd.Series({"grade": 1, "risk_index": 90.0, "sufficient_data_area": 1}), cutoffs) == "moderate"
    assert band(pd.Series({"grade": 1, "risk_index": 10.0, "sufficient_data_area": 1}), cutoffs) == "low"


def test_insufficient_data_never_gets_a_risk_band():
    cutoffs = {1: {"moderate": 0.5, "high": 1.0}}
    assert band(pd.Series({"grade": 1, "risk_index": 3.0, "sufficient_data_area": 0}), cutoffs) == "insufficient"
