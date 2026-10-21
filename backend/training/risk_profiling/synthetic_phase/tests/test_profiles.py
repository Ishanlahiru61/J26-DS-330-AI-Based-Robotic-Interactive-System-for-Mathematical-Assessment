import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from profiles import AREA_COLUMNS, rule_profiles, screening_flag  # noqa: E402


def rows(*areas):
    """One child per tuple of (number_sense, number_relationships, arithmetic) area z."""
    data = []
    for ns, nr, ar in areas:
        data.append({"z_area_number_sense_acc": ns, "z_area_number_sense_rt": ns,
                     "z_area_number_relationships_acc": nr, "z_area_number_relationships_rt": nr,
                     "z_area_arithmetic_acc": ar, "z_area_arithmetic_rt": ar})
    return pd.DataFrame(data)[AREA_COLUMNS]


def test_rule_profiles():
    out = rule_profiles(rows((1.5, 0.2, 0.1), (0.1, 0.0, 2.0), (1.2, 1.3, 0.0), (0.5, 0.6, 0.7)))
    assert out.tolist() == ["number_sense", "arithmetic", "multiple_areas", "no_single_area"]


def test_missing_speed_uses_accuracy_only():
    data = rows((0.0, 0.0, 0.0))
    data["z_area_arithmetic_acc"], data["z_area_arithmetic_rt"] = 1.4, np.nan
    assert rule_profiles(data).tolist() == ["arithmetic"]


def test_screening_flag_is_per_grade_and_needs_sufficient_data():
    scored = pd.DataFrame({"grade": [1] * 100 + [2] * 100, "split": "train",
                           "risk_index": np.r_[np.arange(100.0), np.arange(100.0) + 50],
                           "sufficient_data_area": 1})
    scored.loc[99, "sufficient_data_area"] = 0
    flagged = screening_flag(scored, 35)
    assert flagged[scored["grade"] == 2].mean() == 0.35
    assert not flagged[99]
