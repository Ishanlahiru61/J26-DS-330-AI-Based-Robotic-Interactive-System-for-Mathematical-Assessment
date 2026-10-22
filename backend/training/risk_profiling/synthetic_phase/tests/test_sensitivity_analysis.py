import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from sensitivity_analysis import choose_cutoff, flag  # noqa: E402


def test_flag_uses_same_grade_train_percentile():
    scored = pd.DataFrame({"grade": [1] * 100 + [2] * 100, "split": "train",
                           "risk_index": np.r_[np.arange(100.0), np.arange(100.0) + 1000]})
    flagged = flag(scored, 70)
    assert flagged[scored["grade"] == 1].mean() == flagged[scored["grade"] == 2].mean() == 0.3


def test_choose_cutoff_takes_smallest_flag_share_reaching_target():
    rows = [{"flag_worst_pct": 16, "train": {"sensitivity": 0.47}},
            {"flag_worst_pct": 30, "train": {"sensitivity": 0.72}},
            {"flag_worst_pct": 40, "train": {"sensitivity": 0.80}}]
    assert choose_cutoff(rows, 0.70)["flag_worst_pct"] == 30
    assert choose_cutoff(rows, 0.95) is None
