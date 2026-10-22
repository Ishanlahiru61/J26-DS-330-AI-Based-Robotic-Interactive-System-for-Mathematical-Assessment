import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from final_evaluation import screening, wilson  # noqa: E402


def test_wilson_interval_contains_estimate_and_stays_in_0_1():
    lo, hi = wilson(7, 10)
    assert 0 <= lo < 0.7 < hi <= 1
    assert wilson(0, 5)[0] == pytest.approx(0.0, abs=1e-9)
    assert wilson(3, 0) is None


def test_screening_counts():
    y = np.array([1, 1, 0, 0, 0])
    flagged = np.array([True, False, True, False, False])
    out = screening(y, flagged)
    assert (out["tp"], out["fn"], out["fp"], out["tn"]) == (1, 1, 1, 2)
    assert out["sensitivity"] == 0.5 and out["specificity"] == pytest.approx(0.667, abs=1e-3)
