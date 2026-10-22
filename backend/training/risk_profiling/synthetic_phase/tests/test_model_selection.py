import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from model_selection import COMPLEXITY, compare, select  # noqa: E402


def toy(strong: str, gap: float, n=400, seed=0):
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < 0.2).astype(int)
    data = pd.DataFrame({"dd_label": y})
    for m in COMPLEXITY:
        effect = 1.0 + (gap if m == strong else 0.0)
        data[m] = rng.normal(0, 1, n) + effect * y
    return data


def test_equal_models_choose_the_simplest():
    result = compare(toy("svm", 0.0))
    assert select(result) == "formula"


def test_clearly_better_complex_model_is_chosen():
    result = compare(toy("svm", 3.0))
    assert result["best_auc_model"] == "svm"
    assert not result["models"]["formula"]["not_reliably_worse"]
    assert select(result) == "svm"
