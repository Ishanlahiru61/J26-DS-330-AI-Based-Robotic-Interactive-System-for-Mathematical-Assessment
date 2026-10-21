import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from model_svm import FEATURES, make_pipeline  # noqa: E402


def toy_data(n=300, seed=0):
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < 0.2).astype(int)
    x = pd.DataFrame(rng.normal(0, 1, (n, len(FEATURES))) + y[:, None] * 1.0, columns=FEATURES)
    x.loc[: n // 3, FEATURES[0]] = np.nan          # missing by design (e.g. Grade 1, no F1)
    return x, y


def test_pipeline_handles_missing_features():
    x, y = toy_data()
    model = make_pipeline().fit(x, y)
    assert np.isfinite(model.decision_function(x)).all()


def test_higher_score_for_worse_children():
    x, y = toy_data()
    model = make_pipeline().fit(x, y)
    worse = pd.DataFrame([[2.0] * len(FEATURES)], columns=FEATURES)
    typical = pd.DataFrame([[0.0] * len(FEATURES)], columns=FEATURES)
    assert model.decision_function(worse)[0] > model.decision_function(typical)[0]
