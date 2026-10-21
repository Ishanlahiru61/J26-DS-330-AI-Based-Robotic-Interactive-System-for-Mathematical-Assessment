import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from model_random_forest import FEATURES, make_pipeline  # noqa: E402


def toy_data(n=300, seed=0):
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < 0.2).astype(int)
    x = pd.DataFrame(rng.normal(0, 1, (n, len(FEATURES))) + y[:, None] * 1.0, columns=FEATURES)
    x.loc[: n // 3, FEATURES[0]] = np.nan          # missing by design (e.g. Grade 1, no F1)
    return x, y


def small_forest():
    return make_pipeline().set_params(rf__n_estimators=50, rf__max_depth=3, rf__n_jobs=1)


def test_pipeline_handles_missing_features():
    x, y = toy_data()
    p = small_forest().fit(x, y).predict_proba(x)[:, 1]
    assert np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all()


def test_higher_score_for_worse_children():
    x, y = toy_data()
    model = small_forest().fit(x, y)
    worse = pd.DataFrame([[2.0] * len(FEATURES)], columns=FEATURES)
    typical = pd.DataFrame([[0.0] * len(FEATURES)], columns=FEATURES)
    assert model.predict_proba(worse)[0, 1] > model.predict_proba(typical)[0, 1]
