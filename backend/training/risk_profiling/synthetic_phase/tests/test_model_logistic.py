import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from model_logistic import FEATURES, coefficients, make_pipeline  # noqa: E402


def toy_data(n=300, seed=0):
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < 0.2).astype(int)
    x = pd.DataFrame(rng.normal(0, 1, (n, len(FEATURES))) + y[:, None] * 1.0, columns=FEATURES)
    x.loc[: n // 3, FEATURES[0]] = np.nan          # missing by design (e.g. Grade 1, no F1)
    return x, y


def test_pipeline_handles_missing_features():
    x, y = toy_data()
    assert np.isfinite(make_pipeline().fit(x, y).decision_function(x)).all()


def test_worse_features_get_positive_weights():
    x, y = toy_data()
    coef = coefficients(make_pipeline().fit(x, y))
    assert all(w["learned"] > 0 for w in coef["weights"].values())
    assert "missing_z_f1_symbolic_comparison_acc" in coef["missing_indicators"]
