import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from mastery_explain import URGENT, lime_explanation, shap_values, urgent_probability  # noqa: E402
from mastery_models import FEATURES, svm  # noqa: E402


@pytest.fixture(scope="module")
def fitted():
    rng = np.random.default_rng(0)
    y = rng.choice([0, 1, 2], size=240, p=[0.2, 0.6, 0.2])
    x = pd.DataFrame(rng.normal(0, 1, (240, len(FEATURES))), columns=FEATURES)
    x["z_overall_acc"] += (y - 1) * 1.5
    data = x.assign(mastery=y)
    return svm().fit(x, y), data


def test_shap_values_add_up_to_the_change_in_probability(fitted):
    model, data = fitted
    background, x = data[FEATURES].iloc[:40], data[FEATURES].iloc[200:205]
    values = shap_values(model, background, x)
    f = urgent_probability(model)
    expected = f(x.to_numpy()) - f(background.to_numpy()).mean()
    assert values.sum(axis=1).to_numpy() == pytest.approx(expected, abs=0.02)


def test_lime_gives_a_contribution_for_every_feature_and_flags_weak_accuracy(fitted):
    model, data = fitted
    row = data.iloc[0].copy()
    row[FEATURES] = 0.0
    row["z_overall_acc"] = 2.5                                  # much worse accuracy than peers
    contrib = lime_explanation(model, data, row)
    assert set(contrib) == set(FEATURES)
    assert max(contrib, key=lambda k: abs(contrib[k])) == "z_overall_acc"
    assert contrib["z_overall_acc"] > 0
    assert URGENT == 2
