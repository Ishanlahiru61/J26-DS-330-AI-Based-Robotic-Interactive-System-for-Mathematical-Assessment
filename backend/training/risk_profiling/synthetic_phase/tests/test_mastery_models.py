import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from mastery_models import FEATURES, StackedMastery, accuracy_rule, metrics, select  # noqa: E402


def toy(n=300, seed=0):
    rng = np.random.default_rng(seed)
    y = rng.choice([0, 1, 2], size=n, p=[0.2, 0.6, 0.2])
    x = pd.DataFrame(rng.normal(0, 1, (n, len(FEATURES))), columns=FEATURES)
    x["z_overall_acc"] += (y - 1) * 1.5
    x.loc[: n // 2, "z_area_multiplicative_acc"] = np.nan        # Grades 1-2 have no x/÷
    return x, y


def test_accuracy_rule_baseline():
    x = pd.DataFrame({"z_overall_acc": [1.2, 0.0, -1.5]})
    assert accuracy_rule(x).tolist() == [2, 1, 0]


def test_stacking_meta_learner_uses_out_of_fold_probabilities():
    x, y = toy()
    model = StackedMastery({"rf__n_estimators": 30, "rf__n_jobs": 1}, {}, {"n_estimators": 20, "max_depth": 2})
    oof = model.oof_probabilities(x, y)
    assert oof.shape == (len(y), 6)                       # 3 RF + 3 SVM class probabilities
    model.fit(x, y, meta_x=oof)
    proba = model.predict_proba(x)
    assert proba.shape == (len(y), 3) and np.allclose(proba.sum(axis=1), 1, atol=1e-5)
    assert (model.predict(x) == y).mean() > 0.6


def test_metrics_and_selection_prefer_simplest_equal_model():
    rng = np.random.default_rng(1)
    y = rng.choice([0, 1, 2], size=400, p=[0.2, 0.6, 0.2])
    good = np.where(rng.random(400) < 0.8, y, 1)
    preds = {m: good.copy() for m in ("accuracy_rule", "random_forest", "xgboost", "svm", "stacking")}
    chosen, _ = select(y, preds)
    assert chosen == "accuracy_rule"
    m = metrics(y, good, None)
    assert m["confusion_matrix"]["values"][0][0] + m["confusion_matrix"]["values"][0][1] == int((y == 0).sum())


def test_selection_rejects_model_that_misses_urgent_children():
    rng = np.random.default_rng(2)
    y = rng.choice([0, 1, 2], size=600, p=[0.2, 0.6, 0.2])
    good = np.where(rng.random(600) < 0.8, y, 1)
    misses = np.where(y == 2, 1, good)                       # never predicts Needs Urgent Assistance
    preds = {"accuracy_rule": misses, "random_forest": good, "xgboost": good, "svm": good, "stacking": good}
    assert select(y, preds)[0] == "random_forest"
