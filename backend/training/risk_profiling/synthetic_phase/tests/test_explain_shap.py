import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from explain_shap import SKILLS, by_skill, shap_values  # noqa: E402
from model_xgboost import FEATURES, imbalance, make_pipeline  # noqa: E402


def fitted():
    rng = np.random.default_rng(0)
    y = (rng.random(300) < 0.2).astype(int)
    x = pd.DataFrame(rng.normal(0, 1, (300, len(FEATURES))) + y[:, None], columns=FEATURES)
    x.loc[:100, FEATURES[0]] = np.nan
    model = make_pipeline(imbalance(y)).set_params(xgb__n_estimators=50, xgb__max_depth=2).fit(x, y)
    return model, x


def test_contributions_add_up_to_model_score():
    model, x = fitted()
    values, base = shap_values(model, x)
    margin = model.named_steps["xgb"].predict(model.named_steps["impute"].transform(x), output_margin=True)
    assert base + values.sum(axis=1).to_numpy() == pytest.approx(margin, abs=1e-4)


def test_skills_group_accuracy_speed_and_missing_indicator():
    values = pd.DataFrame([{"z_f1_symbolic_comparison_acc": 0.3, "z_f2_symbolic_comparison_rt": 0.2,
                            "missing_z_f1_symbolic_comparison_acc": 0.1, "z_f3_arithmetic_acc": -0.4}])
    skills = by_skill(values).iloc[0]
    assert skills["number_comparison"] == pytest.approx(0.6)
    assert skills["arithmetic"] == pytest.approx(-0.4)
    assert set(SKILLS) == set(FEATURES)
