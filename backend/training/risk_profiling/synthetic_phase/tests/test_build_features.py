import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from build_features import Z_CAP, apply_norms, fit_norms  # noqa: E402


def raw_table(n=60, seed=0, **extra):
    """Grade-2 children with an accuracy feature (higher = better) and an RT feature (higher = worse)."""
    rng = np.random.default_rng(seed)
    age = rng.integers(84, 96, n)
    df = pd.DataFrame({
        "student_id": [f"S{i}" for i in range(n)], "grade": 2, "age_months": age,
        "split": ["train"] * (n - 2) + ["test", "test"],
        "sufficient_data": 1, "sufficient_data_area": 1,
        "f6_counting_acc": 0.5 + 0.01 * (age - 84) + rng.normal(0, 0.05, n),
        "f5_counting_rt": rng.normal(1.8, 0.2, n),
    })
    for k, v in extra.items():
        df[k] = v
    return df


def test_positive_z_means_worse_for_accuracy_and_rt():
    raw = raw_table()
    raw.loc[58, ["f6_counting_acc", "f5_counting_rt"]] = [0.2, 2.4]   # low accuracy, slow
    z = apply_norms(raw, fit_norms(raw))
    assert z.loc[58, "z_f6_counting_acc"] > 0
    assert z.loc[58, "z_f5_counting_rt"] > 0


def test_norms_use_train_split_only():
    raw = raw_table()
    raw.loc[58:, "f6_counting_acc"] = 100.0                            # extreme test children
    norms = fit_norms(raw)
    assert norms[2]["f6_counting_acc"]["n"] == 58
    assert norms[2]["f6_counting_acc"]["intercept"] < 1


def test_age_is_adjusted_within_grade():
    raw = raw_table(n=400)
    norms = fit_norms(raw)
    assert norms[2]["f6_counting_acc"]["age_slope"] == pytest.approx(0.01, abs=0.003)
    z = apply_norms(raw, norms)
    # Older children are expected to score higher, so age itself does not make z worse.
    assert abs(np.corrcoef(raw["age_months"], z["z_f6_counting_acc"])[0, 1]) < 0.1


def test_z_is_capped_and_missing_stays_missing():
    raw = raw_table()
    raw.loc[58, "f6_counting_acc"] = -50.0
    raw.loc[59, "f6_counting_acc"] = np.nan
    z = apply_norms(raw, fit_norms(raw))
    assert z.loc[58, "z_f6_counting_acc"] == Z_CAP
    assert pd.isna(z.loc[59, "z_f6_counting_acc"])
