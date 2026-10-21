import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from mastery_data import FEATURE_SETS, TRUE_ACCURACY, mastery_labels  # noqa: E402


def test_labels_use_true_ability_with_symmetric_cutoffs():
    rng = np.random.default_rng(0)
    children = pd.DataFrame(rng.normal(0, 1, (2000, 4)), columns=TRUE_ACCURACY)
    children["child_id"] = [f"SYN-{i}" for i in range(2000)]
    tiers = mastery_labels(children)
    share = tiers.value_counts(normalize=True)
    assert set(tiers.unique()) == {0, 1, 2}
    assert abs(share[0] - share[2]) < 0.04 and 0.12 < share[2] < 0.20   # about 16% each side
    worst = children[TRUE_ACCURACY].mean(axis=1).idxmax()
    assert tiers.iloc[worst] == 2


def test_feature_sets_are_nested_for_the_ablation():
    order = ["accuracy_only", "performance", "multimodal", "historical"]
    for small, large in zip(order, order[1:]):
        assert set(FEATURE_SETS[small]) < set(FEATURE_SETS[large])
