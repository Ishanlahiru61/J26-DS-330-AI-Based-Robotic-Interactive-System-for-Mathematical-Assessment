import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from mastery_ablation import step_difference  # noqa: E402


def test_step_difference_detects_a_clear_improvement():
    rng = np.random.default_rng(0)
    y = rng.choice([0, 1, 2], size=500, p=[0.2, 0.6, 0.2])
    weak = np.ones_like(y)                                  # always "Progressing"
    strong = np.where(rng.random(500) < 0.9, y, 1)
    ci = step_difference(y, weak, strong)
    assert ci["macro_f1"][0] > 0 and ci["urgent_recall"][0] > 0
