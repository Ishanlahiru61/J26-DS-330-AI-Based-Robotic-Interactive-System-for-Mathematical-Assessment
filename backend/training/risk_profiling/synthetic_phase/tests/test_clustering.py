import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from clustering import choose_k, decide  # noqa: E402


def bic(full, diag):
    return {"full": dict(enumerate(full, start=1)), "diag": dict(enumerate(diag, start=1))}


def test_choose_k_keeps_one_group_when_bic_does_not_improve():
    assert choose_k(bic([100, 105, 110, 120, 130, 140], [200] * 6)) == (1, "full")


def test_choose_k_needs_a_real_bic_gain():
    assert choose_k(bic([100, 95, 80, 85, 90, 95], [200] * 6)) == (3, "full")   # 1->2 gain 5 < 10: kept 1 until 3


def good(**overrides):
    c = {"k": 3, "silhouette": 0.4, "bootstrap_ari_mean": 0.9, "min_share": 0.1,
         "names": ["number_sense", "arithmetic", "multiple_areas"]}
    c.update(overrides)
    return {"chosen_gmm": c}


def test_decision_rule():
    assert decide(good()) == "gmm"
    assert decide(good(k=1)) == "rule_based"
    assert decide(good(silhouette=0.1)) == "rule_based"
    assert decide(good(bootstrap_ari_mean=0.5)) == "rule_based"
    assert decide(good(min_share=0.02)) == "rule_based"
    assert decide(good(names=["arithmetic", "arithmetic", "number_sense"])) == "rule_based"
