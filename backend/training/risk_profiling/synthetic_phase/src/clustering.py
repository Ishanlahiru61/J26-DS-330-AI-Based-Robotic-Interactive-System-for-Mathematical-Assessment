"""Step 5 (part 2) - data-driven difficulty profiles: GMM (main) and k-means (cross-check).

Children: flagged at the screening cut-off (worst 35%, F5) with all six area z-scores (profiles.py).
Fit on TRAIN flagged children; validation flagged children are only assigned. Labels are never used
in fitting (DD share per cluster is reported for information).

Methods
- Gaussian Mixture Model: k = 1..6, covariance "full" and "diag", n_init = 10. k chosen by BIC
  (lowest; a larger k must improve BIC by >= 10, otherwise the smaller k is kept).
- K-means: k = 2..6, n_init = 10; silhouette per k (cross-check).

Criteria for the chosen GMM: silhouette, Davies-Bouldin, Calinski-Harabasz, minimum cluster share,
mean assignment probability, bootstrap stability (200 refits, Adjusted Rand Index), agreement with
k-means and with the rule-based profiles (ARI), grade balance (chi-square), train vs validation
cluster means, and a name for every cluster (the rule-based rule applied to the cluster centre).

Decision rule (fixed before running): use the GMM profiles only if silhouette >= 0.25, mean bootstrap
ARI >= 0.80, every cluster >= 5% of flagged children, and every cluster has a distinct name.
Otherwise use the rule-based profiles (profiles.py) and keep this pipeline for pilot data.

Usage: python clustering.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency
from sklearn.cluster import KMeans
from sklearn.metrics import (adjusted_rand_score, calinski_harabasz_score, davies_bouldin_score,
                             silhouette_score)
from sklearn.mixture import GaussianMixture

sys.path.insert(0, str(Path(__file__).resolve().parent))
from model_formula import ARTIFACTS, MODULE, RESULTS, SEED  # noqa: E402
from profiles import AREA_COLUMNS, AREAS, WEAK_Z, rule_profile, rule_profiles, screening_flag  # noqa: E402

K_RANGE = range(1, 7)
COVARIANCES = ["full", "diag"]
BIC_MIN_GAIN = 10.0
N_BOOT = 200
CRITERIA = {"silhouette": 0.25, "bootstrap_ari": 0.80, "min_share": 0.05}


def gmm(k: int, cov: str, seed: int = SEED) -> GaussianMixture:
    return GaussianMixture(n_components=k, covariance_type=cov, n_init=10, random_state=seed)


def choose_k(bic: dict) -> tuple[int, str]:
    """Best covariance type by its lowest BIC; then the smallest k within BIC_MIN_GAIN steps."""
    cov = min(COVARIANCES, key=lambda c: min(bic[c].values()))
    k = 1
    for nxt in K_RANGE[1:]:
        if bic[cov][k] - bic[cov][nxt] >= BIC_MIN_GAIN and bic[cov][nxt] == min(bic[cov][j] for j in range(1, nxt + 1)):
            k = nxt
    return k, cov


def name_clusters(centres: pd.DataFrame) -> list[str]:
    return [rule_profile(centres.loc[i, AREAS]) for i in centres.index]


def bootstrap_ari(x: np.ndarray, k: int, cov: str, reference: np.ndarray) -> list[float]:
    rng = np.random.default_rng(SEED)
    out = []
    for b in range(N_BOOT):
        idx = rng.integers(0, len(x), len(x))
        model = GaussianMixture(n_components=k, covariance_type=cov, n_init=3, random_state=b).fit(x[idx])
        out.append(adjusted_rand_score(reference, model.predict(x)))
    return out


def centres_by_area(x: pd.DataFrame, labels: np.ndarray) -> pd.DataFrame:
    z = pd.DataFrame({a: x[[f"z_area_{a}_acc", f"z_area_{a}_rt"]].mean(axis=1) for a in AREAS}, index=x.index)
    return z.groupby(labels).mean()


def decide(report: dict) -> str:
    c = report["chosen_gmm"]
    if c["k"] < 2:
        return "rule_based"
    ok = (c["silhouette"] >= CRITERIA["silhouette"] and c["bootstrap_ari_mean"] >= CRITERIA["bootstrap_ari"]
          and c["min_share"] >= CRITERIA["min_share"] and len(set(c["names"])) == c["k"])
    return "gmm" if ok else "rule_based"


def main() -> None:
    features = pd.read_csv(ARTIFACTS / "features_v1.csv")
    scores = pd.read_csv(ARTIFACTS / "predictions_formula_v1.csv")[["student_id", "risk_index"]]
    labels = pd.read_csv(ARTIFACTS / "synthetic_trials_v1_children.csv").set_index("child_id")["dd_label"]
    data = features.merge(scores, on="student_id")
    data["dd_label"] = data["student_id"].map(labels)
    flag_pct = 100 - json.loads((RESULTS / "sensitivity_analysis.json").read_text(encoding="utf-8"))["screening_percentile_formula"]
    flagged = data[screening_flag(data, flag_pct) & (data["split"] != "test")]
    complete = flagged.dropna(subset=AREA_COLUMNS)
    train, val = complete[complete["split"] == "train"], complete[complete["split"] == "validation"]
    x = train[AREA_COLUMNS].to_numpy()

    bic = {c: {k: float(gmm(k, c).fit(x).bic(x)) for k in K_RANGE} for c in COVARIANCES}
    k, cov = choose_k(bic)
    kmeans_sil = {kk: float(silhouette_score(x, KMeans(kk, n_init=10, random_state=SEED).fit_predict(x)))
                  for kk in range(2, 7)}

    # Criteria are computed for the BIC choice; if BIC chose k = 1, also for the best k >= 2 so the
    # report shows why no profiles were accepted.
    k_eval = k if k >= 2 else min(range(2, 7), key=lambda kk: bic[cov][kk])
    model = gmm(k_eval, cov).fit(x)
    lab = model.predict(x)
    centres = centres_by_area(train[AREA_COLUMNS], lab)
    boots = bootstrap_ari(x, k_eval, cov, lab)
    km = KMeans(k_eval, n_init=10, random_state=SEED).fit_predict(x)
    rule = rule_profiles(train).to_numpy()
    grade_table = pd.crosstab(lab, train["grade"])
    val_lab = model.predict(val[AREA_COLUMNS].to_numpy())

    report = {
        "children": {"flagged_train": int(len(flagged[flagged["split"] == "train"])),
                     "used_train_complete_areas": int(len(train)), "used_validation": int(len(val)),
                     "flag_worst_pct": flag_pct},
        "bic": {c: {str(kk): round(v, 1) for kk, v in d.items()} for c, d in bic.items()},
        "bic_choice": {"k": k, "covariance": cov},
        "kmeans_silhouette": {str(kk): round(v, 3) for kk, v in kmeans_sil.items()},
        "chosen_gmm": {
            "k": k, "k_evaluated": k_eval, "covariance": cov,
            "silhouette": round(float(silhouette_score(x, lab)), 3),
            "davies_bouldin": round(float(davies_bouldin_score(x, lab)), 3),
            "calinski_harabasz": round(float(calinski_harabasz_score(x, lab)), 1),
            "shares": [round(float(s), 3) for s in np.bincount(lab, minlength=k_eval) / len(lab)],
            "min_share": round(float(np.bincount(lab, minlength=k_eval).min() / len(lab)), 3),
            "mean_assignment_probability": round(float(model.predict_proba(x).max(axis=1).mean()), 3),
            "bootstrap_ari_mean": round(float(np.mean(boots)), 3),
            "bootstrap_ari_ci95": [round(float(v), 3) for v in np.percentile(boots, [2.5, 97.5])],
            "ari_vs_kmeans": round(float(adjusted_rand_score(lab, km)), 3),
            "ari_vs_rule_based": round(float(adjusted_rand_score(lab, rule)), 3),
            "grade_chi2_p": round(float(chi2_contingency(grade_table)[1]), 3),
            "names": name_clusters(centres),
            "centres_area_z_train": centres.round(2).to_dict(orient="index"),
            "centres_area_z_validation": centres_by_area(val[AREA_COLUMNS], val_lab).round(2).to_dict(orient="index"),
            "dd_share_per_cluster": train.groupby(lab)["dd_label"].mean().round(3).to_dict(),
        },
        "rule_based": {"weak_z": WEAK_Z,
                       "train_counts": rule_profiles(flagged[flagged["split"] == "train"]).value_counts().to_dict(),
                       "dd_share": flagged[flagged["split"] == "train"].groupby(
                           rule_profiles(flagged[flagged["split"] == "train"]))["dd_label"].mean().round(3).to_dict()},
        "criteria": CRITERIA,
    }
    report["decision"] = decide(report)
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "clustering.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    (MODULE / "config" / "profiles.json").write_text(json.dumps({
        "validated": False,
        "status": "UNVALIDATED: synthetic data have no built-in subtypes; re-run clustering.py on pilot data",
        "method": report["decision"], "areas": AREAS, "weak_z": WEAK_Z, "flag_worst_pct": flag_pct,
        "profiles": AREAS + ["multiple_areas", "no_single_area"],
    }, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
    main()
