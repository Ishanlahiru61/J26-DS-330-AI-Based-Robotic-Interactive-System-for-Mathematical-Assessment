# Step 5 — Difficulty profiles (clustering)

The risk band says **how much** concern there is; a profile says **what kind**. Profiles are given only to
children flagged at the screening cut-off (worst 35% of their grade, register F5). Result: **no distinct
data-driven profiles exist in the synthetic data**, so the system uses **rule-based profiles**; the clustering
pipeline is kept to re-run on pilot data.

## Input

C2 area features from Step 3 (accuracy z and speed z, positive = worse): number_sense (counting, place value),
number_relationships (comparison, ordering), arithmetic (addition, subtraction). These pool more questions than
F1–F8 (more reliable, G11), exist in every grade and match C2's areas. multiplicative_reasoning (Grade 3 only)
is not used, as it would be missing for Grades 1–2. 725 flagged train children; 666 with all six area values
were used for clustering, 119 validation children were assigned only. Labels were never used in fitting.

## Methods

| Method | How | Role |
|---|---|---|
| Rule-based (`src/profiles.py`) | Area z = mean of the area's accuracy and speed z (equal weights, S2); weak if ≥ 1.0 (F1). One weak area → that area; two or more → "multiple areas"; none → "no single area" | Baseline, exact, explainable |
| Gaussian Mixture Model (`src/clustering.py`) | k = 1–6, covariance full / diag, 10 starts; k by BIC (lowest; a larger k must improve BIC by ≥ 10) | Main data-driven method; also known as latent profile analysis |
| K-means | k = 2–6, 10 starts; silhouette per k | Cross-check |

## Criteria and results (`results/clustering.json`)

Decision rule, fixed before running: use GMM profiles only if silhouette ≥ 0.25, mean bootstrap ARI ≥ 0.80,
every cluster ≥ 5% of flagged children and every cluster has a distinct name; otherwise rule-based.

| Criterion | Threshold | Result | Pass |
|---|---|---|---|
| Number of groups (BIC) | k ≥ 2 | **k = 1** (full covariance: BIC 11,167 at k = 1, rising for every larger k) | ❌ |
| Silhouette (GMM, k = 2 evaluated for the record) | ≥ 0.25 | 0.09 (k-means k = 2–6: 0.14–0.16) | ❌ |
| Bootstrap stability, 200 refits | ARI ≥ 0.80 | 0.19 (95% range −0.01 to 0.58) | ❌ |
| Minimum cluster share | ≥ 5% | 34% | ✅ |
| Distinct names | all different | both "no single area": a milder and a more severe group | ❌ |
| Agreement with k-means / rule-based | reported | ARI 0.00 / 0.25 | — |
| Grade balance | reported | chi-square p = 0.92 (clusters are not grades) | — |
| Davies–Bouldin / Calinski–Harabasz | reported | 3.70 / 42.5 (weak separation) | — |

**Decision: rule-based profiles** (`config/profiles.json`, `validated: false`).

The synthetic children were generated without subtypes (one multivariate normal per group, G1–G6), so a
continuum is the expected result: GMM and k-means only split "milder" from "more severe". This confirms that the
pipeline does not invent groups. Real subgroups of mathematical difficulties have been reported in the literature
(e.g. Bartelet et al., 2014, *Research in Developmental Disabilities*; **verify reference before submission**);
whether they exist in Sri Lankan children can only be tested with pilot data by re-running `clustering.py`.

## Rule-based profiles (flagged train children)

| Profile | Children | DD share |
|---|---|---|
| No single area (flagged by the overall score) | 384 (53%) | 8% |
| Mainly number relationships | 106 | 8% |
| Mainly number sense | 106 | 10% |
| Mainly calculation | 88 | 9% |
| **Several areas** | 41 | **41%** |

Children weak in several areas are far more often DD (41% vs 8–10%), consistent with DD being a broad number
difficulty rather than one isolated skill.

## In the child report

Flagged children get one teacher line, e.g. "Difficulty profile: mainly number relationships (comparison,
ordering)". Parents see no extra line (their text already names the skills to practise).

The screening cut-off (4-S) is now part of the report: a flagged child in the low band is shown as
**"worth monitoring"** (534 of 3,000 synthetic children), so the report follows the sensitivity-oriented cut-off
recommended in 4-S instead of only the 16% / 5% bands.
