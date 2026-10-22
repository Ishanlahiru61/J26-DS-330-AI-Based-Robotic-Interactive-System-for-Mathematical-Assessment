# Step 3 — Feature engineering

`src/build_features.py` turns the preprocessed child records ([preprocessing.md](preprocessing.md)) into one
feature row per child. Tests: `tests/test_build_features.py`. Outputs: `artifacts/features_v1.csv`
(not in Git), `results/feature_norms_v1.json`, `results/feature_check.csv`.

## Rules

| # | Rule | Evidence |
|---|---|---|
| E1 | Features: 8 **area** features (accuracy, RT per C2 area; main input) and 13 **task** features F1–F12 | feature_evidence.md; G11 |
| E2 | z-score against **same grade, same age**: within each grade, expected value = a + b × `age_months` (regression-based norming); z = (x − expected) ÷ residual SD | DSM-5 Criterion B (age, A3); NIE (grade, C2); B2 |
| E3 | Norms fitted on the **train split only**, applied to validation and test | P11 (no leakage) |
| E4 | **Positive z = worse** (lower accuracy, slower, larger error, more attempts) | register E1 |
| E5 | z capped at ±3 | real outlier DEC-37 (+5.0) in [D] |
| E6 | Missing task type → missing feature (never 0); `n_features_available` counted | leader's `docs/screening-boundary.md` |
| E7 | Norms use **all** train children (5% DD included), as population norms do; no label is used | E2 |

Accuracy enters as the Laplace-smoothed rate (P8), RT as the median of log(1 + RT) of correct answers (P5).

## The features

| Feature | Built from | Evidence grade (feature_evidence.md) |
|---|---|---|
| `z_area_number_sense_acc/rt` | counting 4+ (T02) | same values as F6/F5 (only risk task in this area) |
| `z_area_number_relationships_acc/rt` | comparison T04 + T05 | A |
| `z_area_arithmetic_acc/rt` | T10–T13 | A (acc) / B (RT) |
| `z_area_multiplicative_acc/rt` | T14 (Grade 3 only) | extension of fact retrieval |
| `z_f1/f2_symbolic_comparison_acc/rt` | T05 (Grade 2+) | A |
| `z_f3/f4_arithmetic_acc/rt` | T10–T14 pooled | A / B |
| `z_f5/f6_counting_rt/acc` | T02 | A / B |
| `z_f7/f8_nonsymbolic_comparison_acc/rt` | T04 | A / B |
| `z_f9_subitizing_acc/rt` | T01 | control (weight 0) |
| `z_f10_relative_error`, `z_f11_attempts`, `z_f12_rt_variability` | risk task types | C (weight 0) |

Not built yet: F13 learning slope (with 2 sessions a slope is a difference of two noisy scores) and finger
counting (C3: only meaningful together with age and accuracy, I1). Both stay weight 0.

## Check on synthetic trials-v1 (`results/feature_check.csv`)

- Typical children average z ≈ −0.03 (norms include the 5% DD children, so typical children sit slightly
  below 0), and fewer than 1.2% of values hit the ±3 cap.
- DD children are worse on every Grade A feature in the train split (g 0.23–0.67). Validation and test g
  vary widely (only 22–23 DD children each), so single-split numbers must not be over-read.
- Subitizing (F9) shows no DD effect, as designed (control).
- **Two cautions for modelling:**
  1. `z_area_number_sense_*` duplicates F5/F6; a model must use one or the other, not both.
  2. F10 (relative error) and F11 (attempts) look strong (train g 0.55 and 0.88), but in the simulation they
     only change when an answer is wrong, so they **repeat accuracy**. They have no published effect size
     (Grade C) and stay out of the risk score.
- Missing: multiplicative area 67% (Grades 1–2 by design), F1/F2 38–44% (Grade 1 by design, and Grade 2–3
  children who met no symbolic comparison item).

## Limitation

The norms here are fitted on **synthetic** children and only illustrate the method. Real norms (register E2)
must be fitted on the pilot's typical children; with about 30–40 pilot children (I6), per-grade norms will be
very imprecise, which is the main evidence gap of the component.
