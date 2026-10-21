# Step 4 — Models

Models are added one at a time. Every model uses the same input (`artifacts/features_v1.csv`), the same split
(train / validation; the test split and the 14 real Decarli test children stay untouched until Step 6) and the
same metrics (AUC with bootstrap 95% CI, sensitivity and specificity at the risk bands). Reporting follows
TRIPOD+AI.

| Model | Status | Script |
|---|---|---|
| 4a Evidence-weighted formula | ✅ done | `src/model_formula.py` |
| 4b SVM (proposal) | ✅ done | `src/model_svm.py` |
| 4-S Sensitivity analysis (cut-off, two-stage, more sessions) | ✅ done | `src/sensitivity_analysis.py` |
| 4c Random Forest (proposal) | ✅ done | `src/model_random_forest.py` |
| 4d XGBoost + SHAP (proposal) | ✅ done | `src/model_xgboost.py`, `src/explain_shap.py` |
| 4e Logistic regression (extra reference, TRIPOD+AI) | ✅ done | `src/model_logistic.py` |
| 4f Model selection | ✅ done: **formula selected** | `src/model_selection.py` |
| 4g Explainable child report | ✅ done | `src/explain_formula.py`, `src/child_report.py`, [child_report.md](child_report.md) |

The proposal names SVM, Random Forest and XGBoost. Logistic regression is added as a reference because
TRIPOD+AI expects ML models to be compared with a regression baseline, and a systematic review found no
performance benefit of ML over logistic regression for clinical prediction (Christodoulou et al., 2019,
*J Clin Epidemiol* 110:12–22).

## 4a — Evidence-weighted formula (baseline)

**Model.** R = Σ wᵢ zᵢ ÷ Σ wᵢ over the features the child has (register E3, D1b). Weights from the meta-analysis
grading (feature_evidence.md, D3): F1 0.90, F3 0.87, F5 0.77, F2 0.75, F7 0.69, F6 0.35, F4 0.19, F8 0.12.
**Nothing is fitted to labels.** Dividing by the available weights keeps a Grade 1 child (no F1/F2, not taught)
on the same scale as older children.

**Risk bands.** R is compared with same-grade train children: ≥ 84th percentile (the 16% worst) = moderate,
≥ 95th (the 5% worst) = high (register F1–F3). Cut-offs are stored in `results/formula_cutoffs.json` with
`validated: false`. Children with `sufficient_data_area = 0` get **insufficient** (leader's screening boundary).

**Results** (`results/model_formula_metrics.json`):

| Data | AUC (95% CI) | Sensitivity / specificity, moderate+ | Sensitivity / specificity, high |
|---|---|---|---|
| Synthetic train (2,066 children, 104 DD) | 0.75 (0.70–0.80) | 0.47 / 0.86 | 0.25 / 0.96 |
| Synthetic validation (445, 23 DD) | 0.69 (0.58–0.79) | 0.30 / 0.86 | 0.13 / 0.97 |
| **Real Decarli calibration (50, 25 DD)** | **0.81 (0.69–0.93)** | not applied (different norms) | — |

AUC by grade (validation): Grade 1 0.69, Grade 2 0.68, Grade 3 0.69. The formula works equally across grades.

**Interpretation.**
1. With **no training**, the published evidence separates real dyscalculic from typical children well
   (AUC 0.81). These children had many lab trials per task, so their scores are reliable.
2. The simulated robot sessions give a lower AUC (0.69–0.75) because each child has only about 4 questions per
   task type (G11). The evidence is sound; the short assessment is the bottleneck.
3. For screening, sensitivity at the moderate band (0.30–0.47) is **too low**: more than half of DD children
   would be missed, and missing a child is the worse error (F4). Later models, a lower cut-off, or a longer
   assessment must improve this.
4. The train/validation gap is within sampling error (overlapping CIs; only 23 DD children in validation).

## 4b — Support Vector Machine

**Model.** Same features (F1–F8), split and bands as 4a, so the comparison is fair. Missing features (by design,
e.g. no F1/F2 in Grade 1) are set to 0 = "as expected for grade and age", with missing indicators.
`class_weight = balanced` (5% DD). RBF kernel; C ∈ {0.01, 0.1, 1, 10} and gamma ∈ {scale, 0.01, 0.1} tuned by
stratified 5-fold cross-validation on the train split only, scored by AUC. Best: C = 0.1, gamma = 0.01
(cross-validated AUC 0.75).

**Results** (`results/model_svm_metrics.json`):

| Data | AUC (95% CI) | Sens. / spec., moderate+ | Sens. / spec., high |
|---|---|---|---|
| Synthetic train | 0.77 (0.72–0.82) | 0.54 / 0.86 | 0.28 / 0.96 |
| Synthetic validation | 0.70 (0.60–0.79) | 0.30 / 0.87 | 0.09 / 0.98 |
| **Real Decarli calibration (50)** | **0.79 (0.66–0.91)** | — | — |

Validation AUC by grade: Grade 1 0.76, Grade 2 0.66, Grade 3 0.66.

**Interpretation.** The SVM performs **the same as the evidence formula** (validation 0.70 vs 0.69, real 0.79 vs
0.81; confidence intervals overlap almost completely). The tuned small gamma makes the SVM nearly linear,
close to a weighted sum like the formula. With these features, learning from labels adds nothing beyond the
published evidence. Sensitivity at the moderate band stays low (0.30).

## 4-S — Sensitivity analysis: how to catch more DD children

At the moderate band (worst 16%) both models catch only 30% of DD children in validation. Three remedies were
tested (`results/sensitivity_analysis.json`). The test split and the 14 real Decarli test children were not used.

**A. Screening cut-off.** A screener should reach sensitivity and specificity of at least 70–80% (Glascoe, 2005;
register F5). The cut-off is the smallest flag share whose **train** sensitivity reaches 0.70 (SVM: out-of-fold
train scores, so the choice is not optimistic).

| Flag the worst … of each grade | Formula: sens. train / validation | Formula: spec. validation | SVM: sens. validation | SVM: spec. validation |
|---|---|---|---|---|
| 5% (high band) | 0.25 / 0.13 | 0.97 | 0.09 | 0.97 |
| 16% (moderate band) | 0.47 / 0.30 | 0.86 | 0.30 | 0.86 |
| 25% | 0.58 / 0.35 | 0.78 | 0.39 | 0.77 |
| **35% (formula screening cut-off)** | **0.73 / 0.57** | **0.71** | 0.57 | 0.70 |
| **45% (SVM screening cut-off)** | 0.79 / 0.65 | 0.62 | **0.70** | **0.60** |

A lower cut-off raises sensitivity, but the price is high: about 1 in 3 children flagged, specificity about 0.70.
With the current assessment length, no cut-off meets both 70% targets.

**B. Two-stage rule (DSM-5 persistence).** Each session scored separately (1,779 train + validation children with
enough data in both sessions, 88 DD), cut-off = worst 35%:

| Rule | Sensitivity | Specificity | Flagged |
|---|---|---|---|
| Session 1 only | 0.67 | 0.68 | 34% |
| Both sessions pooled into one score | 0.67 | 0.68 | 34% |
| Refer only if flagged in **both** sessions | 0.41 | 0.86 | 16% |
| Flag if flagged in **either** session | 0.84 | 0.48 | 53% |

Requiring persistence across two short sessions loses many DD children, because a single session (about 2
questions per task type) is too unreliable. With short sessions, "flag if either" is the high-sensitivity first
stage, and a later, longer check confirms.

**C. More sessions.** The same 3,000 children re-simulated with 1, 2, 4 and 6 sessions (each session still
16–24 questions, C2's limit). Formula score; train + validation children (about 125 DD).

| Sessions (max questions) | AUC | Sens. / spec. at worst 16% | Sens. / spec. at worst 35% |
|---|---|---|---|
| 1 (24) | 0.71 | 0.37 / 0.85 | 0.65 / 0.67 |
| **2 (48), current plan** | 0.74 | 0.44 / 0.86 | 0.70 / 0.68 |
| 4 (96) | **0.80** | 0.56 / 0.86 | **0.80** / 0.68 |
| 6 (144) | 0.79 | 0.54 / 0.87 | 0.75 / 0.68 |

Going from 2 to 4 sessions gives the largest gain: AUC 0.74 → 0.80, and sensitivity at the screening cut-off
0.70 → 0.80. Beyond 4 sessions there is no further gain (4 vs 6 is sampling noise). AUC levels off near 0.80,
the same as the real Decarli children, who had many lab trials (0.81, 4a). This ceiling is set by the size of
the published DD differences, not by the model or the number of questions.

**Conclusions.**
1. **The assessment length is the main lever**, not the model. Recommendation to the team: about **4 sessions**
   per child before a risk band is reported (I6; register F7).
2. For screening, report risk at a **sensitivity-oriented cut-off** (worst ~35%, F5), labelled "needs a closer
   look", not "dyscalculia". The 16% / 5% bands stay as a severity description.
3. Do **not** require a flag in two short sessions (sensitivity 0.41). Use the pooled score of all sessions.
4. Every later model (4c–4e) is reported at both the 16% band and the screening cut-off.
5. All cut-offs are synthetic and **unvalidated**; real cut-offs need pilot data (H1).

## 4c — Random Forest

**Model.** Same features (F1–F8), split, imputation and bands as 4a/4b. 500 trees,
`class_weight = balanced_subsample`. Tuned by stratified 5-fold cross-validation on the train split (AUC):
max_depth ∈ {3, 5, 8}, min_samples_leaf ∈ {10, 30, 100}, max_features ∈ {sqrt, 0.5}. Best: depth 8, leaf 30,
sqrt (cross-validated AUC 0.74). A forest memorises its training children, so **train results and band
cut-offs use out-of-fold scores**. The screening cut-off follows 4-S (smallest flag share with train
sensitivity ≥ 0.70, register F5).

**Results** (`results/model_random_forest_metrics.json`):

| Data | AUC (95% CI) | Sens. / spec., worst 16% | Sens. / spec., worst 5% |
|---|---|---|---|
| Synthetic train (out-of-fold) | 0.74 (0.68–0.79) | 0.53 / 0.86 | 0.23 / 0.96 |
| Synthetic validation | 0.66 (0.55–0.77) | 0.30 / 0.88 | 0.04 / 0.98 |
| **Real Decarli calibration (50)** | **0.81 (0.67–0.93)** | — | — |

Screening cut-off: worst 40% → validation sensitivity 0.61, specificity 0.68.
Validation AUC by grade: Grade 1 0.74, Grade 2 0.61, Grade 3 0.64.

**Which features the forest uses** (permutation importance = drop in validation AUC when a feature is shuffled,
30 repeats): F3 arithmetic accuracy 0.051, F6 counting accuracy 0.041, F4 arithmetic RT 0.038,
F2 symbolic comparison RT 0.018, F1 symbolic comparison accuracy 0.016; F7, F8, F5 ≈ 0. With only 23 DD
children in validation the SDs are about 0.01–0.025, so only the top three are clearly above zero.

**Interpretation.**
1. The forest is **not better** than the formula or the SVM (validation 0.66 vs 0.69–0.70, real 0.81 vs
   0.79–0.81; all CIs overlap). It can learn interactions ("slow AND inaccurate"), but the data gives no
   evidence that such interactions help.
2. It loses slightly more from train to validation, the usual sign of a flexible model fitting noise in 104 DD
   training children.
3. Its feature ranking partly differs from the evidence weights (e.g. counting RT, F5, weight 0.77, is not used).
   With 23 validation DD children this ranking is unstable, so the evidence weights stay the reference;
   XGBoost + SHAP (4d) gives a second view.

## 4d — XGBoost + SHAP

**Model.** Same features (F1–F8), split, imputation and bands as 4a–4c. Class imbalance:
`scale_pos_weight` = typical ÷ DD in train (19.0). subsample = colsample = 0.8. Tuned by stratified 5-fold
cross-validation on the train split (AUC). The first grid (depth 2–4, learning rate 0.03/0.1, 100/300 trees)
chose its simplest corner, so it was extended toward simpler models (depth 1–4, rate 0.01/0.03/0.1,
50/100/300 trees, min_child_weight 1/5); the choice is still made by train cross-validation only.
Best: **depth 1**, rate 0.1, 100 trees, min_child_weight 1 (cross-validated AUC 0.75). Depth 1 means every tree
uses one feature: the model is a sum of per-feature effects, with **no interactions**. Train results and cut-offs
use out-of-fold scores, as in 4c.

**Results** (`results/model_xgboost_metrics.json`):

| Data | AUC (95% CI) | Sens. / spec., worst 16% | Sens. / spec., worst 5% |
|---|---|---|---|
| Synthetic train (out-of-fold) | 0.74 (0.69–0.80) | 0.49 / 0.86 | 0.25 / 0.96 |
| Synthetic validation | 0.65 (0.52–0.77) | 0.30 / 0.89 | 0.13 / 0.97 |
| **Real Decarli calibration (50)** | **0.83 (0.70–0.94)** | — | — |

Screening cut-off: worst 40% → validation sensitivity 0.65, specificity 0.63.
Validation AUC by grade: Grade 1 0.71, Grade 2 0.63, Grade 3 0.62.

**SHAP** (`src/explain_shap.py`, `results/shap_xgboost_summary.json`). TreeExplainer splits each child's score
(log-odds) into one contribution per feature; contributions add up to the score exactly (max error 10⁻⁶).

| Skill (features) | Mean \|SHAP\|, validation |
|---|---|
| Arithmetic (F3, F4) | 0.46 |
| Counting (F5, F6) | 0.41 |
| Number comparison, symbolic (F1, F2) | 0.34 |
| Quantity comparison, non-symbolic (F7, F8) | 0.17 |

- **Agreement with the evidence.** Ranking of F1–F8 by mean |SHAP| vs the meta-analysis weights: Spearman
  ρ = 0.67 (p = 0.07, 8 features). XGBoost learned from labels largely the same order the published evidence
  gives; the largest difference is F7 (non-symbolic accuracy), last for XGBoost, fifth by evidence.
- **Direction.** For every feature a worse z-score gives a higher contribution (Spearman 0.77–0.99; F4 0.50).
- **Skills not assessed.** A Grade 1 child has no symbolic comparison (not taught, C3b), yet SHAP gives F1/F2
  small contributions, because the imputed "as expected" value differs from the average child. A child's
  explanation therefore **leaves out skills that were not assessed** and names them as "not assessed".
- **Worked example** (synthetic child SYN-03092, Grade 1, DD, high band, probability 0.78): arithmetic +0.67,
  counting +0.35, quantity comparison +0.15; number comparison not assessed. A teacher would read: "main
  difficulty in adding and subtracting, some difficulty in counting".

**Interpretation.**
1. XGBoost is **not better** than the other models (validation 0.65; real 0.83, the highest, but all CIs overlap).
   Cross-validation chose depth 1, so even the most flexible model found **no useful interactions**.
2. SHAP gives what the product needs: per-child contributions grouped by skill, which the child report (after
   4f) will turn into plain language. The same report can use the formula's exact contributions (w·z / Σw) if
   the formula is chosen.

## 4e — Logistic regression (reference)

**Model.** risk (log-odds) = b₀ + Σ bᵢ zᵢ. Same features (F1–F8), split, imputation and bands as 4a–4d;
`class_weight = balanced`; L2 penalty with C tuned by stratified 5-fold cross-validation on the train split.
The first grid (C 0.001–10) chose its edge, so C = 100 was added; cross-validated AUC was the same at 10 and
100 (0.751), i.e. a plateau where the penalty hardly matters. Best C = 100. Train results and cut-offs use
out-of-fold scores.

**Results** (`results/model_logistic_metrics.json`):

| Data | AUC (95% CI) | Sens. / spec., worst 16% | Sens. / spec., worst 5% |
|---|---|---|---|
| Synthetic train (out-of-fold) | 0.74 (0.69–0.79) | 0.52 / 0.86 | 0.25 / 0.96 |
| Synthetic validation | 0.69 (0.58–0.80) | 0.30 / 0.89 | 0.09 / 0.97 |
| **Real Decarli calibration (50)** | **0.78 (0.65–0.90)** | — | — |

Screening cut-off: worst 40% → validation sensitivity 0.65, specificity 0.62.
Validation AUC by grade: Grade 1 0.75, Grade 2 0.69, Grade 3 0.65.

**Learned weights vs evidence weights** (share = weight ÷ sum of weights):

| Feature | Learned weight | Learned share | Evidence share |
|---|---|---|---|
| F1 symbolic comparison accuracy | 0.86 | 27% | 19% |
| F3 arithmetic accuracy | 0.53 | 17% | 19% |
| F6 counting accuracy | 0.43 | 13% | 8% |
| F5 counting RT | 0.39 | 12% | 17% |
| F2 symbolic comparison RT | 0.35 | 11% | 16% |
| F8 non-symbolic comparison RT | 0.29 | 9% | 3% |
| F4 arithmetic RT | 0.18 | 6% | 4% |
| F7 non-symbolic comparison accuracy | 0.17 | 5% | 15% |

All learned weights are positive (worse → more risk). Rank agreement with the evidence: Spearman ρ = 0.71
(p = 0.047). The largest differences: F7 is weighted less and F6/F8 more than the evidence suggests; with
104 DD training children these differences are within sampling noise.

Missing indicators get large, unstable weights (e.g. missing counting accuracy −6.5) because very few children
miss those tasks; F1 and F2 indicators (both missing in Grade 1) cancel out (+1.11 − 1.01). They carry no
clinical meaning and are not used for explanations.

**Interpretation.**
1. Logistic regression performs **the same as every ML model** (validation 0.69, real 0.78; CIs overlap), as
   the systematic review predicts (Christodoulou et al., 2019).
2. Weights learned from synthetic labels largely **reproduce the published evidence weights**: an
   independent check that the formula's weights are reasonable.

## Comparison so far

| Model | Validation AUC | Real Decarli AUC | Validation sens. / spec., worst 16% | Screening cut-off: validation sens. / spec. |
|---|---|---|---|---|
| 4a Formula (no training) | 0.69 (0.58–0.79) | 0.81 (0.69–0.93) | 0.30 / 0.86 | worst 35%: 0.57 / 0.71 |
| 4b SVM | 0.70 (0.60–0.79) | 0.79 (0.66–0.91) | 0.30 / 0.87 | worst 45%: 0.70 / 0.60 |
| 4c Random Forest | 0.66 (0.55–0.77) | 0.81 (0.67–0.93) | 0.30 / 0.88 | worst 40%: 0.61 / 0.68 |
| 4d XGBoost | 0.65 (0.52–0.77) | 0.83 (0.70–0.94) | 0.30 / 0.89 | worst 40%: 0.65 / 0.63 |
| 4e Logistic regression | 0.69 (0.58–0.80) | 0.78 (0.65–0.90) | 0.30 / 0.89 | worst 40%: 0.65 / 0.62 |

Screening cut-offs are chosen on train (4-S, F5). Validation has only 23 DD children, so differences of a few
points are within noise.

## 4f — Model selection

**Rule** (written before running): among the models whose AUC is **not reliably worse** than the best model,
choose the **simplest and most explainable** one. Order: formula > logistic regression > XGBoost > Random
Forest > SVM (table below). "Not reliably worse" = the 95% paired-bootstrap CI of (AUC best − AUC model)
includes 0. The bootstrap is paired (the same children resampled for every model) and stratified (DD and
typical resampled separately), 2,000 resamples. Primary data: validation; secondary: train (out-of-fold for
trained models) + validation. Test data not used.

| Model | Needs training | Weights come from | Per-child explanation |
|---|---|---|---|
| Formula | No | Published evidence (meta-analysis) | Exact (w·z per feature) |
| Logistic regression | Yes | Learned | Exact (b·z per feature) |
| XGBoost | Yes | Learned, 100 trees | SHAP (exact for trees) |
| Random Forest | Yes | Learned, 500 trees | Approximate only |
| SVM | Yes | Learned, RBF kernel | None |

**Results** (`results/model_selection.json`):

| Model | Validation AUC (23 DD) | Difference from best (95% CI) | Train OOF + validation AUC (127 DD) | Difference from best (95% CI) | Real Decarli AUC (50) |
|---|---|---|---|---|---|
| **Formula** | 0.69 | 0.01 (−0.04 to 0.07) | 0.74 | 0.02 (−0.003 to 0.04) | **0.81** |
| Logistic regression | 0.69 | 0.003 (−0.04 to 0.06) | 0.73 | 0.03 (0.006 to 0.05) | 0.78 |
| XGBoost | 0.65 | 0.05 (−0.02 to 0.13) | 0.73 | 0.03 (0.005 to 0.06) | 0.83 |
| Random Forest | 0.66 | 0.03 (−0.02 to 0.09) | 0.72 | 0.04 (0.01 to 0.06) | 0.81 |
| SVM (best on synthetic) | 0.70 | — | 0.76 | — | 0.79 |

**Decision: the evidence-weighted formula** (`config/selected_model.json`, `validated: false`).
1. On validation, no model is reliably better than any other.
2. On the larger secondary comparison SVM is best by 0.02 AUC, but the formula is still not reliably worse
   (CI includes 0); logistic regression, XGBoost and Random Forest are slightly but reliably worse there.
3. On the real Decarli children the formula (0.81) is ahead of SVM (0.79); all CIs overlap.
4. The formula needs no training on synthetic labels, every weight is cited, and its explanation of a child
   is exact. The other four models are kept as evidence that the choice costs no accuracy.

Settings carried forward: bands at the 84th / 95th same-grade percentile (`results/formula_cutoffs.json`),
screening flag at the worst 35% (4-S, F5), "insufficient" when `sufficient_data_area = 0`.
Step 6 evaluates this one model once on the untouched test data.

## Step 6 — Final evaluation

The selected formula was evaluated once on the untouched test data: see
[tripod_ai_report.md](tripod_ai_report.md) and `results/final_evaluation.json`.
