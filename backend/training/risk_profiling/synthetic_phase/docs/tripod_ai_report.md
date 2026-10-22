# TRIPOD+AI report — Learning-Risk Profiling (synthetic phase)

Development and internal evaluation of a dyscalculia-risk **screening** model for children aged 6–8 (Grades 1–3)
in a Sri Lankan robotic cube mathematics assessment. Structured by the TRIPOD+AI statement (Collins et al., 2024,
*BMJ* 385:e078378); check item wording against the published checklist before submission. Every number below is
in `results/` and every decision in [evidence_register.md](evidence_register.md) (row IDs in brackets).

## Title and abstract

**Title.** Development and internal evaluation of an evidence-weighted screening score for developmental
dyscalculia risk in Grade 1–3 children, using simulated robot-assessment data and real external children.

**Abstract.** *Objective:* flag children aged 6–8 who may need a closer look for dyscalculia, from data already
recorded by the robot assessment (Components 1–3). *Design:* prediction-model development on 3,000 simulated
children (5% DD) generated from published effect sizes, with checks on 64 real children (Decarli et al., 2020).
*Predictors:* eight age- and grade-normed task measures (F1–F8). *Models:* an evidence-weighted formula (no
training), SVM, Random Forest, XGBoost and logistic regression; the formula was selected by a pre-specified rule.
*Results (held-out test):* synthetic AUC 0.66 (95% CI 0.55–0.77); screening flag (worst 35%) sensitivity 0.59
(0.39–0.77), specificity 0.69 (0.65–0.74); real test children AUC 0.73 (0.45–0.98, n = 14). *Conclusion:* the
score works as a screening aid in principle but is **not validated** for use; the short assessment limits
sensitivity, and real Sri Lankan pilot data are required.

## Introduction

- **Background.** Developmental dyscalculia (DD) affects about 3–7% of children ([HSK-2019], G3). DSM-5 requires
  persistent difficulties below chronological age (A1–A2). Published tests are specialist-administered.
- **Objective.** Screening support, not diagnosis: identify children who need a closer look, explain why, and
  suggest practice (A2, F4). Intended users: teachers; parents receive a simplified text.

## Methods

### Data sources
| Data | Children | Use |
|---|---|---|
| Synthetic children (`synthetic_children_v1`) | 3,000 used (150 DD, 5%) | Development, validation, test |
| Simulated robot sessions | 2 sessions × 16–24 questions per child, 120,187 answers | Operational data format (G10) |
| Real children, Decarli et al. (2020), OSF gjpb3 | 50 calibration (25 DD), 14 test (7 DD); Italian, aged 7–14 | Generator calibration, model checks, final test |

Synthetic group differences come from meta-analytic effect sizes ([M], [S]) and correlations of the real
calibration children (G1–G2); absolute levels are placeholders (G9, `config/simulation_parameters.json`).

### Participants and setting
Target: Sri Lankan Grade 1–3 children (ages 6–8) using the robot cube assessment. No real target-population data
exist yet; this report covers the synthetic phase only (H3).

### Outcome
DD label. Synthetic: set by the generator at 5% prevalence. Real: the Decarli DD criterion, ≤ 5th percentile on
standardised mathematics tests ([D], F2). The outcome was never an input to the features.

### Predictors
F1–F8 ([feature_evidence.md](feature_evidence.md)): symbolic comparison accuracy/speed, arithmetic accuracy/speed,
counting accuracy/speed, non-symbolic comparison accuracy/speed. Each is a z-score against same-grade children of
the same age (regression norming on train children only, E2–E3), positive = worse, capped at ±3 (E5). All come
from Component 1/2 outputs ([team_interface.md](team_interface.md)).

### Data preparation and missing data
Invalid, out-of-curriculum and unassigned answers removed (P1–P11, [preprocessing.md](preprocessing.md)).
Missingness is mostly **by design** (symbolic comparison not taught in Grade 1, C3b): the formula renormalises
its weights over available features (D1b); ML models imputed 0 ("as expected") with missing indicators.
Children without data in every expected C2 area receive "insufficient data", never a risk result.

### Sample size
3,000 children with 150 events exceeds the Riley et al. (2020) minimum of 1,425 for 8 predictors at 5%
prevalence (G7). Real test: 14 children, far too few for precise estimates (stated with every result).

### Analysis
- Split 70/15/15 by child, stratified by grade and label (P-split); the test split and the 14 real test children
  were not used until Step 6.
- Models ([modelling.md](modelling.md)): evidence formula R = Σwᵢzᵢ ÷ Σwᵢ (weights from meta-analysis grading, D3);
  SVM, Random Forest, XGBoost and L2 logistic regression tuned by stratified 5-fold cross-validation on train only;
  class imbalance by class weights; out-of-fold train scores for trained models.
- Risk bands: worst 16% / 5% of same-grade train children (F1–F3). Screening flag: smallest share with train
  sensitivity ≥ 0.70 (worst 35%, F5; target from Glascoe, 2005).
- Model selection rule written before running (4f, E3i): simplest model not reliably worse than the best
  (paired bootstrap of AUC differences).
- Performance: AUC with stratified bootstrap 95% CI (2,000 resamples); sensitivity and specificity with Wilson
  95% CI; flag-rate calibration against nominal percentiles; by grade.

### Class imbalance and fairness
Prevalence 5% handled by class weights / `scale_pos_weight`; the formula is unaffected (no training). Fairness:
performance reported by grade; sex, language and socio-economic status are not available in the synthetic data
and must be checked in the pilot.

### Model output and explanation
Risk band (low / worth monitoring / moderate / high / insufficient), skill-level explanation with exact formula
contributions, rule-based difficulty profile, and Component 2 practice suggestions; teacher and parent texts
([child_report.md](child_report.md), [clustering.md](clustering.md), R1–R5, S1–S4).

### Development vs evaluation
Development, tuning and selection used train and validation children only. The final evaluation ran once on
2026-10-21 with the frozen configuration (`config/selected_model.json`, code commit 859d763).

## Open science
Code, tests (59), configuration, synthetic samples and aggregate results are in this repository
(`backend/training/risk_profiling/synthetic_phase/`). Real child-level data are public at OSF gjpb3 but kept out
of Git (`data/private/`). No registration.

## Patient and public involvement
None yet. Report wording should be reviewed by 2–3 teachers before the pilot (R5).

## Results

### Participants
Synthetic test: 450 children (22 DD); 14 (3.1%) had insufficient data, leaving 436 (22 DD). Train: 2,066 usable
(104 DD); validation: 445 (23 DD). Real test: 14 children (7 DD).

### Model development
All five models performed alike on validation (AUC 0.65–0.70; real calibration 0.78–0.83; [modelling.md](modelling.md)
4f). The evidence formula was selected: no model was reliably better, it needs no training on synthetic labels, and
its explanations are exact.

### Final performance (`results/final_evaluation.json`)

| Data | Measure | Result (95% CI) |
|---|---|---|
| Synthetic test (436, 22 DD) | AUC | **0.66** (0.55–0.77) |
| | Screening flag (worst 35%): sensitivity / specificity | **0.59** (0.39–0.77) / **0.69** (0.65–0.74) |
| | Moderate or high (worst 16%): sensitivity / specificity | 0.27 (0.13–0.48) / 0.85 (0.82–0.88) |
| | High (worst 5%): sensitivity / specificity | 0.14 (0.05–0.33) / 0.95 (0.92–0.97) |
| Real Decarli test (14, 7 DD) | AUC | **0.73** (0.45–0.98) |

**Calibration of flag rates** (share of new children flagged vs intended): high 5.7% vs 5%; moderate or high
15.4% vs 16%; screening flag 32.1% vs 35%. The train-based cut-offs transfer well to new children.

**By grade** (synthetic test):

| Grade | Children (DD) | AUC (95% CI) | Screening sensitivity / specificity |
|---|---|---|---|
| 1 | 150 (7) | 0.87 (0.72–0.99) | 0.71 / 0.73 |
| 2 | 146 (7) | 0.69 (0.57–0.81) | 0.71 / 0.69 |
| 3 | 140 (8) | **0.43** (0.25–0.59) | 0.38 / 0.66 |

**Child report safety:** 900 reports (teacher + parent) for all test children; 0 rule violations (no diagnostic
words, no unassessed skill shown to parents, no result for insufficient data).

## Discussion

### Interpretation
- The test AUC (0.66) is close to validation (0.69): no sign of overfitting, as expected for a model that was not
  trained on labels.
- On real children the score separates DD from typical children (0.73; earlier 0.81 on 50 calibration children),
  but 14 children allow no precise estimate (CI 0.45–0.98).
- At the screening flag the score catches about 6 in 10 DD children while flagging about 1 in 3 children; this is
  below the 70–80% screening standard (Glascoe, 2005). The sensitivity analysis (4-S) shows the main remedy:
  about 4 sessions per child instead of 2 (F7).
- **Grade 3 performed poorly** (AUC 0.43, upper CI 0.59) while validation Grade 3 was 0.65. With 8 DD children
  this may be chance, but it must not be ignored: possible reasons are that Grade 3 arithmetic pools ×/÷ with
  +/− (F3/F4), and that the multiplicative area is not used. The model was **not changed** after this result;
  Grade 3 is a priority for the pilot analysis.

### Limitations
1. Development data are simulated; absolute performance levels are placeholders (G9) and the generator has no
   subtypes (Step 5).
2. Real checks use Italian children aged 7–14 without arithmetic measures; not the target population.
3. Small real test set (n = 14) and few synthetic test DD children (22).
4. Short assessment (2 sessions) limits reliability (G11) and sensitivity.
5. No data on sex, language or socio-economic status for fairness checks.
6. Cut-offs, mastery thresholds and skill levels are unvalidated (`validated: false`).

### Usability and next steps
- **Intended use:** screening support for teachers, never a diagnosis (H3). Not ready for use with children.
- **Next:** ethics approval; pilot with Sri Lankan Grade 1–3 children with professional assessment as reference;
  re-estimate norms and cut-offs; re-run clustering; check Grade 3 and fairness; teacher review of the report;
  agree session count with Component 2 (F7, I9).

## References
See [evidence_register.md](evidence_register.md), Sources: DSM-5; TRIPOD+AI (Collins et al., 2024);
Haberstroh & Schulte-Körne (2019, 2022); Schwenk et al. (2017); Decarli et al. (2020); Riley et al. (2020);
Christodoulou et al. (2019); Glascoe (2005); Lundberg et al. (2020); NIE Teacher's Guides Grades 1–3.
