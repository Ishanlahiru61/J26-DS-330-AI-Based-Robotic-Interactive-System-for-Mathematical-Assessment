# Component 2 Machine-Learning Research Audit

## Audit conclusion

The ASSISTments research is internally traceable and independently validated through frozen Step 1–3 artifacts. It supports one defensible claim: **on the ASSISTments 2012–2013 School Data with Affect dataset, accumulated and compact chronological learner history improved first-attempt correctness probability prediction, and H2 gradient boosting outperformed the H2 logistic-regression baseline; historical response-time features did not add practically useful value.** It does not validate dyscalculia detection, causal activity effects, local Sri Lankan transfer, or direct ranking of the eight robot games.

The public-data model and local prototype are separate. `H2_GB_RT_OFF_v1` predicts first-attempt correctness for supported ASSISTments observations. The operational C2 application applies curriculum, hardware, coverage, prerequisite, reinforcement and Tier A/B rules. Because no validated ASSISTments-to-local candidate adapter or local outcome study exists, local `predicted_probability` is correctly `null`.

## 1. Authoritative evidence hierarchy

1. Frozen summaries, result tables, manifests and independent validation reports in `ASSISTments Dataset/01_ASSISTments_Dataset_Preparation`, `02_Chronological_Learner_History`, and `03_Experimental_Modeling` are the result sources of truth.
2. `run_step1_pipeline.py`, `run_step2_pipeline.py`, and `run_step3_pipeline.py` are the methodological sources of truth.
3. Saved row-level predictions substantiate the ROC and calibration curves and permit explicitly post-freeze threshold diagnostics.
4. The operational code and its architecture/limitations documents establish the deployment boundary.
5. Existing Step 2 and Step 3 notebooks are explanatory rather than complete executable pipelines: the original Step 2 notebook has zero code cells; Step 3 has five unexecuted analysis cells. The new notebooks in this package repair the demonstration/reproducibility layer without replacing the authoritative scripts.

## 2. Dataset audit

Dataset edition: **ASSISTments 2012–2013 School Data with Affect**. The raw CSV hash is `1d06aee9e649c5ba9db49052fe9a86e68d0c3ccb4709a7487289d7721f5464db`.

| Dataset statistic | Verified value | Source file | Method used | Notes |
|---|---:|---|---|---|
| Raw interactions | 6,123,270 | Step 1 Full Profile / freeze summary | Chunked full-file audit | All rows retained in Step 1 |
| Original columns | 35 | Step 1 Full Profile | Header/schema audit | Broad cleaned output has 44; core has 36 |
| Learners | 46,674 | Step 1 Full Profile | Distinct `user_id` | 691 learners have one event; 3,498 have fewer than five |
| Problems | 179,999 | Step 1 Full Profile | Distinct `problem_id` | 228,368 repeated learner–problem rows beyond first are legitimate repetitions |
| ASSISTments | 138,555 | Step 1 Full Profile | Distinct `assistment_id` | Context/grouping identifier |
| Skill names / IDs | 198 / 265 | Step 1 Full Profile | Distinct non-missing values | ASSISTments labels are not NIE levels or local games |
| Missing `skill` | 3,493,190 (57.0478%) | Step 1 Data Dictionary | Null count | 17,656 learners have no skill-tagged interaction |
| Missing `skill_id` | 3,411,457 (55.7130%) | Step 1 Data Dictionary | Null count | Not imputed to a fabricated skill |
| Exact full-row duplicates | 0 | Step 1 Full Profile | `problem_log_id` uniqueness/full-row check | All 6,123,270 log IDs unique |
| Correctness 1 / 0 | 4,141,564 / 1,976,383 | Step 1 Full Profile | Full distribution | Binary supervised rows total 6,117,947 |
| Fractional correctness | 5,323 | Step 1/2 summaries | Values strictly between 0 and 1 | Retained in histories; excluded as binary targets |
| Attempt count | 0–29; 46,570 zeros | Step 1 Full Profile | Distribution audit | No missing or negative attempts |
| Response time | milliseconds; median 22,331; p95 182,432; p99 377,531.31 | Step 1 Full Profile | Quantiles | 13 negative, no missing/zero, maximum 345,055,238 ms |
| Timestamp range | 2012-09-01 to 2013-08-31 starts | Step 1 Full Profile | Parsed timestamps | Zero start/end parse failures |
| Timestamp anomalies | 727 negative durations; 1,696 learner/start tie groups | Step 1 Full Profile / Step 2 summary | QA flags and grouping | 3,443 rows occur in tied groups; all retained |
| Supervised exclusions | 5,323 rows | Step 2 target audit | `correct IN (0,1)` | Exclusion applies only to target use, not historical context |

### Raw-column treatment

- **Model context:** `skill_id`, `problem_type`, `tutor_mode`, `original`, `position`, and `template_id` enter H0. `template_id` is represented by log training-frequency, not target statistics.
- **History inputs:** `correct`, `attempt_count`, `ms_first_response`, `skill_id`, `problem_type`, `problem_id`, `assistment_id`, and timestamps create shifted or accumulated fields. Current correctness, attempts and response time are never used directly to predict that same event.
- **Target:** exact `correct == 0` or `correct == 1` becomes `target_first_attempt_correct`; fractional partial credit receives a null target.
- **Ordering/grouping:** `user_id`, parsed `start_time`, and `problem_log_id`. The last is only a deterministic physical-order tie-breaker.
- **Leakage-sensitive exclusions:** current `correct`, current attempt/hint/action/answer fields, `end_time`, derived duration and current affect confidence fields.
- **QA-only:** duplicate ID `problemlogid`, parse flags, overlap/duration checks and technical previous-row fields.

The full 35-column classification and all 42 derived feature definitions are in `Feature_Dictionary.xlsx`.

## 3. Chronological learner-history reconstruction

Step 2 sorts by `user_id ASC, start_time_parsed ASC, problem_log_id ASC`. History is learner-specific. The deterministic log ID order makes output stable but does not assert a causal order among rows with identical timestamps.

The central leakage control is DuckDB `RANGE BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW EXCLUDE GROUP`, ordered only by `start_time_parsed`. Consequently every row sharing the current timestamp is excluded from every predictive history window. The pipeline retains and flags tied records. Twelve automated leakage tests passed, including prior-count, target-balance, recent-window bounds, first-event RT exclusion, fractional-target policy, chronological order, and manual historical recount checks.

Missing skill IDs do not erase learner-wide history. Skill-conditioned features are null/not applicable when the current skill is unavailable. Historical RT summaries include only positive valid values. Negative RTs remain QA evidence but are ineligible for RT features. Negative or overlapping start/end intervals do not change start-time ordering; end-time/duration are excluded from predictors.

### Exact history contracts

**H0 — 14 fields (minimal/current plus immediate strict-prior history):** `skill_id`, `problem_type`, `tutor_mode`, `original`, `position`, `template_id`, `previous_correct_raw`, `previous_attempt_count`, `previous_ms_first_response`, `previous_skill_id`, `previous_problem_type`, `time_since_previous_seconds`, `has_strict_prior_interaction`, `previous_interaction_rt_valid`.

**H1 — H0 plus 15 fields, 29 cumulative:** `prior_interaction_count`, `prior_target_eligible_count`, `prior_full_correct_count`, `prior_incorrect_count`, `prior_correct_rate`, `prior_valid_rt_count`, `prior_mean_ms_first_response`, `prior_mean_attempt_count`, `prior_skill_exposure_count`, `prior_skill_target_eligible_count`, `prior_skill_full_correct_count`, `prior_skill_correct_rate`, `prior_same_problem_exposure_count`, `prior_same_assistment_exposure_count`, `prior_same_problem_type_exposure_count`.

**H2 — H1 plus 13 fields, 42 cumulative:** `lag2_target_first_attempt_correct`, `lag3_target_first_attempt_correct`, `lag4_target_first_attempt_correct`, `lag5_target_first_attempt_correct`, `recent3_target_eligible_count`, `recent3_full_correct_count`, `recent3_correct_rate`, `recent5_target_eligible_count`, `recent5_full_correct_count`, `recent5_correct_rate`, `interactions_since_last_full_correct`, `previous_same_skill_correct_raw`, `time_since_previous_skill_seconds`.

H0 already contains the immediately preceding correctness through `previous_correct_raw`; H2 adds lags 2–5 and recent-window summaries. H0/H1/H2 are nested raw contracts. RT OFF removes only `previous_ms_first_response`, `previous_interaction_rt_valid`, `prior_valid_rt_count`, and `prior_mean_ms_first_response`, producing fitted raw counts of 12/25/38. RT ON uses 14/29/42. One-hot expansion produces transformed widths of 309–339.

## 4. Split and preprocessing design

Seed `20260928` shuffles 46,674 distinct learners and assigns approximately 70/15/15 percent. Every learner appears exactly once.

| Split | Learners | All interactions | Supervised rows | Positive rate |
|---|---:|---:|---:|---:|
| Train | 32,672 | 4,262,767 | 4,259,096 | 0.676716 |
| Validation | 7,001 | 922,594 | 921,808 | 0.678663 |
| Test | 7,001 | 937,909 | 937,043 | 0.676349 |

Chronology and partitioning solve different problems: chronology prevents future/same-time information entering a row’s history; learner-disjoint partitions test generalization to unseen learners.

The common H2 RT-ON preprocessor is fitted only on supervised training learners, then feature columns are selected for each experiment. Numeric features use training-median imputation and training standardization. Small categoricals use training-fitted one-hot encoding, explicit missing values, unknown-category ignoring, and rare-category grouping below 1,000 occurrences. `template_id` uses training frequency, `log1p`, and standardization. RT ON applies `log1p`, median/MAD robust scaling, ±8 clipping and a previous-problem-type context with training-global fallback; minimum context support is 1,000. There is no empirical outcome-based difficulty proxy.

## 5. Executed experiment matrix

Exactly eight experiments completed: six regularized logistic models (H0/H1/H2 × RT OFF/ON) and two XGBoost models (H2 × RT OFF/ON). No H0/H1 gradient boosting run was claimed or found.

- Logistic Regression implementation: `SGDClassifier(loss="log_loss", penalty="l2", alpha=1e-5, max_iter=30, tol=1e-4, shuffle=True, average=True, class_weight=None, random_state=20260928)`.
- XGBoost 3.0.4: binary logistic objective, log-loss evaluation, maximum 350 trees, learning rate 0.08, depth 6, minimum child weight 20, subsample 0.80, column subsample 0.80, L2 lambda 2, L1 alpha 0, histogram method, max bin 255, early stopping 20, seed 20260928.
- No documented hyperparameter search was found. These are fixed, auditable settings. Early stopping uses validation data for the two GB runs.

LR is the scalable, coefficient-inspectable baseline for engineered history features. GB is a reasonable nonlinear benchmark for thresholds and interactions. H0→H1 isolates accumulated history; H1→H2 isolates compact ordered history; RT OFF→ON tests incremental timing value.

## 6. Results and metrics

Native test probabilities:

| Experiment | Log loss | Brier | ROC-AUC | ECE |
|---|---:|---:|---:|---:|
| H0_LR_RT_OFF | 0.591088 | 0.201599 | 0.664176 | 0.010295 |
| H0_LR_RT_ON | 0.590826 | 0.201424 | 0.663709 | 0.014256 |
| H1_LR_RT_OFF | 0.569530 | 0.192496 | 0.704559 | 0.007086 |
| H1_LR_RT_ON | 0.569838 | 0.192580 | 0.703898 | 0.008486 |
| H2_LR_RT_OFF | 0.562612 | 0.189680 | 0.713327 | 0.009693 |
| H2_LR_RT_ON | 0.562774 | 0.189731 | 0.713158 | 0.010529 |
| H2_GB_RT_OFF | 0.534998 | 0.179078 | 0.750158 | 0.010002 |
| H2_GB_RT_ON | 0.535337 | 0.179221 | 0.749632 | 0.009816 |

Log loss is the primary probability-quality measure and penalizes confident errors. Brier score is squared probability error. ROC-AUC measures ranking/discrimination, not probability calibration. ECE is a ten equal-frequency-bin calibration-gap summary. PR-AUC, calibration intercept/slope and learner-weighted diagnostics were also recorded. Accuracy can look good under a common threshold while probabilities are poorly calibrated, so it was not the selection criterion.

Accuracy, precision, recall and F1 were not in the original pipeline. From the stored selected calibrated test predictions, at the explicitly declared threshold 0.5, this audit derives accuracy **0.734105**, precision **0.751299**, recall **0.907165**, and F1 **0.821907**. These are post-freeze diagnostics, not retrospectively presented selection evidence.

H0→H1 LR RT-OFF reduced row-weighted test log loss by 0.021558; the equal-learner-weighted paired improvement was 0.010499 (95% bootstrap interval 0.008361–0.012767). H1→H2 reduced row-weighted log loss by 0.006919; equal-learner improvement was 0.010862 (0.009331–0.012334). These are paired learner-bootstrap uncertainty intervals, not a general claim of external statistical significance. Gains were not uniform in the earliest interactions.

RT was negligible/inconsistent: LR ON–OFF test log-loss differences were −0.000263 (H0), +0.000308 (H1), +0.000163 (H2); H2 GB RT ON was +0.000339 worse. The extra measurement dependence was not justified.

## 7. Calibration and final selection

Native validation ECE ≥0.01 triggered a validation-only Platt attempt. Retention required validation log-loss improvement ≥0.0005 and ECE improvement ≥0.002. Only both GB variants met the rule. Platt regression was fitted to the logit of native validation probabilities and validation labels; test labels were never fitted.

Selection used calibrated validation results where retained. Candidates within 0.001 of the best validation log loss were eligible; the least complex candidate then won, with log loss and Brier as tie-breakers. `H2_GB_RT_OFF` was selected with validation log loss 0.531555. Its final calibrated test metrics are:

| Metric | Verified value |
|---|---:|
| ROC-AUC | 0.7501579955 |
| Log loss | 0.5344418770 |
| Brier score | 0.1789547233 |
| ECE | 0.0038469301 |
| Calibration intercept / slope | −0.000965 / 0.994885 |

The public identifier `H2_GB_RT_OFF_v1` names the model. The deployed research demonstration additionally loads `H2_GB_RT_OFF_platt_v1.joblib`. A fresh frozen-model inference reproduced stored probability 0.6981496215 as 0.6981496423 (absolute difference 2.09×10⁻⁸), after verifying model/preprocessor/calibrator hashes.

## 8. Mathematics, dyscalculia context and transfer boundary

Learner history, correctness, skill context and trends are relevant because mathematical performance is heterogeneous and evolves across repeated tasks. The eight local games cover counting, comparison, ordering, place value, addition, subtraction, equal grouping and equal sharing—domains relevant to foundational mathematics and commonly discussed in mathematics-learning-difficulty research. This motivates structured observations; it does not make ASSISTments a diagnosed-dyscalculia dataset or C2 a diagnostic model.

No defensible ASSISTments skill-ID-to-eight-game mapping is implemented. The local `feature_adapter.py` explicitly reports partial support and raises for unsupported model input. Therefore the two-part contribution is:

1. a controlled public-data experiment showing that history-aware features improve ASSISTments outcome prediction; and
2. a functional curriculum-constrained, history-aware local recommendation policy.

The gap is local transfer validation: candidate alignment, ethically collected tangible-interaction outcomes, support/coverage evidence, recalibration and evaluation of recommendation behavior.

## 9. Limitations and unresolved evidence

- Observational first-attempt prediction is associational, not causal activity-effectiveness estimation.
- Learner-disjoint ASSISTments performance does not establish transportability to Sri Lankan children, Grades 1–5, languages, devices or robots.
- The proposal’s local pilot (approximately 25–30 learners), shadow/transfer evaluation and public-to-local calibration have not been completed.
- No formal hyperparameter search is documented.
- Current skill missingness is very high; skill-conditioned features are unavailable for many rows.
- Start-time chronology cannot guarantee real-time availability when prior activities overlap; operational documentation notes about 21.0318% overlap.
- Threshold metrics depend on 0.5 and prevalence and were not part of the original selection methodology.
- There is no validated diagnostic sensitivity, specificity, risk probability, grade-equivalent score or mastery threshold.

## 10. Verification performed for this finalization

- Read and visually inspected relevant approved-proposal pages 20–35.
- Verified Step 1–3 frozen counts, contracts, results and independent PASS reports.
- Read the exact history and modeling implementations.
- Regenerated six figures from saved tables and row-level predictions.
- Derived threshold diagnostics from exact labels/predictions with threshold explicitly fixed at 0.5.
- Executed all five new notebooks in their default lightweight mode; all passed after the documented runtime dependencies were available.
- Ran frozen model inference without retraining; PASS.
- Did not modify Steps 1–7, the finalized C2 code, or any GitHub repository.

