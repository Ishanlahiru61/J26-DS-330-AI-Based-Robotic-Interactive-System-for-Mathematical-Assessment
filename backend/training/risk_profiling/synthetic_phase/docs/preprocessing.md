# Step 2 — Preprocessing

`src/preprocess.py` turns `cube_assessment_trials` into one clean record per child. Tests:
`tests/test_preprocess.py`. Run report: `results/preprocessing_report.json`.

## Rules

| # | Rule | Evidence |
|---|---|---|
| P1 | Validate the trials table against schema v1; stop on any error | dataset_schema.md |
| P2 | Remove trials with `valid_trial = 0` (cube detection failed) | B1 |
| P3 | Remove trials whose task type is not taught at the child's grade (`in_curriculum = 0`) | DSM-5 Criterion B; C3b, C3c |
| P4 | Remove trials with no task type (e.g. comparison without `representation`) | C7 |
| P5 | Response time: only `rt_valid` trials (below 15 s, no robot demo), only **correct** answers, transformed with **log(1 + RT)**; summarised by the median | I3; RT skew in [D] (1.06–1.71); C2 uses log(1 + RT) ([P-C2] p. 24) |
| P6 | Behaviour `uncertain` is missing, not "no finger counting" | I2 |
| P7 | Pool **all sessions** of a child, per task type and per C2 area | A4; G11 |
| P8 | Accuracy smoothed as (correct + 1) / (questions + 2) (Laplace correction), raw value kept | Small counts (median 4 questions per task type) give extreme raw rates by chance |
| P9 | Area records use risk task types only, so area scores keep their evidence base | G11 remedy; task_taxonomy.md |
| P10 | Data sufficiency flags: `sufficient_data` (every risk task type for the grade) and `sufficient_data_area` (every C2 area for the grade) | Leader's `docs/screening-boundary.md`: missing data must give an insufficient result, not a confident label |
| P11 | Child-level split train / validation / test = 70 / 15 / 15, stratified by grade (and label when known), each child in one part only | Same split as C2 ([P-C2] p. 23); prevents leakage |

Labels are used only to balance the split (P11). Outlier capping is done on z-scores in Step 3.

## Outputs (in `artifacts/`, rebuilt by the script)

| File | One row = | Key columns |
|---|---|---|
| `preprocessed_children_v1.csv` | child | grade, age_months, n_sessions, n_questions, missing_risk_tasks, sufficient_data, missing_areas, sufficient_data_area, split |
| `preprocessed_child_task_v1.csv` | child × task type | n_questions, n_correct, accuracy_raw, accuracy_smoothed, n_rt, median_log_rt, sd_log_rt, mean_relative_error, mean_attempts, finger_counting_rate, risk_use, split |
| `preprocessed_child_area_v1.csv` | child × C2 area | same as above, risk task types only |

## Result on synthetic trials-v1 (3,000 children)

| Item | Value |
|---|---|
| Trials in / kept | 120,187 / 116,595 (3,592 invalid removed) |
| Response times used | 86,792 correct, usable answers; 29,803 excluded (wrong answers with robot demo, or ≥ 15 s) |
| Median questions per child | **4 per task type**, **6 per area** |
| Children with every risk task type | 2,336 / 3,000 (Grade 1: 99.8%, Grade 2: 78.0%, **Grade 3: 55.8%**) |
| Children with every area | 2,947 / 3,000 (Grade 3: 95.9%) |
| Split (children / DD) | train 2,100 / 105; validation 450 / 23; test 450 / 22 |

**Consequence for Step 3:** with C2's 16–24 questions over up to 8 activities, many Grade 3 children never meet
one of the risk task types. Area-level features are therefore the main input; task-level features are used
where available.
