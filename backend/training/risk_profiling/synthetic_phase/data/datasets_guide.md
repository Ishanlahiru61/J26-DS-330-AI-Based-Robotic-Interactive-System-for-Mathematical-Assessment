# Risk-profiling datasets (Component 4: Learning-Risk Profiling)

Guide to **every dataset** of the risk-profiling component: what it is, one row = what, whether it is real or
made up, where it lives, and how to rebuild it. Owner: Ranasinghe I.V.L. Code and docs:
[`synthetic_phase/`](../synthetic_phase_guide.md).

> **No dataset here contains a real child.** Synthetic and demo rows are labelled in every row
> (`data_origin`, `is_synthetic`, `SYN-`/`DEMO-` IDs). Real research data stays outside Git (`data/raw/`, `data/private/`).

## 1. The datasets at a glance

| # | Dataset | One row = | Rows | Real? | Where | In Git |
|---|---|---|---|---|---|---|
| 1 | `synthetic_children_v1.csv` | simulated child: 8 evidence-based z-scores + DD label | 10,000 (500 DD) | synthetic | this folder | ✅ |
| 2 | `demo_cube_assessment_trials.csv` | one answered question (operational format) | 240 (6 children) | **demo**, placeholder numbers | this folder | ✅ |
| 3 | `demo_student_performance_summary.csv` | child × session × task type | 98 | demo | this folder | ✅ |
| 4 | `synthetic_trials_v1_sample.csv` | one answered question | 1,178 (first 30 children of #5) | synthetic | this folder | ✅ |
| 5 | `synthetic_trials_v1.csv` | one answered question | 120,187 (3,000 children, 150 DD) | synthetic | `backend/training/risk_profiling/synthetic_phase/artifacts/` | ❌ rebuilt (14.8 MB) |
| 6 | `synthetic_summary_v1.csv` | child × session × task type | 49,226 | synthetic | artifacts | ❌ rebuilt |
| 7 | `synthetic_trials_v1_children.csv` | child: grade, age, DD label of #5 | 3,000 | synthetic | artifacts | ❌ rebuilt |
| 8 | `preprocessed_children_v1.csv` | child: data sufficiency, split | 3,000 | synthetic | artifacts | ❌ rebuilt |
| 9 | `preprocessed_child_task_v1.csv` | child × task type, all sessions pooled | 28,510 | synthetic | artifacts | ❌ rebuilt |
| 10 | `preprocessed_child_area_v1.csv` | child × C2 area, all sessions pooled | 9,947 | synthetic | artifacts | ❌ rebuilt |
| 12 | `features_v1.csv` | child: 21 z-scores (same grade and age), split, sufficiency flags | 3,000 | synthetic | artifacts | ❌ rebuilt |
| 11 | `real_decarli_calibration_z.csv` / `real_decarli_test_z.csv` | real child (Decarli et al., 2020): z-scores + DD label | 50 / 14 | **real** (public research data) | `data/private/risk_profiling/` | ❌ never |

## 2. How they connect

```
Published evidence (meta-analysis effect sizes, Decarli correlations)
        │
        ▼
(1) synthetic_children_v1 ──────────────────────────────┐ research layer (model training)
        │ children answer questions                     │
        ▼                                               │
(5) synthetic_trials_v1  ──► (6) summary  (mastery)     │
        │                                               │
        ▼ Step 2 preprocessing                          │
(8)(9)(10) preprocessed child records ──► (12) features ──► models ──► test on (11) real children
```

The robot system will produce files in the format of **#2/#5** (trials) and **#3/#6** (summary). Real pilot data
replaces the synthetic trials without code changes.

## 3. Which file is "the final dataset"?

- **Operational format** (what the robot system sends and keeps): trials (#5, sample #4) and session summary (#6).
- **Model input** (latest step): **#12 `features_v1.csv`**, one row of z-scores per child, built from the
  preprocessed records #8–#10. Labels for synthetic children: #7.
- **Real test**: #11 (14 Decarli children, never used before the final evaluation).

## 4. Columns

- Trials (#2, #4, #5) and summary (#3, #6): every column, its source component and evidence row:
  [`dataset_schema.md`](../docs/dataset_schema.md).
- Research children (#1) and real children (#11): [`final_dataset.md`](../docs/final_dataset.md).
- Preprocessed records (#8–#10): [`preprocessing.md`](../docs/preprocessing.md).
- Features (#12): [`feature_engineering.md`](../docs/feature_engineering.md).

Key trial columns in one line: `student_id`, `session_id`, `age_months`, `grade` (student profile) ·
`c2_activity`, `c2_area`, `difficulty_tier`, `representation` (C2) · `expected_answer`, `given_answer`,
`accuracy`, `error_value`, `response_time_s`, `attempts`, `robot_demo_triggered` (optional), `valid_trial` (C1) ·
`behavior_label` (C3). This component adds `task_type` (T01–T16), `in_curriculum`, `robot_demo_source`
(`component1` or `derived`) and `rt_valid`.

## 5. What is evidence and what is a placeholder

| Part | Status |
|---|---|
| DD vs typical differences (#1, drive #5) | **Evidence**: meta-analysis effect sizes, Decarli correlations, 5% prevalence |
| Task types, grade rules | **Evidence**: NIE Grade 1–3 guides, dyscalculia tests (BDS, ZAREKI-R, TEDI-MATH) |
| Session design (2 sessions × 16–24 questions) | **Team sources**: pilot plan, C2 |
| Typical % correct, typical RT, attempts, behaviour rates, age per grade | **PLACEHOLDER** (`config/simulation_parameters.json`) until pilot data exists |
| Mastery cut-offs 0.80 / 0.50 | **PLACEHOLDER**, unvalidated (`config/mastery_thresholds.json`) |
| Demo files (#2, #3) | **All performance numbers made up**; format illustration only, never for results |

## 6. Rebuild everything (from the repository root)

```bash
pip install -r backend/training/risk_profiling/synthetic_phase/requirements.txt
# real data (needs the Decarli workbooks in data/raw/decarli2020/, from https://osf.io/gjpb3/)
python backend/training/risk_profiling/synthetic_phase/src/build_real_labeled_participant_dataset.py
python backend/training/risk_profiling/synthetic_phase/src/build_final_dataset.py        # (1) and (11)
# operational and synthetic trials
python backend/training/risk_profiling/synthetic_phase/src/make_demo_trials.py           # (2)
python backend/training/risk_profiling/synthetic_phase/src/build_session_summary.py backend/training/risk_profiling/synthetic_phase/data/demo_cube_assessment_trials.csv backend/training/risk_profiling/synthetic_phase/data/demo_student_performance_summary.csv   # (3)
python backend/training/risk_profiling/synthetic_phase/src/generate_synthetic_trials.py  # (4)-(7), about 5 minutes
python backend/training/risk_profiling/synthetic_phase/src/preprocess.py                 # (8)-(10)
python backend/training/risk_profiling/synthetic_phase/src/build_features.py             # (12)
python -m pytest backend/training/risk_profiling/synthetic_phase/tests                   # 68 tests
```

All generators use fixed seeds, so the files are identical on every rebuild.

## 7. Known limitations

1. Synthetic data reproduces the literature by construction; only the real test (#11) gives evidence of performance.
2. With 16–24 questions per session, each child has a median of 4 questions per task type; per-task scores are
   noisy and effects shrink (g ≈ 0.21–0.47 vs 0.7–0.9 published). Area-level records (#10) are the main input.
3. `age_months` and `grade` are not yet sent by any component; `representation` and the grade × activity table
   await confirmation from C2; robot-demo logging awaits confirmation from C1.
4. Real Decarli children are Italian, aged 7–14, and have no arithmetic measures.
