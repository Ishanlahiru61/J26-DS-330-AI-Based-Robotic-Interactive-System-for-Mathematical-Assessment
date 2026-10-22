# Operational datasets — schema v1

The two datasets this component uses at runtime. Column definitions in code: `src/dataset_schema.py`.
Built by `src/build_session_summary.py`. Tested by `tests/test_operational_datasets.py`.

```
C1 + C2 + C3 + student_profiles ──► (1) cube_assessment_trials ──► (2) student_performance_summary
        one row per question                                         one row per student × session × task type
```

## (1) cube_assessment_trials — one row per answered question

| Column | Type | Source | Meaning | Register |
|---|---|---|---|---|
| `data_origin` | real / demo / synthetic | this component | Real rows can never be confused with demo rows | G5 |
| `student_id` | text | student_profiles | Pseudonymous ID; **no name, no date of birth** | B1, B2b |
| `session_id`, `session_index` | text, int | orchestrator | Session and its order (1 = first) | A4, B4 |
| `question_id`, `interaction_position` | text, int | C2 | Question and its order in the session | B1 |
| `age_months` | int | student_profiles (from DOB) | Age at the session date | B2 |
| `grade` | 1–3 | student_profiles | School grade (what has been taught) | B2, C2 |
| `c2_activity` | 8 values | C2 | counting, comparison, ordering, place_value, addition, subtraction, multiplication, division | B3 |
| `c2_area` | 4 values | C2 | number_sense, number_relationships, arithmetic_calculation, multiplicative_reasoning | B3 |
| `difficulty_tier` | A / B | C2 | Foundational / extended | B3 |
| `representation` | non_symbolic / symbolic / verbal / mixed | C2 (**requested**) | Cubes vs digits; needed to tell T04 from T05 | C7 |
| `expected_answer`, `given_answer` | number | C1 | Correct answer; answer detected from the cubes | B1 |
| `accuracy` | 0 / 1 | C1 | First attempt correct | B1 |
| `error_value` | number | C1 | \|expected − given\| | B1 |
| `response_time_s` | seconds | C1 | Time to first response | B1 |
| `attempts` | int | C1 | Attempts (≥ 1) | B1 |
| `robot_demo_triggered` | 0 / 1, **optional** | C1 | Robot demonstrated (wrong answer or > 15 s). If C1 does not send it, it is derived by C1's rule (see `robot_demo_source`) | I3, B1c |
| `valid_trial` | 0 / 1 | C1 | Validation status usable | B1 |
| `behavior_label` | finger_counting / other_hand_movement / uncertain | C3 | `uncertain` is treated as missing, not as "no finger counting" | I1, I2 |

**Derived by this component** (not sent by anyone):

| Column | Rule | Register |
|---|---|---|
| `task_type`, `math_domain`, `task_form`, `operation` | From C2 activity + representation + set size + tier ([task_taxonomy.md](task_taxonomy.md) Section 4). Empty when information is missing (e.g. comparison without representation) | C7 |
| `in_curriculum` | 1 if the task type is taught at the child's grade (NIE) | C3b, C3c |
| `robot_demo_source` | `component1` = value logged by C1; `derived` = filled by this component with C1's rule (wrong answer **or** RT ≥ 15 s). A C1 value always has priority; the two are never treated as the same source | B1c |
| `rt_valid` | 1 if valid, no robot demo, and RT < 15 s | I3 |

## (2) student_performance_summary — one row per student × session × task type

| Column | Rule |
|---|---|
| keys | `data_origin`, `student_id`, `session_id`, `session_index`, `age_months`, `grade`, `task_type`, `c2_activity`, `c2_area` |
| `in_curriculum`, `risk_use` | From the taxonomy: `risk`, `control` (subitizing), `record` (weight 0) |
| `n_trials`, `n_valid_trials` | Count of questions, and of valid questions |
| `accuracy_rate` | Correct ÷ valid questions |
| `error_rate` | 1 − accuracy_rate |
| `mean_relative_error` | Mean of error_value ÷ expected_answer (F10) |
| `median_rt_correct_s` | Median RT of **correct** answers with `rt_valid` = 1 (F2, F4, F5, F8 definition) |
| `rt_cov` | SD ÷ mean of valid RTs (F12) |
| `mean_attempts` | Mean attempts (F11) |
| `robot_demo_rate_component1` | Share of questions with a demo, **C1-logged rows only** |
| `robot_demo_rate_derived`, `n_robot_demo_derived` | Same share for rows filled by the rule, and how many such rows (kept apart from the logged rate, B1c) |
| `finger_counting_rate` | Share of questions with finger counting (uncertain excluded) |
| `improvement_slope` | Slope of accuracy_rate over this and earlier sessions for the same task type; empty in session 1 (F13, A4) |
| `mastery_level` | NIE three levels from `accuracy_rate`; empty if fewer than `min_valid_trials` (C6, M1) |
| `age_grade_mismatch` | **Empty until** the age–grade rule is sourced (B2c, C1) |

## Status of the files

| File | What it is |
|---|---|
| `data/demo_cube_assessment_trials.csv` | **DEMO** 6 children × 2 sessions × 20 questions. All performance numbers are **placeholders** (`src/make_demo_trials.py`). DEMO-006 has no logged robot demos, to show the fallback. Shows the format only; never used for training or results |
| `data/demo_student_performance_summary.csv` | The summary built from the demo trials |
| real files | Come from the pilot. Child-level real data goes to `data/private/` (not in Git) |

## Synthetic trials v1 (`src/generate_synthetic_trials.py`)

The 10,000 research children ([final_dataset.md](final_dataset.md)) answer questions in this schema, so the whole
pipeline (trials → summary → features → model) runs on the format the robot will produce.

| Item | Value | Source |
|---|---|---|
| Children | 3,000 (150 DD, 5%), Grades 1–3 in equal thirds | G3, G7 |
| Sessions × questions | 2 sessions × 16–24 questions | [P-C3] pilot plan; [C2-msg] |
| Activities | uniform over activities taught at the grade | NIE grade tables |
| DD effect | from each child's evidence-based z-scores (F1–F8); task types without evidence get an independent z with **no DD shift** | G1–G4, G8 |
| Absolute levels (typical % correct, typical RT, attempts, behaviour) | **PLACEHOLDERS** in `config/simulation_parameters.json` | to replace with pilot data |
| Output | 120,187 trials, 49,226 summary rows | `artifacts/` (not in Git, rebuild with the script); 30-child sample in `data/synthetic_trials_v1_sample.csv` |

**Check** (`results/synthetic_trials_check.csv`): DD children are worse on every risk task type and show no
difference on task types without evidence, as designed. The child-level effects (g ≈ 0.21–0.47 on error rate)
are **much smaller** than the published effects (g ≈ 0.7–0.9), because each child answers only **3–5 questions
per task type** over 2 sessions. Few binary answers give an unreliable score, and unreliability shrinks
observed group differences (classical test-theory attenuation). **Design consequence:** the pilot plan
(2 sessions × 16–24 questions) is too short for reliable per-task scores. Pool task types into areas, and/or
use more sessions (register B4, G11).

## Known gaps

1. `representation`, the grade × activity table and place-value area need confirmation from C2 (task_taxonomy.md Section 4).
2. `age_months` and `grade` are not yet in the shared contract (B2b).
3. Mastery cut-offs (0.80 / 0.50) are **unvalidated placeholders** in `config/mastery_thresholds.json` (M1).
4. With 16–24 questions per session, most cells have 2–3 questions; single-session rates are noisy (B4).
