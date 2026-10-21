# Risk-profiling training — synthetic phase

> **Status: development on synthetic data.** Everything in this folder is built and tested on
> synthetic children (plus the public Decarli 2020 data for checking). It is kept separate until
> real pilot data is collected; then the final component is built next to this folder.

Offline mastery and screening-risk research belongs here. Thresholds and models
must remain labelled unvalidated until the agreed research validation succeeds.

## Contents

| Path | Purpose |
|---|---|
| `docs/evidence_register.md` | Every design decision with its source and status (verified / open / assumption) |
| `docs/feature_evidence.md` | Step 1: candidate features, effect sizes and provisional weights |
| `docs/nie_curriculum_map.md` | NIE Grade 1–3 competencies used for age expectations |
| `docs/task_taxonomy.md` | Task types T01–T16: curriculum × dyscalculia tests × meta-analysis evidence |
| `docs/team_interface.md` | Fields received from the cube-vision, adaptive-question and behaviour modules |
| `docs/data_source.md` | Citation and column meanings of the Decarli et al. (2020) research data |
| `src/compute_effect_sizes.py` | Recomputes DD-vs-control effect sizes → `results/decarli2020_effect_sizes.csv` |
| `src/build_real_labeled_participant_dataset.py` | Merges the four Decarli workbooks and makes a stratified participant split |
| `src/build_final_dataset.py` | Builds the synthetic training set and the real calibration/test sets (see `docs/final_dataset.md`) |
| `docs/final_dataset.md` | Data card for the research (z-score) datasets |
| `docs/dataset_schema.md` | Schema v1 of the operational datasets (trials, session summary) |
| `src/dataset_schema.py` | Column definitions, task-type assignment, validation |
| `src/build_session_summary.py` | Trials → session summary (`python src/build_session_summary.py <trials.csv> <summary.csv>`) |
| `src/make_demo_trials.py` | Writes a clearly labelled DEMO trials file (placeholder numbers) |
| `config/mastery_thresholds.json` | NIE mastery cut-offs, **unvalidated** |
| `src/generate_synthetic_trials.py` | Synthetic trials + summary from the research children (writes `artifacts/`, ~5 min) |
| `src/preprocess.py`, `docs/preprocessing.md` | Step 2: trials → one clean record per child (per task type and per area), split 70/15/15 |
| `src/build_features.py`, `docs/feature_engineering.md` | Step 3: one row of z-scores per child (same grade and age, train-split norms) |
| `src/model_formula.py`, `docs/modelling.md` | Step 4a: evidence-weighted risk formula and its evaluation |
| `src/model_svm.py` | Step 4b: SVM (RBF), same features/split/bands as 4a |
| `src/model_random_forest.py` | Step 4c: Random Forest, out-of-fold train scores, permutation importance (slow: ~2 h on a laptop) |
| `src/model_xgboost.py` | Step 4d: XGBoost, out-of-fold train scores, saves the model to `artifacts/` (~1 min) |
| `src/explain_shap.py` | Step 4d: SHAP contributions per feature and skill, agreement with evidence weights, worked example |
| `src/model_logistic.py` | Step 4e: logistic regression (reference), learned vs evidence weights (~10 s) |
| `src/model_selection.py`, `config/selected_model.json` | Step 4f: compares the five models, chooses one by a written rule (formula selected) |
| `src/explain_formula.py`, `src/child_report.py`, `config/report_templates/`, `docs/child_report.md` | Step 4g: per-child explanation by skill and plain-language teacher/parent report (English; add languages as template files) |
| `src/profiles.py`, `src/clustering.py`, `config/profiles.json`, `docs/clustering.md` | Step 5: difficulty profiles; rule-based (used) and GMM / k-means with stability checks (rejected on synthetic data) |
| `src/final_evaluation.py`, `docs/tripod_ai_report.md` | Step 6: single evaluation of the frozen model on the test data; TRIPOD+AI report |
| `src/make_figures.py`, `results/figures/` | Eight presentation figures drawn from the saved results (model comparison, ROC, sessions, cut-offs, weights, SHAP, grades, profiles) |
| `src/mastery_data.py`, `src/mastery_models.py`, `src/mastery_ablation.py`, `src/mastery_explain.py`, `src/mastery_final.py`, `docs/mastery_classification.md` | Proposal objectives: 3-tier mastery labels and features, stacking ensemble vs single models, ablation, SHAP + LIME, final test |
| `dashboard-react/` ([guide](dashboard-react/react_dashboard_guide.md)) | Colourful React dashboard prototype: `npm install`, `npm run dev` |
| `src/sensitivity_analysis.py` | Step 4-S: screening cut-off, two-stage rule, more sessions (~15 min) |
| `config/simulation_parameters.json` | Simulation settings; each marked with its source or PLACEHOLDER |
| `tests/` | `python -m pytest backend/training/risk_profiling/synthetic_phase/tests` |

**Guide to every dataset:** [`data/datasets_guide.md`](data/datasets_guide.md)

## Data (not in Git)

Download the four workbooks from https://osf.io/gjpb3/ into `data/raw/decarli2020/`.
Participant-level outputs are written to `data/private/risk_profiling/`. Both
folders are ignored by Git.

```bash
python -m pip install -r backend/training/risk_profiling/synthetic_phase/requirements.txt
python backend/training/risk_profiling/synthetic_phase/src/compute_effect_sizes.py
python backend/training/risk_profiling/synthetic_phase/src/build_real_labeled_participant_dataset.py
python backend/training/risk_profiling/synthetic_phase/src/build_final_dataset.py
```

The synthetic training set is written to `data/` in this folder
(committed). Real child-level files go to `data/private/risk_profiling/` (ignored).

Status: Step 1 (feature evidence) and Step 3 (final dataset v1) complete. The
risk formula and model evaluation are next.
