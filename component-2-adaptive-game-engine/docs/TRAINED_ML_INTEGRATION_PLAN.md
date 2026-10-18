# Trained ML Integration Plan

Date: 17 October 2026  
Isolated working copy: `09_C2_Trained_ML_Integration/component-2-adaptive-game-engine`

## Guardrails

- Treat `07_GitHub_Adaptive_ML_Refinement/component-2-adaptive-game-engine` as read-only.
- Treat the complete `ASSISTments Dataset` workspace, its frozen datasets, notebooks, models, preprocessors, calibration objects and reports as read-only.
- Preserve the public `Component2Service` workflow and C1/C4 contracts.
- Never synthesize ASSISTments identifiers or platform context for a local cube activity.
- Use only features available before the candidate response.
- Describe simulation as functional evidence, not real-child validation.

## Existing files inspected

- `src/component2_app/service.py`, `database.py`, `coverage.py`, `questions.py`
- `src/recommendation_engine/` candidate, curriculum, learner-state, fallback, local estimator, scorer and recommendation modules
- `config/activity_catalog.csv`, `adaptive_assessment_settings.json`, recommendation/progression/fallback/abstention configuration
- `artifacts/model_manifest.json`
- all current automated tests and the standalone adaptive assessment demo
- original Step 2 feature-generation code and frozen chronological feature parquet
- original Step 3 experiment registry, feature contract, model results, calibration decisions, feature importance, model artifacts and preprocessors

## Expected changes

- `src/component2_app/service.py`: connect candidate scoring and the shared recommendation policy to the existing assessment loop.
- `src/recommendation_engine/recommender.py`: accept candidate-specific trained-model evidence while retaining safe fallback.
- `src/recommendation_engine/learner_state.py` and/or a new feature builder: expose strictly prior transferable fields.
- `artifacts/model_manifest.json`: retain the frozen research-model boundary and register the deployment model.
- `requirements-runtime.txt` / `requirements-research.txt`: change only if artifact loading or evaluation requires it.
- `README.md`: document the unified command and scientific claim boundary.

## New files

- `src/recommendation_engine/predictor.py`: common prediction result/interface and explicit abstention.
- `src/recommendation_engine/transferable_model.py`: leakage-safe local candidate feature construction and frozen artifact loader.
- `src/recommendation_engine/recommendation_policy.py`: one prediction-to-decision policy used by the service.
- `research/train_transferable_model.py`: reproducible learner-split training/evaluation pipeline against frozen ASSISTments history.
- `artifacts/transferable/`: newly trained model, preprocessing metadata, evaluation metrics and hashes.
- `tools/demo_trained_ml_assessment.py`: one continuous Grade 3 strong/struggling workflow through `Component2Service` and C4 export.
- focused tests for feature chronology, provenance, fallback, prediction-sensitive decisions, curriculum safety, coverage and summary export.
- `docs/TRAINED_MODEL_DEPLOYMENT_AUDIT.md`
- `docs/TRAINED_ML_INTEGRATION_CHANGE_LOG.md`
- `docs/TRAINED_ML_GITHUB_TRANSFER.md`

## Model artifacts required

- Research benchmark (external/read-only): the eight original Step 3 model artifacts and their shared train-only preprocessor/calibration reports.
- Proposed deployment artifact (copied into this isolated application): a newly named classifier trained from frozen Step 2 rows using only fields reproducible from local history and local candidate metadata.
- Artifact metadata must record feature order, target definition, learner-level split, metrics, dependency versions, source provenance and SHA-256.

## Dependencies

- Runtime: Python standard library plus the smallest compatible loader needed by the selected deployment artifact.
- Research-only training/evaluation: DuckDB, pandas, NumPy, scikit-learn and XGBoost as already pinned in `requirements-research.txt`.

## Risks to the current backend

- Domain shift: ASSISTments predictive validity does not establish validity for Sri Lankan Grade 1-5 cube games.
- Feature mismatch: candidate difficulty/tier semantics must be explicit and comparable; unsupported fields must trigger abstention.
- Regression risk: model-driven tier selection could disrupt coverage, readiness or bounded reinforcement; curriculum and safety filters remain authoritative.
- Cold start: insufficient same-skill history must use the documented Beta-Bernoulli/rule fallback.
- Artifact portability: model loading must fail closed and report provenance rather than silently substituting values.
- Performance claims: held-out ASSISTments metrics, policy tests and simulated workflows must be reported separately from real-child evaluation.

## Verification plan

1. Run the existing baseline suite in the isolated copy.
2. Reproduce the eight-model audit from frozen reports.
3. Train and evaluate the transferable model with learner-level separation and chronological features only.
4. Add focused unit/integration tests, including counterfactual predictions that change decisions.
5. Run the full suite and the end-to-end strong/struggling demo.
6. Record exact outputs, skipped tests, artifact hashes and every changed path in the final audit, change log and transfer guide.
