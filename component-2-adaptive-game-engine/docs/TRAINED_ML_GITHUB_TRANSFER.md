# Trained ML GitHub Transfer Guide

## Source and destination

Copy only the listed paths from:

`C:\Users\Monil A\Desktop\Research Documents\09_C2_Trained_ML_Integration\component-2-adaptive-game-engine`

into the repository root of the Component 2 GitHub codebase. Do not replace an entire directory. Review repository-local changes before overwriting any listed existing file.

## Existing files to replace

- `README.md`
- `requirements-runtime.txt`
- `artifacts/model_manifest.json`
- `src/component2_app/coverage.py`
- `src/component2_app/service.py`
- `src/recommendation_engine/learner_state.py`
- `src/recommendation_engine/recommender.py`
- `tests/test_adaptive_assessment.py`
- `tools/demo_adaptive_assessment.py`
- `docs/ADAPTIVE_ML_FEASIBILITY_AUDIT.md`
- `docs/PP1_INTEGRATION_HANDOVER.md`
- `docs/TEST_REPORT.md`

## New files to add

- `src/recommendation_engine/predictor.py`
- `research/train_transferable_model.py`
- `tools/demo_trained_ml_assessment.py`
- `tests/test_trained_ml_workflow.py`
- `artifacts/transferable/model.json`
- `artifacts/transferable/metadata.json`
- `artifacts/transferable/evaluation.json`
- `artifacts/transferable/calibrator.joblib`
- `artifacts/transferable/manifest.json`
- `docs/TRAINED_ML_INTEGRATION_PLAN.md`
- `docs/TRAINED_MODEL_DEPLOYMENT_AUDIT.md`
- `docs/TRAINED_ML_INTEGRATION_CHANGE_LOG.md`
- `docs/TRAINED_ML_GITHUB_TRANSFER.md`

## Existing files to leave untouched

Leave every unlisted file intact, especially:

- SQLite migrations and database files;
- C1, C4 and robot-task JSON schemas;
- activity, curriculum, progression, abstention and fallback configuration;
- question generators and all eight game definitions;
- Streamlit views and `streamlit_app.py`;
- original research notebooks, source datasets, frozen eight models, preprocessors, calibration artifacts and result reports.

Do not copy any `__pycache__`, `.pyc`, temporary database or local environment directory.

## Required runtime artifacts and dependencies

The five files under `artifacts/transferable/` are required. The live model does not need the external ASSISTments workspace. The external workspace is needed only to reproduce training or run frozen-model replay tests.

New live dependencies already pinned in `requirements-runtime.txt`:

- `pandas==3.0.1`
- `numpy==2.3.5`
- `joblib==1.6.0`
- `xgboost==3.0.4`

`streamlit==1.65.0` remains pinned. Research-only training continues to use the packages in `requirements-research.txt`.

## PowerShell setup and canonical demo

From the repository root:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-runtime.txt
$env:PYTHONPATH = (Resolve-Path .\src).Path
.\.venv\Scripts\python.exe .\tools\demo_trained_ml_assessment.py
```

The demo needs no dataset and writes no persistent learner database; it uses isolated temporary databases.

## Test commands

Application tests without the external frozen workspace:

```powershell
$env:PYTHONPATH = (Resolve-Path .\src).Path
Remove-Item Env:C2_RESEARCH_WORKSPACE -ErrorAction SilentlyContinue
.\.venv\Scripts\python.exe -m unittest discover -s tests -p 'test_*.py' -v
```

Complete tests including frozen-model replay:

```powershell
$env:PYTHONPATH = (Resolve-Path .\src).Path
$env:C2_RESEARCH_WORKSPACE = 'C:\Users\Monil A\Desktop\Research Documents\ASSISTments Dataset'
.\.venv\Scripts\python.exe -m unittest discover -s tests -p 'test_*.py' -v
```

Reproduce the deployment model from the read-only research workspace:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-research.txt
.\.venv\Scripts\python.exe .\research\train_transferable_model.py --workspace 'C:\Users\Monil A\Desktop\Research Documents\ASSISTments Dataset'
```

Retraining overwrites files in `artifacts/transferable/`; do this only in a development branch/copy and review the new hashes and metrics before transfer.

## Known limitations

- Test metrics are held-out ASSISTments results, not local-child or robotic-game validation.
- The live feature mapping assumes correctness/history semantics transfer across domains; calibration under local game conditions is unknown.
- The model has no direct tier/difficulty feature because the source dataset has no defensibly comparable Tier A/Tier B label. Difficulty selection combines generic predicted success with independent local readiness rules.
- Previously unseen local skills with identical history can receive identical probabilities; deterministic curriculum/coverage tie-breaking then decides the game.
- Thresholds 0.40, 0.65 and 0.75 are documented project heuristics, not clinically or pedagogically validated cutoffs.
- Beta-Bernoulli remains an untrained local baseline/fallback and abstains before two same-skill observations.
- C4 output remains descriptive and non-diagnostic.
