# C2 adaptive-question research and training

This directory contains offline research, training, artifact installation, and
simulation tooling for C2. Runtime inference is implemented in
`backend/app/modules/adaptive_questions/`; application startup never trains a
model.

## Training and artifact tools

- `train_transferable_model.py` preserves the transferable-model training entry
  point. It expects the external, read-only ASSISTments research workspace and
  is not required to run the backend.
- `install_runtime_model.py` verifies and copies only `model.json` and
  `calibrator.joblib` from an approved source into the runtime model directory.
  It is intended for controlled artifact replacement or recovery.

The current `model.json` and `calibrator.joblib` are tracked in the team
repository alongside `manifest.json`, `metadata.json`, and `evaluation.json`.
A normal checkout therefore does not require the installer.

For an approved recovery operation, run from the repository root:

```powershell
python backend/training/adaptive_questions/install_runtime_model.py `
  "C:\approved\TRANSFER_H2_GB_RT_OFF_v1"
```

This command replaces runtime artifacts only after manifest checks; do not run
it casually against reviewed artifacts.

## Research notebooks

The six notebooks preserve the staged C2 research workflow:

1. `notebooks/01_Dataset_Exploration_and_Preparation.ipynb` — source-data
   exploration and preparation.
2. `notebooks/02_Chronological_History_and_Feature_Engineering.ipynb` —
   chronological history and leakage-controlled feature engineering.
3. `notebooks/03_Experimental_Model_Training_Methodology.ipynb` — experimental
   training methodology.
4. `notebooks/04_Model_Evaluation_and_Selection.ipynb` — model evaluation and
   selection across the recorded experiments.
5. `notebooks/05_Lightweight_Final_Model_Demonstration.ipynb` — selected
   lightweight model demonstration.
6. `notebooks/06_Adaptive_Recommendation_and_Assessment.ipynb` — adaptive
   recommendation and assessment demonstration.

These are research records, not proof of clinical effectiveness or validation
with children.

## Frozen experiment records

- `experiments/ASSISTments_Step3_Experiment_Registry.csv` records the frozen
  eight-model experiment definitions and configuration registry.
- `experiments/ASSISTments_Step3_Model_Results.csv` records the corresponding
  frozen performance results.

Treat both CSVs as research provenance. Runtime identity and deployment checks
remain governed by the model manifest and metadata.

## Adaptive simulation

`run_adaptive_simulation.py` exercises the stateful C2 service with deterministic
profiles from `simulation_profiles.json`. From the repository root:

```powershell
python backend/training/adaptive_questions/run_adaptive_simulation.py --help
python backend/training/adaptive_questions/run_adaptive_simulation.py --no-write
```

The script uses packages already listed in `backend/requirements.txt` except
`jsonschema`, which it imports directly but the requirements file does not
currently declare. Install it in the review environment before running the
simulation until that dependency declaration is reconciled:

```powershell
python -m pip install jsonschema
```

The simulation uses synthetic profiles. It is useful for integration and policy
checks but is not hardware validation, real-child validation, or clinical
validation.
