# Adaptive-question runtime models

This directory contains the repository-managed C2 inference package:

| File | Purpose |
| --- | --- |
| `transferable/model.json` | XGBoost predictor used by the runtime |
| `transferable/calibrator.joblib` | Approved calibration artifact; integrity checked and loaded only when metadata enables runtime calibration |
| `transferable/manifest.json` | Required filenames and SHA-256 identities |
| `transferable/metadata.json` | Model version, 25-feature order, training policy, and runtime calibration flag |
| `transferable/evaluation.json` | Frozen held-out research metrics and limitations |

Both `model.json` and `calibrator.joblib` are tracked team-repository artifacts.
They are available in a normal checkout; the training installer is an optional
controlled replacement/recovery tool, not a routine application-startup step.

The runtime verifies `transferable/manifest.json` before loading. Do not edit, regenerate, or
replace any artifact without the model-governance and review process, and update
the manifest atomically with any approved artifact change.

Offline training, provenance, six research notebooks, and the frozen eight-model
experiment registry/results are documented in
`backend/training/adaptive_questions/README.md`.
