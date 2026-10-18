# Adaptive Question Training

Offline training and artifact installation for C2 live here. Runtime inference
is under `backend/app/modules/adaptive_questions/`; it never trains a model.

- `train_transferable_model.py` preserves the Version 10 transferable-model
  training entry point. It expects the external, read-only ASSISTments research
  workspace and is not part of application startup.
- `install_runtime_model.py` copies only `model.json` and
  `calibrator.joblib` from an approved source, checking the committed manifest
  before and after the copy.

From the repository root:

```powershell
python backend/training/adaptive_questions/install_runtime_model.py `
  "C:\approved\TRANSFER_H2_GB_RT_OFF_v1"
```

The binary files remain ignored by Git. `manifest.json`, `metadata.json`, and
`evaluation.json` are tracked so deployments can verify identity, feature
order, limitations, and held-out ASSISTments metrics.
