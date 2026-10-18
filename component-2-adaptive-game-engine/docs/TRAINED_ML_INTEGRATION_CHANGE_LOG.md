# Trained ML Integration Change Log

Date: 17 October 2026  
Working copy: `09_C2_Trained_ML_Integration/component-2-adaptive-game-engine`

The source application and original ASSISTments research workspace were not modified. No application file was deleted.

## Modified files

| Path | Change |
|---|---|
| `README.md` | Describes the trained-primary/fallback workflow and unified demo command. |
| `requirements-runtime.txt` | Adds pinned pandas, NumPy, joblib and XGBoost runtime dependencies. |
| `artifacts/model_manifest.json` | Preserves the frozen research-model contract and registers the new live model/fallback. Backward-compatible frozen scorer keys remain. |
| `src/component2_app/coverage.py` | Accepts candidate-skill predictions; uses model evidence for bounded reinforce/advance and ranking within equally under-covered games. |
| `src/component2_app/service.py` | Builds strictly-prior feature state, invokes the common predictor router inside `next_task`, preserves the service/C1/C4 API, and exports model claim boundaries. |
| `src/recommendation_engine/learner_state.py` | Adds prior correct count and prior interactions-since-correct state fields. |
| `src/recommendation_engine/recommender.py` | Reports trained, Beta fallback and rule pathways separately with provenance. |
| `tests/test_adaptive_assessment.py` | Updates expected live provenance from the prior Beta-only implementation to the trained primary model. |
| `tools/demo_adaptive_assessment.py` | Corrects its claim-boundary text; the new canonical demo is listed below. |
| `docs/ADAPTIVE_ML_FEASIBILITY_AUDIT.md` | Marks the earlier Beta-only feasibility conclusion as historical/superseded. |
| `docs/PP1_INTEGRATION_HANDOVER.md` | Points readers to the trained-ML handover files. |
| `docs/TEST_REPORT.md` | Replaces stale counts with the final 70-test, model-evaluation and demo record. |

## New source, research and test files

| Path | Purpose |
|---|---|
| `src/recommendation_engine/predictor.py` | Replaceable trained predictor + Beta-Bernoulli fallback + explicit abstention. |
| `research/train_transferable_model.py` | Reproducible, learner-split, leakage-safe training/calibration/evaluation script. |
| `tools/demo_trained_ml_assessment.py` | Canonical continuous Grade 3 strong/struggling demonstration through `Component2Service`. |
| `tests/test_trained_ml_workflow.py` | Artifact, feature, fallback, provenance, prediction-sensitive game and difficulty tests. |

## New model artifacts

| Path | Bytes | SHA-256 / note |
|---|---:|---|
| `artifacts/transferable/model.json` | 1,233,236 | `dc31acb1307568b9f7886d98cf163650ba2eb86ed59c938f28af183cc93403f2` |
| `artifacts/transferable/metadata.json` | 2,053 | `c55f1049eb3e27dfffca169ef0ff6cede5ea0565d1c45294b2442fa54d3850a3` |
| `artifacts/transferable/evaluation.json` | 1,062 | `2bc6a95291a2ec1a6984524a8ef36a508382dc4832f3d7640d739fd62a6a865b` |
| `artifacts/transferable/calibrator.joblib` | 831 | Attempted Platt object retained for reproducibility but not loaded because calibration was rejected. |
| `artifacts/transferable/manifest.json` | 627 | Artifact inventory and hashes. |

## New documentation

- `docs/TRAINED_ML_INTEGRATION_PLAN.md`
- `docs/TRAINED_MODEL_DEPLOYMENT_AUDIT.md`
- `docs/TRAINED_ML_INTEGRATION_CHANGE_LOG.md`
- `docs/TRAINED_ML_GITHUB_TRANSFER.md`

## Generated files removed

Python `__pycache__` directories created while testing were removed from the isolated copy and must not be transferred.

## Verification record

- Training/evaluation: completed successfully using the frozen learner partitions.
- Canonical demo: completed successfully for Grade 3 strong and struggling simulations.
- Complete test suite in the pinned runtime: **70 passed; 0 failed; 0 skipped** in 87.202 seconds.
- Frozen `H2_GB_RT_OFF_v1` replay was included by setting `C2_RESEARCH_WORKSPACE`; it passed the saved-reference tolerance check.
- Streamlit smoke tests passed. The emitted “missing ScriptRunContext” lines are expected warnings from Streamlit's bare test runner, not failures.
