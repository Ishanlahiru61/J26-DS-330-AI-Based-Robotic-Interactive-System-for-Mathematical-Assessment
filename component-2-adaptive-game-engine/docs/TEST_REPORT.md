# Verification Report

Final run: 17 October 2026  
Environment: Python 3.12; dependencies pinned by `requirements-runtime.txt` and `requirements-research.txt`; Windows.

## Complete command

```powershell
$env:PYTHONPATH = (Resolve-Path .\src).Path
$env:C2_RESEARCH_WORKSPACE = 'C:\Users\Monil A\Desktop\Research Documents\ASSISTments Dataset'
python -m unittest discover -s tests -p 'test_*.py' -v
```

Result: **70 passed, 0 failed, 0 skipped** in 87.202 seconds.

The run included:

- the original backend, question, all-grade end-to-end and Streamlit smoke tests;
- frozen `H2_GB_RT_OFF_v1` schema/provenance/reference replay against the external read-only workspace;
- trained artifact loading and prediction provenance;
- strict exclusion of current-response and ASSISTments identifier features;
- explicit cold-start, missing-artifact, Beta-Bernoulli and rule abstention/fallback behavior;
- a counterfactual test where changing only candidate probabilities changes the selected game;
- a counterfactual test where probability thresholds change ADVANCE versus REINFORCE while Tier B readiness remains mandatory;
- curriculum, prerequisite, hardware, bounded reinforcement, coverage, session isolation, C1 and descriptive C4 behavior.

Streamlit emitted its expected bare-runner `missing ScriptRunContext` warnings; every smoke test passed.

## Model training/evaluation

`research/train_transferable_model.py` completed successfully using frozen learner-disjoint partitions. The deployment artifact used a deterministic 1,200,000-row training sample and 300,000-row validation sample, then evaluated the complete 937,043-row test partition.

| Evaluation | ROC-AUC | Log loss | Brier | Rows |
|---|---:|---:|---:|---:|
| TRANSFER_H2_GB_RT_OFF_v1 native test | 0.712077 | 0.563200 | 0.190298 | 937,043 |
| TRANSFER attempted Platt test | 0.712077 | 0.563208 | 0.190301 | 937,043 |
| Beta-Bernoulli supported test subset | 0.679452 | 0.589205 | 0.201510 | 316,502 |
| H2_GB_RT_OFF_v1 retained Platt reference | 0.750158 | 0.534442 | 0.178955 | 937,043 |

Platt calibration for the transferable model was rejected by the validation improvement rule, so native probabilities are used.

## Demonstration

`tools/demo_trained_ml_assessment.py` passed. The strong Grade 3 simulation completed all eight skills in 29 valid responses with 28 trained-ML decisions and Tier B advancement. The struggling simulation completed in 31 valid responses with 30 trained-ML decisions and bounded Tier A reinforcement. Both produced a descriptive `C2_TO_C4_v2.0.0` export.

These are functional simulations, not real-child performance results.
