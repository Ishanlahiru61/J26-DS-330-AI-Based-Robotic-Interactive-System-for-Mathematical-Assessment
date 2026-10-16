# Executable notebook guide

These notebooks reconstruct and explain the frozen Component 2 ASSISTments research without replacing its methodology or retraining by default.

## Environment

Use Python 3.12 with `pandas`, `numpy`, `pyarrow`, `scikit-learn`, `matplotlib`, `joblib`, `xgboost==3.0.4`, `nbformat`, and `nbclient`. Start Jupyter in this folder, or set:

```powershell
$env:ASSISTMENTS_WORKSPACE = 'C:\Users\Monil A\Desktop\Research Documents\ASSISTments Dataset'
```

Every notebook checks that the local workspace exists and raises a clear `FileNotFoundError` if it does not.

## Notebook modes

| Notebook | Default mode | What it does | Expensive option |
|---|---|---|---|
| `01_Dataset_Exploration_and_Preparation.ipynb` | Lightweight | Loads frozen Step 1 profile, dictionary and distributions | `RUN_RAW_SCAN=1` reads only a 10,000-row raw schema sample |
| `02_Chronological_History_and_Feature_Engineering.ipynb` | Lightweight | Loads the exact feature contract and one 500-row real Parquet batch | None |
| `03_Experimental_Model_Training_Methodology.ipynb` | Lightweight documentation | Loads the executed registry and exact parameters | `RUN_FULL_EXPERIMENTS=1` invokes the authoritative full Step 3 pipeline |
| `04_Model_Evaluation_and_Selection.ipynb` | Lightweight | Loads frozen metrics and post-freeze threshold diagnostics | None |
| `05_Lightweight_Final_Model_Demonstration.ipynb` | Lightweight inference | Verifies hashes and reproduces one frozen test-row probability | None; no training |

## Full reproduction warning

The original research pipeline operates on 6.1 million interactions. Full Step 1–3 reproduction requires the raw CSV, substantial disk/RAM, the original dependencies, and a long runtime. Notebook 3 does not silently run this workload. Its opt-in cell calls the original `03_Experimental_Modeling/run_step3_pipeline.py`; it does not train a small toy model and does not overwrite the frozen reported results unless the researcher deliberately authorizes the run in the original workspace.

## Interpretation boundary

Notebook 5 produces a first-attempt correctness probability for an ASSISTments observation. It does not recommend one of the eight robot games, estimate dyscalculia risk, or validate transfer to Sri Lankan learners. The operational prototype correctly leaves local `predicted_probability` as null.

