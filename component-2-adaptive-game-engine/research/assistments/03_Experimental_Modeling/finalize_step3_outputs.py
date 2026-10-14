#!/usr/bin/env python3
"""Finalize reports after all expensive model outputs have been persisted."""
import json
import os
import platform
import time

import duckdb
import numpy as np
import pandas as pd
import scipy
import sklearn
import xgboost

import run_step3_pipeline as p


def main():
    results = pd.read_csv(p.OUTPUTS / "ASSISTments_Step3_Model_Results.csv")
    uncertainty = pd.read_csv(p.OUTPUTS / "ASSISTments_Step3_Learner_Level_Uncertainty.csv")
    subgroup = pd.read_csv(p.OUTPUTS / "ASSISTments_Step3_Subgroup_Diagnostics.csv")
    registry = pd.read_csv(p.OUTPUTS / "ASSISTments_Step3_Experiment_Registry.csv")
    audit = pd.read_csv(p.OUTPUTS / "ASSISTments_Step3_Split_Audit.csv")
    reliability = pd.read_csv(p.OUTPUTS / "ASSISTments_Step3_Reliability_Summary.csv")
    selected = json.loads((p.OUTPUTS / "ASSISTments_Step3_Selected_Configuration.json").read_text(encoding="utf-8"))
    p.plot_reliability(reliability, ["H0_LR_RT_OFF", "H1_LR_RT_OFF", "H2_LR_RT_OFF", "H2_GB_RT_OFF", selected["experiment_id"]])
    p.write_reports(results, uncertainty, subgroup, registry, selected, audit)
    p.create_notebook()
    runtime = {
        "total_seconds": None,
        "finalization_seconds": 0,
        "model_runtime_seconds_sum": float(registry.runtime_seconds.sum()),
        "note": "Original full run completed all models and analyses, then required report-only recovery after optional Markdown dependency was absent.",
        "platform": platform.platform(),
        "python": platform.python_version(),
        "packages": {"numpy": np.__version__, "pandas": pd.__version__, "scipy": scipy.__version__, "scikit-learn": sklearn.__version__, "xgboost": xgboost.__version__, "duckdb": duckdb.__version__},
        "cpu_count": os.cpu_count(),
        "peak_memory_not_instrumented": True,
    }
    (p.OUTPUTS / "ASSISTments_Step3_Runtime_Environment.json").write_text(json.dumps(runtime, indent=2), encoding="utf-8")
    p.create_manifest()
    print(f"Finalization complete. Selected: {selected['experiment_id']}")


if __name__ == "__main__":
    main()
