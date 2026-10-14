#!/usr/bin/env python3
"""Independent validator for frozen ASSISTments Step-3 artifacts."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parent
STEP2 = ROOT.parent / "02_Chronological_Learner_History"
SOURCE = STEP2 / "data" / "processed" / "ASSISTments_2012_2013_history_features.parquet"
OUT = ROOT / "outputs"
PART = ROOT / "data" / "partitions"
PREP = ROOT / "data" / "prepared"
MODELS = ROOT / "models"
SEED = 20260928
EXPECTED_HASH = "7c422ba3dcc536b2d2d88550dd4a4b6ce546082609f17eab3d17f5c4939d97ef"
EXPECTED_ELIGIBLE = 6_117_947
RT = {"previous_ms_first_response", "previous_interaction_rt_valid", "prior_valid_rt_count", "prior_mean_ms_first_response"}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(8 * 1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    checks = []
    def check(name: str, condition: bool, detail: str = "") -> None:
        checks.append({"check": name, "status": "PASS" if condition else "FAIL", "detail": detail})
        if not condition:
            raise AssertionError(f"{name}: {detail}")

    check("Step-2 input hash", sha256(SOURCE) == EXPECTED_HASH, sha256(SOURCE))
    learners = {n: pd.read_csv(PART / f"{n}_learners.csv")["user_id"].to_numpy() for n in ["train", "validation", "test"]}
    sets = {k: set(v.tolist()) for k, v in learners.items()}
    check("train-validation learner disjointness", not (sets["train"] & sets["validation"]))
    check("train-test learner disjointness", not (sets["train"] & sets["test"]))
    check("validation-test learner disjointness", not (sets["validation"] & sets["test"]))
    check("all learners assigned once", len(set.union(*sets.values())) == sum(map(len, sets.values())) == 46674)

    # Reproduce split directly from sorted learner IDs.
    all_ids = np.sort(np.concatenate(list(learners.values())))
    rng = np.random.default_rng(SEED)
    shuffled = all_ids.copy(); rng.shuffle(shuffled)
    n_train = int(round(len(shuffled) * .70)); n_val = int(round(len(shuffled) * .15))
    expected = {"train": np.sort(shuffled[:n_train]), "validation": np.sort(shuffled[n_train:n_train+n_val]), "test": np.sort(shuffled[n_train+n_val:])}
    check("split reproducibility", all(np.array_equal(expected[k], learners[k]) for k in expected), f"seed={SEED}")

    supervised = {n: pq.read_table(PREP / f"{n}_supervised_rows.parquet").to_pandas() for n in ["train", "validation", "test"]}
    check("supervised target total", sum(len(x) for x in supervised.values()) == EXPECTED_ELIGIBLE)
    check("only binary eligible targets", all(set(x["y_true"].unique()).issubset({0, 1}) for x in supervised.values()))
    check("supervised learners respect partitions", all(set(supervised[n].user_id.unique()) <= sets[n] for n in supervised))

    contract = pd.read_csv(OUT / "ASSISTments_Step3_Model_Feature_Contract.csv")
    hsets = {h: set(contract.loc[contract[h] == 1, "feature"]) for h in ["H0", "H1", "H2"]}
    check("feature contract counts", [len(hsets[h]) for h in ["H0", "H1", "H2"]] == [14, 29, 42])
    check("H0 subset H1 subset H2", hsets["H0"] < hsets["H1"] < hsets["H2"])
    tagged_rt = set(contract.loc[contract.RT_dependent == 1, "feature"])
    check("approved RT tag exact", tagged_rt == RT, str(sorted(tagged_rt)))
    pair = pd.read_csv(OUT / "ASSISTments_Step3_RT_Feature_Pair_Audit.csv")
    check("RT ON adds approved RT only", pair["only_approved_rt_added"].astype(bool).all())

    prep_audit = json.loads((OUT / "ASSISTments_Step3_Preprocessing_Fit_Audit.json").read_text(encoding="utf-8"))
    check("preprocessing fitted on train only", prep_audit["validation_learners_used"] == 0 and prep_audit["test_learners_used"] == 0)
    check("no target used in fitted transformations", prep_audit["target_used_for_rt_or_encoding"] is False)
    check("fitted train learner hash", prep_audit["fit_learner_file_sha256"] == sha256(PART / "train_learners.csv"))
    check("RT methodology", prep_audit["rt_log_transform"] == "log1p" and "MAD" in prep_audit["rt_robust_scale"])
    check("no empirical difficulty fitted", prep_audit["difficulty_proxy"] == "none")

    registry = pd.read_csv(OUT / "ASSISTments_Step3_Experiment_Registry.csv")
    results = pd.read_csv(OUT / "ASSISTments_Step3_Model_Results.csv")
    expected_exps = {f"{h}_LR_RT_{rt}" for h in ["H0", "H1", "H2"] for rt in ["OFF", "ON"]} | {f"H2_GB_RT_{rt}" for rt in ["OFF", "ON"]}
    check("experiment registry complete", set(registry.experiment_id) == expected_exps and (registry.status == "COMPLETE").all())
    check("results include native validation/test", all(len(results[(results.experiment_id == e) & (results.calibration == "native")]) == 2 for e in expected_exps))
    check("results row counts consistent", all(int(r.n_rows) == len(supervised[r.split]) for _, r in results.iterrows()))
    check("result probabilities imply finite metrics", np.isfinite(results[["Log Loss", "Brier Score", "ROC-AUC", "PR-AUC"]].to_numpy()).all())

    prediction_files = list(PREP.glob("predictions_*.parquet"))
    native_files = [p for p in prediction_files if p.stem.endswith("_native")]
    check("native prediction file count", len(native_files) == 16, str(len(native_files)))
    for path in prediction_files:
        table = pq.read_table(path, columns=["user_id", "split", "y_true", "predicted_probability"])
        frame = table.to_pandas()
        split = str(frame["split"].iloc[0])
        check(f"prediction count {path.name}", len(frame) == len(supervised[split]))
        check(f"probability bounds {path.name}", frame.predicted_probability.between(0, 1, inclusive="both").all())
        check(f"prediction targets binary {path.name}", set(frame.y_true.unique()).issubset({0, 1}))

    for _, r in registry.iterrows():
        check(f"model exists {r.experiment_id}", Path(r.model_artifact).exists())
    check("preprocessor exists", (MODELS / "preprocessors" / "step3_h2_rt_on_preprocessor_v1.joblib").exists())

    manifest_path = OUT / "ASSISTments_Step3_Artifact_Manifest.csv"
    manifest = pd.read_csv(manifest_path)
    for _, r in manifest.iterrows():
        p = Path(r.path)
        check(f"manifest artifact exists {p.name}", p.exists())
        check(f"manifest size {p.name}", p.stat().st_size == int(r.bytes))
        check(f"manifest hash {p.name}", sha256(p) == r.sha256)

    selected = json.loads((OUT / "ASSISTments_Step3_Selected_Configuration.json").read_text(encoding="utf-8"))
    check("selected experiment registered", selected["experiment_id"] in expected_exps)
    summary_path = OUT / "ASSISTments_Step3_Final_Freeze_Summary.md"
    summary = summary_path.read_text(encoding="utf-8")
    summary = summary.replace("Independent validation: **pending execution of `validate_step3_artifacts.py`** (updated after validation)", "Independent validation: **PASS**")
    summary = summary.replace("Artifacts are ready for Step 4 only after the independent validator passes.", "Artifacts passed independent validation and are ready for the next recommendation-development step.")
    summary_path.write_text(summary, encoding="utf-8")

    report = pd.DataFrame(checks)
    report.to_csv(OUT / "ASSISTments_Step3_Independent_Validation.csv", index=False)
    (OUT / "ASSISTments_Step3_Independent_Validation.json").write_text(json.dumps({"status": "PASS", "checks": checks}, indent=2), encoding="utf-8")
    # Summary and new validation artifacts changed after the pipeline manifest; regenerate it last.
    rows = []
    for p in sorted(ROOT.rglob("*")):
        if p.is_file() and p.name != "ASSISTments_Step3_Artifact_Manifest.csv" and "__pycache__" not in p.parts:
            rows.append({"path": str(p.resolve()), "role": "Step-3 artifact", "bytes": p.stat().st_size, "sha256": sha256(p)})
    pd.DataFrame(rows).to_csv(manifest_path, index=False)
    print(f"PASS: {len(checks)} independent validation checks")


if __name__ == "__main__":
    main()
