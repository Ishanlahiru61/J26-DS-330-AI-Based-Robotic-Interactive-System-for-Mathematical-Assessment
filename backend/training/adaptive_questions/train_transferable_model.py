#!/usr/bin/env python3
"""Train the deployment predictor from frozen, strictly-prior ASSISTments features.

The script reads but never modifies the external research workspace.  It uses
the original learner-disjoint partitions and excludes every identifier,
platform-context field, response-time field and current-response field.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import duckdb
import joblib
import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from xgboost import XGBClassifier


MODEL_VERSION = "TRANSFER_H2_GB_RT_OFF_v1"
FEATURES = [
    "previous_correct_raw",
    "previous_attempt_count",
    "has_strict_prior_interaction",
    "prior_interaction_count",
    "prior_target_eligible_count",
    "prior_full_correct_count",
    "prior_incorrect_count",
    "prior_correct_rate",
    "prior_mean_attempt_count",
    "prior_skill_exposure_count",
    "prior_skill_target_eligible_count",
    "prior_skill_full_correct_count",
    "prior_skill_correct_rate",
    "lag2_target_first_attempt_correct",
    "lag3_target_first_attempt_correct",
    "lag4_target_first_attempt_correct",
    "lag5_target_first_attempt_correct",
    "recent3_target_eligible_count",
    "recent3_full_correct_count",
    "recent3_correct_rate",
    "recent5_target_eligible_count",
    "recent5_full_correct_count",
    "recent5_correct_rate",
    "interactions_since_last_full_correct",
    "previous_same_skill_correct_raw",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def metrics(y: np.ndarray, probability: np.ndarray) -> dict:
    return {
        "roc_auc": float(roc_auc_score(y, probability)),
        "log_loss": float(log_loss(y, probability)),
        "brier_score": float(brier_score_loss(y, probability)),
        "n_rows": int(len(y)),
    }


def load_split(con, parquet: Path, partitions: Path, split: str, limit: int | None) -> pd.DataFrame:
    learner_file = partitions / f"{split}_learners.csv"
    fields = ",".join(f'h."{name}"' for name in FEATURES)
    limiter = ""
    if limit:
        # Sampling occurs only after the frozen learner split.  The stable hash
        # makes repeated training reproducible without crossing partitions.
        limiter = f"QUALIFY row_number() OVER (ORDER BY hash(h.problem_log_id, 20261017)) <= {int(limit)}"
    query = f"""
      SELECT h.problem_log_id,h.target_first_attempt_correct AS target,{fields}
      FROM read_parquet(?) h
      JOIN read_csv_auto(?) p USING(user_id)
      WHERE h.target_eligible
      {limiter}
    """
    return con.execute(query, [str(parquet), str(learner_file)]).df()


def beta_probabilities(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    attempts = frame["prior_skill_target_eligible_count"].fillna(0).to_numpy(float)
    correct = frame["prior_skill_full_correct_count"].fillna(0).to_numpy(float)
    supported = attempts >= 2
    probability = (correct[supported] + 1.0) / (attempts[supported] + 2.0)
    return supported, probability


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", required=True)
    parser.add_argument("--output", default=str(Path(__file__).resolve().parents[1] / "artifacts" / "transferable"))
    parser.add_argument("--train-limit", type=int, default=1_200_000)
    parser.add_argument("--validation-limit", type=int, default=300_000)
    args = parser.parse_args()

    workspace = Path(args.workspace).resolve()
    parquet = workspace / "02_Chronological_Learner_History/data/processed/ASSISTments_2012_2013_history_features.parquet"
    partitions = workspace / "03_Experimental_Modeling/data/partitions"
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    if not parquet.is_file() or not partitions.is_dir():
        raise FileNotFoundError("Frozen history parquet or learner partitions are unavailable")

    with duckdb.connect() as con:
        train = load_split(con, parquet, partitions, "train", args.train_limit)
        validation = load_split(con, parquet, partitions, "validation", args.validation_limit)
        test = load_split(con, parquet, partitions, "test", None)

    x_train = train[FEATURES].astype(float)
    y_train = train["target"].to_numpy(int)
    x_validation = validation[FEATURES].astype(float)
    y_validation = validation["target"].to_numpy(int)
    x_test = test[FEATURES].astype(float)
    y_test = test["target"].to_numpy(int)

    model = XGBClassifier(
        n_estimators=350,
        learning_rate=0.06,
        max_depth=5,
        min_child_weight=20,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=2.0,
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        random_state=20261017,
        n_jobs=6,
        early_stopping_rounds=20,
    )
    model.fit(x_train, y_train, eval_set=[(x_validation, y_validation)], verbose=False)
    raw_validation = model.predict_proba(x_validation)[:, 1]
    raw_test = model.predict_proba(x_test)[:, 1]

    eps = 1e-7
    validation_logit = np.log(np.clip(raw_validation, eps, 1 - eps) / np.clip(1 - raw_validation, eps, 1 - eps)).reshape(-1, 1)
    calibrator = LogisticRegression(C=1e6, solver="lbfgs", random_state=20261017)
    calibrator.fit(validation_logit, y_validation)
    test_logit = np.log(np.clip(raw_test, eps, 1 - eps) / np.clip(1 - raw_test, eps, 1 - eps)).reshape(-1, 1)
    calibrated_test = calibrator.predict_proba(test_logit)[:, 1]

    raw_val_metrics = metrics(y_validation, raw_validation)
    raw_test_metrics = metrics(y_test, raw_test)
    calibrated_test_metrics = metrics(y_test, calibrated_test)
    retain_calibration = (
        log_loss(y_validation, calibrator.predict_proba(validation_logit)[:, 1])
        <= raw_val_metrics["log_loss"] - 0.0005
    )

    beta_support, beta_probability = beta_probabilities(test)
    beta_metrics = metrics(y_test[beta_support], beta_probability)
    beta_metrics["support_fraction"] = float(beta_support.mean())

    model_path = output / "model.json"
    calibrator_path = output / "calibrator.joblib"
    metadata_path = output / "metadata.json"
    metrics_path = output / "evaluation.json"
    model.save_model(model_path)
    joblib.dump(calibrator, calibrator_path)
    evaluation = {
        "model_version": MODEL_VERSION,
        "validation_native": raw_val_metrics,
        "test_native": raw_test_metrics,
        "test_platt": calibrated_test_metrics,
        "calibration_retained": bool(retain_calibration),
        "beta_bernoulli_test_supported_rows": beta_metrics,
        "research_winner_test_platt_reference": {
            "model_version": "H2_GB_RT_OFF_v1",
            "roc_auc": 0.7501579955024732,
            "log_loss": 0.5344418770242123,
            "brier_score": 0.17895472334084186,
            "source": "ASSISTments_Step3_Model_Results.csv",
        },
    }
    metadata = {
        "model_version": MODEL_VERSION,
        "algorithm": "XGBoost binary classifier",
        "target": "observed next first-attempt correctness",
        "feature_order": FEATURES,
        "current_response_features": [],
        "identifier_features": [],
        "response_time_features": [],
        "candidate_specific_features": [
            "prior_skill_exposure_count",
            "prior_skill_target_eligible_count",
            "prior_skill_full_correct_count",
            "prior_skill_correct_rate",
            "previous_same_skill_correct_raw",
        ],
        "training_partition": "original frozen train_learners.csv; stable within-partition row sample",
        "validation_partition": "original frozen validation_learners.csv; stable within-partition row sample",
        "test_partition": "complete original frozen test_learners.csv",
        "train_rows": int(len(train)),
        "validation_rows": int(len(validation)),
        "test_rows": int(len(test)),
        "random_seed": 20261017,
        "calibration_applied_at_runtime": bool(retain_calibration),
        "transfer_boundary": "Associational evidence trained on ASSISTments; not validated or calibrated on local cube-game children.",
        "source_parquet_sha256": sha256(parquet),
    }
    metrics_path.write_text(json.dumps(evaluation, indent=2), encoding="utf-8")
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    manifest = {
        "model_version": MODEL_VERSION,
        "files": {
            name.name: {"bytes": name.stat().st_size, "sha256": sha256(name)}
            for name in (model_path, calibrator_path, metadata_path, metrics_path)
        },
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"metadata": metadata, "evaluation": evaluation, "manifest": manifest}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
