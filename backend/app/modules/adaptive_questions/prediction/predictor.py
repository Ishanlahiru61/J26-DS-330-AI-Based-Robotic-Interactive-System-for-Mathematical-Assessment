"""Replaceable prediction layer for the live assessment service."""
from __future__ import annotations

import json
import hashlib
import math
from pathlib import Path

from .local_state_model import estimate_skill_success, load_adaptive_settings


ARTIFACT_DIR = Path(__file__).resolve().parents[1] / "models" / "transferable"


class TransferableTrainedPredictor:
    """Load the frozen numeric-only deployment model and score skill contexts."""

    def __init__(self, artifact_dir: str | Path = ARTIFACT_DIR):
        self.artifact_dir = Path(artifact_dir)
        self.model = None
        self.calibrator = None
        self.metadata = None
        self.load_error = None
        try:
            import joblib
            from xgboost import XGBClassifier

            manifest = json.loads((self.artifact_dir / "manifest.json").read_text(encoding="utf-8"))
            for filename, expected in manifest["files"].items():
                path = self.artifact_dir / filename
                actual = hashlib.sha256(path.read_bytes()).hexdigest()
                if path.stat().st_size != expected["bytes"] or actual != expected["sha256"]:
                    raise ValueError(f"artifact integrity check failed: {filename}")
            self.metadata = json.loads((self.artifact_dir / "metadata.json").read_text(encoding="utf-8"))
            self.model = XGBClassifier()
            self.model.load_model(self.artifact_dir / "model.json")
            if self.metadata.get("calibration_applied_at_runtime"):
                self.calibrator = joblib.load(self.artifact_dir / "calibrator.joblib")
        except Exception as exc:
            self.load_error = f"ARTIFACT_LOAD_FAILED:{type(exc).__name__}"

    @staticmethod
    def _recent_values(outcomes: tuple[int, ...], width: int) -> tuple[int, int, float | None]:
        values = list(outcomes[-width:])
        return len(values), sum(values), (sum(values) / len(values) if values else None)

    def _features(self, state, skill: str) -> dict:
        recent = tuple(state.recent_outcomes[-5:])
        r3n, r3c, r3rate = self._recent_values(recent, 3)
        r5n, r5c, r5rate = self._recent_values(recent, 5)
        skill_outcomes = tuple(state.recent_skill_outcomes_by_skill.get(skill, ()))
        skill_n = int(state.prior_skill_exposures.get(skill, 0))
        skill_correct = int(state.prior_skill_correct.get(skill, 0))
        prior = int(state.prior_interaction_count)
        correct_total = int(state.prior_correct_count)
        return {
            "previous_correct_raw": recent[-1] if recent else None,
            "previous_attempt_count": 1 if recent else None,
            "has_strict_prior_interaction": int(prior > 0),
            "prior_interaction_count": prior,
            "prior_target_eligible_count": prior,
            "prior_full_correct_count": correct_total,
            "prior_incorrect_count": prior - correct_total,
            "prior_correct_rate": correct_total / prior if prior else None,
            "prior_mean_attempt_count": 1.0 if prior else None,
            "prior_skill_exposure_count": skill_n,
            "prior_skill_target_eligible_count": skill_n,
            "prior_skill_full_correct_count": skill_correct,
            "prior_skill_correct_rate": skill_correct / skill_n if skill_n else None,
            "lag2_target_first_attempt_correct": recent[-2] if len(recent) >= 2 else None,
            "lag3_target_first_attempt_correct": recent[-3] if len(recent) >= 3 else None,
            "lag4_target_first_attempt_correct": recent[-4] if len(recent) >= 4 else None,
            "lag5_target_first_attempt_correct": recent[-5] if len(recent) >= 5 else None,
            "recent3_target_eligible_count": r3n,
            "recent3_full_correct_count": r3c,
            "recent3_correct_rate": r3rate,
            "recent5_target_eligible_count": r5n,
            "recent5_full_correct_count": r5c,
            "recent5_correct_rate": r5rate,
            "interactions_since_last_full_correct": state.interactions_since_last_correct,
            "previous_same_skill_correct_raw": skill_outcomes[-1] if skill_outcomes else None,
        }

    def predict(self, state, skill: str) -> dict:
        version = self.metadata.get("model_version") if self.metadata else "TRANSFER_H2_GB_RT_OFF_v1"
        base = {
            "source": "TRAINED_ML", "model_version": version,
            "target": "next first-attempt correctness under transferable ASSISTments feature semantics",
            "skill": skill, "success_probability": None, "valid": False,
            "uses_strict_prior_history": True,
            "domain": "ASSISTments-trained; local cube-game deployment",
            "validation_status": "HELD_OUT_ASSISTMENTS_ONLY_NOT_LOCAL_CHILD_VALIDATION",
            "assumptions": [
                "local correctness and skill-history definitions match the frozen training features",
                "cross-domain calibration is unverified",
                "tier difficulty is not a model input",
            ],
            "abstention_reason": None,
        }
        if self.load_error:
            return {**base, "abstention_reason": self.load_error}
        if state.prior_interaction_count < 1:
            return {**base, "abstention_reason": "COLD_START_NO_PRIOR_INTERACTION"}
        if not state.timestamp_valid:
            return {**base, "abstention_reason": "TIMESTAMP_HISTORY_ANOMALY"}
        try:
            import numpy as np
            import pandas as pd

            features = self._features(state, skill)
            ordered = self.metadata["feature_order"]
            frame = pd.DataFrame([[features[name] for name in ordered]], columns=ordered, dtype=float)
            probability = float(self.model.predict_proba(frame)[0, 1])
            if self.calibrator is not None:
                clipped = min(max(probability, 1e-7), 1 - 1e-7)
                probability = float(self.calibrator.predict_proba(np.array([[math.log(clipped / (1 - clipped))]]))[0, 1])
            if not math.isfinite(probability) or not 0 <= probability <= 1:
                raise ValueError("invalid model probability")
            return {**base, "success_probability": probability, "valid": True,
                    "feature_values": features, "abstention_reason": None}
        except Exception as exc:
            return {**base, "abstention_reason": f"INFERENCE_FAILED:{type(exc).__name__}"}


class PredictorRouter:
    """Trained predictor first; local Beta-Bernoulli only on abstention."""

    def __init__(self, trained: TransferableTrainedPredictor | None = None):
        self.trained = trained or TransferableTrainedPredictor()

    def predict(self, state, skill: str) -> dict:
        trained = self.trained.predict(state, skill)
        if trained["valid"]:
            return trained
        fallback = estimate_skill_success(state, skill, load_adaptive_settings())
        fallback.update({
            "source": "BETA_BERNOULLI_FALLBACK",
            "valid": fallback["success_probability"] is not None,
            "primary_abstention_reason": trained["abstention_reason"],
            "validation_status": "EXPERIMENTAL_UNVALIDATED_FOR_REAL_CHILDREN",
        })
        if fallback["success_probability"] is None:
            fallback["source"] = "ABSTAIN"
        return fallback

    def predict_many(self, state, skills: list[str]) -> dict[str, dict]:
        return {skill: self.predict(state, skill) for skill in dict.fromkeys(skills)}
