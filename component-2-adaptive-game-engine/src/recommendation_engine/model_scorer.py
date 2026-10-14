from __future__ import annotations

from pathlib import Path
import joblib
import numpy as np


RT_FIELDS = {"previous_ms_first_response", "previous_interaction_rt_valid", "prior_valid_rt_count", "prior_mean_ms_first_response"}


class FrozenModelScorer:
    def __init__(self, model_path: str | Path, preprocessor_path: str | Path, calibration_path: str | Path):
        from xgboost import XGBClassifier
        self.prep = joblib.load(preprocessor_path)
        self.calibrator = joblib.load(calibration_path)
        self.model = XGBClassifier()
        self.model.load_model(model_path)
        source = np.asarray(self.prep["source_map"], dtype=object)
        self.selected_columns = np.flatnonzero(~np.isin(source, list(RT_FIELDS)))

    def score_pretransformed(self, matrix) -> np.ndarray:
        if matrix.shape[1] != len(self.selected_columns):
            raise ValueError("invalid input feature width")
        native = np.clip(self.model.predict_proba(matrix)[:, 1], 1e-7, 1 - 1e-7)
        logits = np.log(native / (1 - native)).reshape(-1, 1)
        result = self.calibrator.predict_proba(logits)[:, 1]
        if not np.all(np.isfinite(result)) or not np.all((0 <= result) & (result <= 1)):
            raise ValueError("invalid calibrated probability")
        return result
