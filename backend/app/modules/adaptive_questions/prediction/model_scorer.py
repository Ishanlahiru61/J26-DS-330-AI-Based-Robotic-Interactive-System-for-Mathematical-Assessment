from __future__ import annotations

from pathlib import Path
from dataclasses import dataclass
import sys
import json
import hashlib
import importlib.util
import joblib
import numpy as np


RT_FIELDS = {"previous_ms_first_response", "previous_interaction_rt_valid", "prior_valid_rt_count", "prior_mean_ms_first_response"}

@dataclass
class RTStats:
    min_support: int
    global_median: float
    global_scale: float
    problem_type: dict
    prior_mean_median: float
    prior_mean_scale: float


class FrozenModelScorer:
    def __init__(self, model_path: str | Path, preprocessor_path: str | Path, calibration_path: str | Path):
        from xgboost import XGBClassifier
        main = sys.modules['__main__']
        previous = getattr(main, 'RTStats', None)
        main.RTStats = RTStats
        try:
            self.prep = joblib.load(preprocessor_path)
        finally:
            if previous is None:
                del main.RTStats
            else:
                main.RTStats = previous
        self.calibrator = joblib.load(calibration_path)
        self.model = XGBClassifier()
        self.model.load_model(model_path)
        source = np.asarray(self.prep["source_map"], dtype=object)
        self.selected_columns = np.flatnonzero(~np.isin(source, list(RT_FIELDS)))
        if len(self.selected_columns) != self.model.n_features_in_:
            raise ValueError('Frozen preprocessor/model feature widths disagree')
        self.raw_features = list(self.prep['preprocessor'].feature_names_in_)

    @classmethod
    def from_workspace(cls, workspace: str | Path):
        root = Path(workspace).resolve()
        manifest = json.loads((Path(__file__).resolve().parents[2] / 'artifacts/model_manifest.json').read_text())
        paths = []
        for artifact in manifest['external_workspace_relative_artifacts']:
            path = root / artifact['path']
            if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != artifact['sha256']:
                raise ValueError(f'Missing or changed frozen artifact: {path}')
            paths.append(path)
        scorer = cls(*paths)
        scorer.artifacts = manifest['external_workspace_relative_artifacts']
        return scorer

    def score_raw_research(self, rows, *, provenance: str):
        if provenance != 'FROZEN_ASSISTMENTS_HISTORY':
            raise ValueError('Unsupported input provenance; local game transfer is unavailable')
        missing = set(self.raw_features) - set(rows.columns)
        if missing:
            raise ValueError(f'Missing original features: {sorted(missing)}')
        # Reuse the preserved training-time engineering implementation; never fit.
        path = Path(__file__).resolve().parents[3] / 'research/assistments/step3/run_step3_pipeline.py'
        spec = importlib.util.spec_from_file_location('c2_frozen_engineering', path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        engineered = module.engineer(rows, self.prep['rt_stats'], self.prep['template_frequency'])
        matrix = self.prep['preprocessor'].transform(engineered[self.raw_features]).tocsr().astype(np.float32)
        return self.score_pretransformed(matrix[:, self.selected_columns])

    def score_pretransformed(self, matrix) -> np.ndarray:
        if matrix.shape[1] != len(self.selected_columns):
            raise ValueError("invalid input feature width")
        native = np.clip(self.model.predict_proba(matrix)[:, 1], 1e-7, 1 - 1e-7)
        logits = np.log(native / (1 - native)).reshape(-1, 1)
        result = self.calibrator.predict_proba(logits)[:, 1]
        if not np.all(np.isfinite(result)) or not np.all((0 <= result) & (result <= 1)):
            raise ValueError("invalid calibrated probability")
        return result
