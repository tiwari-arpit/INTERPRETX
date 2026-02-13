import numpy as np
from configs.settings import DRIFT_ZSCORE_THRESHOLD


class DriftDetector:
    def __init__(self):
        self.mean = None
        self.std = None

    def fit(self, reference_X):
        reference_X = np.array(reference_X, dtype=float)
        self.mean = reference_X.mean(axis=0)
        self.std = reference_X.std(axis=0) + 1e-9
        return self

    def check(self, x):
        x = np.array(x, dtype=float)
        z = np.abs((x - self.mean) / self.std)
        score = float(np.mean(z))
        return {
            "drift_score": score,
            "max_feature_zscore": float(np.max(z)),
            "drift_flag": bool(score > DRIFT_ZSCORE_THRESHOLD),
        }
