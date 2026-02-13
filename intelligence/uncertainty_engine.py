import numpy as np
from configs.settings import OOD_ZSCORE_THRESHOLD


class UncertaintyEngine:
    def __init__(self):
        self.mean = None
        self.std = None

    def fit(self, X):
        X = np.array(X, dtype=float)
        self.mean = X.mean(axis=0)
        self.std = X.std(axis=0) + 1e-9
        return self

    def evaluate(self, probability, x):
        p = float(np.clip(probability, 1e-6, 1.0 - 1e-6))
        entropy = -(p * np.log2(p) + (1.0 - p) * np.log2(1.0 - p))
        ambiguity = 1.0 - abs(2.0 * p - 1.0)

        x = np.array(x, dtype=float)
        z = np.abs((x - self.mean) / self.std)
        z_mean = float(np.mean(z))
        z_max = float(np.max(z))
        ood = z_mean > OOD_ZSCORE_THRESHOLD or z_max > (OOD_ZSCORE_THRESHOLD * 2.0)

        confidence_band = "high"
        if p < 0.75 and p > 0.25:
            confidence_band = "low"
        elif p < 0.9 and p > 0.1:
            confidence_band = "medium"

        return {
            "entropy": float(entropy),
            "ambiguity": float(ambiguity),
            "mean_zscore": z_mean,
            "max_zscore": z_max,
            "ood_or_ambiguous": bool(ood or ambiguity > 0.7),
            "confidence_band": confidence_band,
        }
