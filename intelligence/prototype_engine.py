import numpy as np
from configs.settings import PROTOTYPE_K


class PrototypeEngine:
    def __init__(self, reference_X, reference_y, feature_names):
        self.reference_X = np.array(reference_X, dtype=float)
        self.reference_y = np.array(reference_y)
        self.feature_names = list(feature_names)
        self.scale = self.reference_X.std(axis=0) + 1e-9

    def retrieve(self, x, k=None):
        x = np.array(x, dtype=float)
        if k is None:
            k = PROTOTYPE_K

        diff = (self.reference_X - x) / self.scale
        dist = np.sqrt((diff * diff).sum(axis=1))
        idx = np.argsort(dist)[:k]

        out = []
        for i in idx:
            row = {}
            for name, value in zip(self.feature_names, self.reference_X[i]):
                row[name] = float(value)
            out.append(
                {
                    "index": int(i),
                    "distance": float(dist[i]),
                    "label": int(self.reference_y[i]),
                    "features": row,
                }
            )
        return out
