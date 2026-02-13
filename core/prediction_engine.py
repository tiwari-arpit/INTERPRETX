import numpy as np


class PredictionEngine:
    def __init__(self, model):
        self.model = model

    def _positive_probability(self, X):
        if hasattr(self.model, "predict_proba"):
            probs = self.model.predict_proba(X)
            if probs.shape[1] == 1:
                return probs[:, 0]
            return probs[:, 1]

        if hasattr(self.model, "decision_function"):
            score = self.model.decision_function(X)
            return 1.0 / (1.0 + np.exp(-score))

        preds = self.model.predict(X)
        return np.clip(preds.astype(float), 0.0, 1.0)

    def predict(self, X):
        probs = self._positive_probability(X)
        labels = (probs >= 0.5).astype(int)
        confidence = np.maximum(probs, 1.0 - probs)

        records = []
        for i in range(len(labels)):
            records.append(
                {
                    "prediction": int(labels[i]),
                    "probability": float(probs[i]),
                    "confidence": float(confidence[i]),
                }
            )
        return records

    def predict_one(self, x):
        return self.predict(np.array([x]))[0]
