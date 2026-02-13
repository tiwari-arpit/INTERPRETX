import numpy as np
from core.prediction_engine import PredictionEngine
from configs.settings import PERTURBATION_STEPS, NOISE_SCALE, BORDERLINE_MARGIN


class RobustnessEngine:
    def __init__(self, model, feature_scales):
        self.model = model
        self.feature_scales = np.array(feature_scales)
        self.prediction_engine = PredictionEngine(model)

    def evaluate(self, x):
        x = np.array(x, dtype=float)
        base = self.prediction_engine.predict_one(x)

        flips = 0
        probs = []
        for _ in range(PERTURBATION_STEPS):
            noise = np.random.normal(0.0, 1.0, size=len(x))
            perturb = x + noise * self.feature_scales * NOISE_SCALE
            pred = self.prediction_engine.predict_one(perturb)
            probs.append(pred["probability"])
            if pred["prediction"] != base["prediction"]:
                flips += 1

        probs = np.array(probs)
        flip_rate = flips / float(PERTURBATION_STEPS)
        mean_prob = float(np.mean(probs))
        prob_std = float(np.std(probs))
        borderline = abs(base["probability"] - 0.5) <= BORDERLINE_MARGIN or flip_rate > 0.2

        return {
            "flip_rate": float(flip_rate),
            "perturbed_probability_mean": mean_prob,
            "perturbed_probability_std": prob_std,
            "borderline_or_unstable": bool(borderline),
        }
