import numpy as np
from core.prediction_engine import PredictionEngine
from configs.settings import COUNTERFACTUAL_MAX_ITERS


class CounterfactualEngine:
    def __init__(self, model, feature_names, feature_bounds, step_scales):
        self.model = model
        self.feature_names = list(feature_names)
        self.feature_bounds = feature_bounds
        self.step_scales = np.array(step_scales)
        self.prediction_engine = PredictionEngine(model)

    def _clip(self, x):
        out = x.copy()
        for i, (low, high) in enumerate(self.feature_bounds):
            out[i] = np.clip(out[i], low, high)
        return out

    def _score(self, x, target_class):
        p = self.prediction_engine.predict_one(x)["probability"]
        return p if target_class == 1 else 1.0 - p

    def generate(self, x):
        x = np.array(x, dtype=float)
        base = self.prediction_engine.predict_one(x)
        target_class = 1 - base["prediction"]

        current = x.copy()
        changed = {}

        for _ in range(COUNTERFACTUAL_MAX_ITERS):
            now_pred = self.prediction_engine.predict_one(current)
            if now_pred["prediction"] == target_class:
                edits = []
                for idx, (from_v, to_v) in changed.items():
                    edits.append(
                        {
                            "feature": self.feature_names[idx],
                            "from": float(from_v),
                            "to": float(to_v),
                            "delta": float(to_v - from_v),
                        }
                    )
                edits = sorted(edits, key=lambda d: abs(d["delta"]))
                return {
                    "found": True,
                    "target_class": int(target_class),
                    "steps": len(changed),
                    "counterfactual_prediction": now_pred,
                    "edits": edits,
                }

            current_score = self._score(current, target_class)
            best = None

            for i in range(len(current)):
                step = self.step_scales[i]
                if step == 0:
                    continue

                for direction in (-1.0, 1.0):
                    candidate = current.copy()
                    candidate[i] += direction * step
                    candidate = self._clip(candidate)
                    gain = self._score(candidate, target_class) - current_score
                    move_cost = abs(candidate[i] - current[i]) / (abs(self.step_scales[i]) + 1e-9)

                    if move_cost == 0:
                        continue

                    value = gain / move_cost
                    if best is None or value > best["value"]:
                        best = {
                            "value": value,
                            "candidate": candidate,
                            "index": i,
                        }

            if best is None or best["value"] <= 0:
                break

            i = best["index"]
            next_x = best["candidate"]
            if i not in changed:
                changed[i] = (x[i], next_x[i])
            else:
                changed[i] = (changed[i][0], next_x[i])
            current = next_x

        return {
            "found": False,
            "target_class": int(target_class),
            "steps": 0,
            "reason": "No feasible low-cost flip found within search budget",
        }
