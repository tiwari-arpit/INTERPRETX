import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from core.prediction_engine import PredictionEngine


class CausalEngine:
    def __init__(self, feature_names, treatment_feature, confounders):
        self.feature_names = list(feature_names)
        self.treatment_feature = treatment_feature
        self.confounders = list(confounders)

        self.treatment_idx = self.feature_names.index(self.treatment_feature)
        self.fit_columns = [self.treatment_feature] + self.confounders

        self.model = LogisticRegression(max_iter=2000)
        self.prediction_engine = PredictionEngine(self.model)

    def _to_frame(self, X):
        return pd.DataFrame(X, columns=self.feature_names)

    def fit(self, X, y):
        df = self._to_frame(X)
        self.model.fit(df[self.fit_columns].values, y)
        return self

    def _predict_prob_from_df(self, df):
        X_use = df[self.fit_columns].values
        return self.model.predict_proba(X_use)[:, 1]

    def global_ate(self, X, low_q=0.2, high_q=0.8):
        df = self._to_frame(X)
        low = float(df[self.treatment_feature].quantile(low_q))
        high = float(df[self.treatment_feature].quantile(high_q))

        low_df = df.copy()
        high_df = df.copy()
        low_df[self.treatment_feature] = low
        high_df[self.treatment_feature] = high

        p_low = self._predict_prob_from_df(low_df).mean()
        p_high = self._predict_prob_from_df(high_df).mean()

        return {
            "treatment_feature": self.treatment_feature,
            "intervention_low": low,
            "intervention_high": high,
            "average_treatment_effect": float(p_high - p_low),
            "mean_outcome_low": float(p_low),
            "mean_outcome_high": float(p_high),
        }

    def intervention_effect(self, x, delta):
        x = np.array(x, dtype=float)
        before = self.prediction_engine.predict_one(
            np.concatenate(([x[self.treatment_idx]], x[[self.feature_names.index(c) for c in self.confounders]]))
        )

        x_new = x.copy()
        x_new[self.treatment_idx] = x_new[self.treatment_idx] + delta

        after = self.prediction_engine.predict_one(
            np.concatenate(([x_new[self.treatment_idx]], x_new[[self.feature_names.index(c) for c in self.confounders]]))
        )

        return {
            "treatment_feature": self.treatment_feature,
            "delta": float(delta),
            "probability_before": float(before["probability"]),
            "probability_after": float(after["probability"]),
            "effect": float(after["probability"] - before["probability"]),
        }
