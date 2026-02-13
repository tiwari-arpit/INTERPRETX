import json
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score

from configs.settings import RANDOM_SEED, TEST_SIZE, COUNTERFACTUAL_STEP_SCALE
from core.model_register import ModelRegistry
from core.prediction_engine import PredictionEngine
from intelligence.counterfactual_engine import CounterfactualEngine
from intelligence.robustness_engine import RobustnessEngine
from intelligence.uncertainty_engine import UncertaintyEngine
from intelligence.prototype_engine import PrototypeEngine
from intelligence.fairness_engine import FairnessEngine
from intelligence.causal_engine import CausalEngine
from monitoring.drift_checks import DriftDetector
from governance.risk_engine import RiskEngine


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def generate_credit_data(n=2500, seed=RANDOM_SEED):
    rng = np.random.default_rng(seed)

    age = rng.integers(21, 70, n)
    annual_income_k = rng.normal(72, 26, n).clip(18, 220)
    debt_to_income = rng.normal(36, 15, n).clip(2, 90)
    utilization_ratio = rng.beta(2.2, 2.6, n).clip(0.01, 0.99)
    late_payments_12m = rng.poisson(1.2, n).clip(0, 12)
    loan_amount_k = rng.normal(26, 12, n).clip(2, 110)
    employment_years = rng.normal(6, 4, n).clip(0, 35)
    savings_k = rng.normal(16, 14, n).clip(0, 120)
    previous_defaults = rng.binomial(1, 0.14, n)
    gender = rng.binomial(1, 0.48, n)

    logit = (
        -3.0
        + 0.032 * (debt_to_income - 30)
        + 2.1 * utilization_ratio
        + 0.32 * late_payments_12m
        + 0.8 * previous_defaults
        + 0.018 * loan_amount_k
        - 0.017 * annual_income_k
        - 0.04 * employment_years
        - 0.012 * savings_k
        + 0.14 * gender
        + rng.normal(0, 0.35, n)
    )

    prob_default = sigmoid(logit)
    y = rng.binomial(1, prob_default)

    feature_names = [
        "age",
        "annual_income_k",
        "debt_to_income",
        "utilization_ratio",
        "late_payments_12m",
        "loan_amount_k",
        "employment_years",
        "savings_k",
        "previous_defaults",
        "gender",
    ]

    X = np.column_stack(
        [
            age,
            annual_income_k,
            debt_to_income,
            utilization_ratio,
            late_payments_12m,
            loan_amount_k,
            employment_years,
            savings_k,
            previous_defaults,
            gender,
        ]
    )

    return {
        "name": "credit_risk_monitoring",
        "X": X,
        "y": y,
        "feature_names": feature_names,
        "sensitive_feature": "gender",
        "treatment_feature": "utilization_ratio",
        "intervention_delta": -0.08,
    }


def generate_diabetes_data(n=3000, seed=RANDOM_SEED + 17):
    rng = np.random.default_rng(seed)

    age = rng.integers(18, 80, n)
    bmi = rng.normal(28, 5.5, n).clip(16, 50)
    glucose_mgdl = rng.normal(145, 40, n).clip(70, 360)
    carb_intake_g = rng.normal(62, 24, n).clip(10, 220)
    activity_minutes = rng.normal(38, 22, n).clip(0, 180)
    current_insulin_units = rng.normal(18, 9, n).clip(0, 80)
    sleep_hours = rng.normal(6.7, 1.3, n).clip(3, 10)
    stress_index = rng.normal(5.0, 2.2, n).clip(0, 10)
    renal_risk = rng.binomial(1, 0.16, n)
    sex = rng.binomial(1, 0.5, n)

    logit = (
        -4.8
        + 0.024 * (glucose_mgdl - 110)
        + 0.018 * (carb_intake_g - 50)
        - 0.03 * activity_minutes
        - 0.05 * current_insulin_units
        + 0.065 * (bmi - 27)
        + 0.22 * renal_risk
        + 0.11 * stress_index
        - 0.09 * (sleep_hours - 7)
        + 0.08 * sex
        + rng.normal(0, 0.42, n)
    )

    prob_need_dose_increase = sigmoid(logit)
    y = rng.binomial(1, prob_need_dose_increase)

    feature_names = [
        "age",
        "bmi",
        "glucose_mgdl",
        "carb_intake_g",
        "activity_minutes",
        "current_insulin_units",
        "sleep_hours",
        "stress_index",
        "renal_risk",
        "sex",
    ]

    X = np.column_stack(
        [
            age,
            bmi,
            glucose_mgdl,
            carb_intake_g,
            activity_minutes,
            current_insulin_units,
            sleep_hours,
            stress_index,
            renal_risk,
            sex,
        ]
    )

    return {
        "name": "diabetes_insulin_dose_monitoring",
        "X": X,
        "y": y,
        "feature_names": feature_names,
        "sensitive_feature": "sex",
        "treatment_feature": "current_insulin_units",
        "intervention_delta": 3.0,
    }


def get_bounds_and_steps(X):
    mins = X.min(axis=0)
    maxs = X.max(axis=0)
    stds = X.std(axis=0) + 1e-9

    bounds = []
    for low, high in zip(mins, maxs):
        bounds.append((float(low), float(high)))

    step_scales = stds * COUNTERFACTUAL_STEP_SCALE
    return bounds, step_scales, stds


def linear_feature_contributions(model, x, feature_names):
    if not hasattr(model, "coef_"):
        return {}

    coef = model.coef_[0]
    contrib = coef * x
    ordered = np.argsort(np.abs(contrib))[::-1]

    top = {}
    for idx in ordered[:6]:
        top[feature_names[idx]] = float(contrib[idx])
    return top


def run_use_case(bundle):
    X = bundle["X"]
    y = bundle["y"]
    feature_names = bundle["feature_names"]
    sensitive_feature = bundle["sensitive_feature"]
    treatment_feature = bundle["treatment_feature"]
    intervention_delta = bundle["intervention_delta"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_SEED,
        stratify=y,
    )

    model = LogisticRegression(max_iter=3000, class_weight="balanced")
    model.fit(X_train, y_train)

    registry = ModelRegistry()
    registry.register(bundle["name"], model)

    bounds, step_scales, feature_scales = get_bounds_and_steps(X_train)

    prediction_engine = PredictionEngine(model)
    counterfactual_engine = CounterfactualEngine(model, feature_names, bounds, step_scales)
    robustness_engine = RobustnessEngine(model, feature_scales)
    uncertainty_engine = UncertaintyEngine().fit(X_train)
    prototype_engine = PrototypeEngine(X_train, y_train, feature_names)
    fairness_engine = FairnessEngine()

    confounders = [f for f in feature_names if f != treatment_feature]
    causal_engine = CausalEngine(feature_names, treatment_feature, confounders)
    causal_engine.fit(X_train, y_train)

    drift_detector = DriftDetector().fit(X_train)
    risk_engine = RiskEngine()

    test_probs = prediction_engine.predict(X_test)
    y_pred = np.array([p["prediction"] for p in test_probs])
    y_prob = np.array([p["probability"] for p in test_probs])

    accuracy = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)

    sample_index = 0
    x = X_test[sample_index]

    pred = prediction_engine.predict_one(x)
    counter = counterfactual_engine.generate(x)
    robust = robustness_engine.evaluate(x)
    uncertainty = uncertainty_engine.evaluate(pred["probability"], x)
    prototypes = prototype_engine.retrieve(x)

    sensitive_idx = feature_names.index(sensitive_feature)
    fairness = fairness_engine.evaluate(y_test, y_pred, X_test[:, sensitive_idx])

    causal_global = causal_engine.global_ate(X_test)
    causal_local = causal_engine.intervention_effect(x, intervention_delta)

    drift = drift_detector.check(x)
    governance = risk_engine.evaluate(pred, robust, uncertainty, drift, fairness, causal_local)

    local_explanation = linear_feature_contributions(model, x, feature_names)

    sample_features = {}
    for name, value in zip(feature_names, x):
        sample_features[name] = float(value)

    result = {
        "use_case": bundle["name"],
        "metrics": {
            "accuracy": float(accuracy),
            "roc_auc": float(auc),
        },
        "sample_input": sample_features,
        "prediction_engine": pred,
        "counterfactual_engine": counter,
        "stability_robustness_engine": robust,
        "uncertainty_quantification_engine": uncertainty,
        "prototype_similarity_engine": prototypes,
        "fairness_bias_engine": fairness,
        "causal_reasoning_engine": {
            "global": causal_global,
            "local": causal_local,
        },
        "drift_monitoring": drift,
        "governance_risk_scoring_engine": governance,
        "linear_feature_contribution_explanation": local_explanation,
    }

    return result


def main():
    np.random.seed(RANDOM_SEED)

    credit_bundle = generate_credit_data()
    diabetes_bundle = generate_diabetes_data()

    credit_report = run_use_case(credit_bundle)
    diabetes_report = run_use_case(diabetes_bundle)

    final_output = {
        "summary": "Explainable AI pipeline for credit risk monitoring and diabetes-insulin dosage monitoring/prediction",
        "reports": [credit_report, diabetes_report],
    }

    print(json.dumps(final_output, indent=2))


if __name__ == "__main__":
    main()
