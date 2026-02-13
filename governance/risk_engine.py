from configs.settings import TRUST_THRESHOLD, REVIEW_THRESHOLD


class RiskEngine:
    def evaluate(self, prediction, robustness, uncertainty, drift, fairness, local_causal):
        confidence_score = prediction["confidence"]
        stability_score = 1.0 - robustness["flip_rate"]
        uncertainty_penalty = min(1.0, uncertainty["ambiguity"])
        drift_penalty = min(1.0, drift["drift_score"] / 3.0)
        fairness_penalty = 1.0 if fairness["bias_flag"] else 0.0
        causal_sensitivity_penalty = min(1.0, abs(local_causal["effect"]))

        score = (
            0.38 * confidence_score
            + 0.27 * stability_score
            + 0.15 * (1.0 - uncertainty_penalty)
            + 0.10 * (1.0 - drift_penalty)
            + 0.05 * (1.0 - fairness_penalty)
            + 0.05 * (1.0 - causal_sensitivity_penalty)
        )

        if score >= TRUST_THRESHOLD:
            decision = "TRUST"
        elif score >= REVIEW_THRESHOLD:
            decision = "REVIEW"
        else:
            decision = "ESCALATE"

        reasons = []
        if robustness["flip_rate"] > 0.2:
            reasons.append("High prediction instability under small perturbations")
        if uncertainty["ood_or_ambiguous"]:
            reasons.append("High ambiguity or possible out-of-distribution input")
        if fairness["bias_flag"]:
            reasons.append("Fairness gap beyond policy threshold")
        if drift["drift_flag"]:
            reasons.append("Input drift detected against training profile")
        if abs(local_causal["effect"]) > 0.2:
            reasons.append("Decision is highly sensitive to treatment intervention")

        return {
            "risk_score": float(score),
            "decision": decision,
            "reasons": reasons,
        }
