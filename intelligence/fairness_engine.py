import numpy as np
from configs.settings import BIAS_GAP_THRESHOLD


class FairnessEngine:
    def __init__(self, positive_label=1):
        self.positive_label = positive_label

    def _safe_rate(self, num, den):
        return float(num) / float(den) if den > 0 else 0.0

    def evaluate(self, y_true, y_pred, sensitive_feature):
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        s = np.array(sensitive_feature)

        groups = sorted(list(np.unique(s)))
        group_metrics = {}

        for g in groups:
            idx = s == g
            yt = y_true[idx]
            yp = y_pred[idx]

            tp = np.sum((yt == self.positive_label) & (yp == self.positive_label))
            fp = np.sum((yt != self.positive_label) & (yp == self.positive_label))
            tn = np.sum((yt != self.positive_label) & (yp != self.positive_label))
            fn = np.sum((yt == self.positive_label) & (yp != self.positive_label))

            pos_rate = self._safe_rate(np.sum(yp == self.positive_label), len(yp))
            tpr = self._safe_rate(tp, tp + fn)
            fpr = self._safe_rate(fp, fp + tn)

            group_metrics[str(g)] = {
                "count": int(len(yp)),
                "positive_rate": pos_rate,
                "tpr": tpr,
                "fpr": fpr,
            }

        pos_rates = [group_metrics[str(g)]["positive_rate"] for g in groups]
        tprs = [group_metrics[str(g)]["tpr"] for g in groups]
        fprs = [group_metrics[str(g)]["fpr"] for g in groups]

        dp_gap = float(max(pos_rates) - min(pos_rates)) if len(pos_rates) > 1 else 0.0
        eo_tpr_gap = float(max(tprs) - min(tprs)) if len(tprs) > 1 else 0.0
        eo_fpr_gap = float(max(fprs) - min(fprs)) if len(fprs) > 1 else 0.0
        eo_gap = max(eo_tpr_gap, eo_fpr_gap)

        bias_flag = dp_gap > BIAS_GAP_THRESHOLD or eo_gap > BIAS_GAP_THRESHOLD

        return {
            "by_group": group_metrics,
            "demographic_parity_gap": dp_gap,
            "equalized_odds_gap": float(eo_gap),
            "bias_flag": bool(bias_flag),
        }
