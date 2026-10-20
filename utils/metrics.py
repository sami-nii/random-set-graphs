"""Evaluation metrics of the paper (App. "Evaluation metrics").

OOD detection treats OOD nodes as the positive class and a higher score as
"more uncertain": AUROC, AUPRC and FPR at 95% TPR. Calibration on ID nodes:
ECE (15 bins), NLL and Brier score.
"""

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve

from utils.isotonic_calibration import expected_calibration_error

EPS = 1e-12


def entropy(probabilities):
    probabilities = np.clip(np.asarray(probabilities, dtype=np.float64), EPS, 1.0)
    return -(probabilities * np.log(probabilities)).sum(axis=1)


def fpr_at_95_tpr(is_ood, scores):
    fpr, tpr, _ = roc_curve(is_ood, scores)
    return float(fpr[np.searchsorted(tpr, 0.95, side="left")])


def ood_metrics(scores, is_ood, prefix):
    """``{prefix}_auroc``, ``{prefix}_auprc``, ``{prefix}_fpr95`` plus mean ID/OOD scores."""
    scores = np.asarray(scores, dtype=np.float64)
    is_ood = np.asarray(is_ood, dtype=np.int64)
    metrics = {}
    if (is_ood == 0).any():
        metrics[f"{prefix}_mean_id"] = float(scores[is_ood == 0].mean())
    if (is_ood == 1).any():
        metrics[f"{prefix}_mean_ood"] = float(scores[is_ood == 1].mean())
    if np.unique(is_ood).size > 1:
        metrics[f"{prefix}_auroc"] = float(roc_auc_score(is_ood, scores))
        metrics[f"{prefix}_auprc"] = float(average_precision_score(is_ood, scores))
        metrics[f"{prefix}_fpr95"] = fpr_at_95_tpr(is_ood, scores)
    return metrics


def calibration_metrics(probabilities, targets, prefix):
    """``{prefix}_acc``, ``{prefix}_ece``, ``{prefix}_nll``, ``{prefix}_brier`` on ID nodes."""
    probabilities = np.asarray(probabilities, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.int64)
    if len(targets) == 0:
        return {}
    one_hot = np.eye(probabilities.shape[1])[targets]
    return {
        f"{prefix}_acc": float((probabilities.argmax(axis=1) == targets).mean()),
        f"{prefix}_ece": expected_calibration_error(probabilities, targets, num_bins=15),
        f"{prefix}_nll": float(-np.log(probabilities[np.arange(len(targets)), targets] + EPS).mean()),
        f"{prefix}_brier": float(((probabilities - one_hot) ** 2).sum(axis=1).mean()),
    }
