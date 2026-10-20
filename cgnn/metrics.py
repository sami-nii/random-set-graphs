"""Evaluation metrics shared by all models.

AUROC
-----
Config key ``auroc_impl`` (``set_auroc_impl``):

- ``"torchmetrics"`` (default, used for the paper): ``torchmetrics``' binary AUROC, which
  maps scores outside ``[0, 1]`` through a sigmoid before ranking them (in float32, very
  large scores such as Mahalanobis distances or high-temperature energies become ties);
- ``"exact"``: rank-based (Mann-Whitney U) AUROC in float64 on the raw scores, ties
  counted as 1/2.

Conventions: ``targets`` are 1 for OOD, 0 for ID; higher score = more OOD.
"""

from __future__ import annotations

import torch
from torchmetrics.functional.classification import binary_auroc as _tm_binary_auroc
from torchmetrics.functional.classification import multiclass_f1_score

AUROC_IMPLS = ("exact", "torchmetrics")
_AUROC_IMPL = "exact"


def set_auroc_impl(name: str) -> None:
    global _AUROC_IMPL
    if name not in AUROC_IMPLS:
        raise ValueError(f"auroc_impl must be one of {AUROC_IMPLS}, got {name!r}")
    _AUROC_IMPL = name


def get_auroc_impl() -> str:
    return _AUROC_IMPL


def _average_ranks(x: torch.Tensor) -> torch.Tensor:
    """1-based ranks with ties replaced by their average rank (like scipy.stats.rankdata)."""
    sorted_x, order = torch.sort(x)
    _, inverse, counts = torch.unique_consecutive(sorted_x, return_inverse=True, return_counts=True)
    ends = torch.cumsum(counts, dim=0).to(torch.float64)
    starts = ends - counts.to(torch.float64) + 1
    ranks_sorted = ((starts + ends) / 2)[inverse]
    ranks = torch.empty_like(ranks_sorted)
    ranks[order] = ranks_sorted
    return ranks


def exact_binary_auroc(scores: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    scores = scores.detach().reshape(-1).to(torch.float64)
    positives = targets.detach().reshape(-1).to(scores.device) > 0
    n_pos = positives.sum().to(torch.float64)
    n_neg = (~positives).sum().to(torch.float64)
    if n_pos == 0 or n_neg == 0:
        return torch.tensor(float("nan"), dtype=torch.float64)
    ranks = _average_ranks(scores)
    return (ranks[positives].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def binary_auroc(scores: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """AUROC of ``scores`` for detecting ``targets == 1`` (OOD). Returns a 0-dim float tensor."""
    if _AUROC_IMPL == "torchmetrics":
        return _tm_binary_auroc(scores, targets.long())
    return exact_binary_auroc(scores, targets).to(torch.float32)


def multiclass_f1(preds: torch.Tensor, labels: torch.Tensor, num_classes: int) -> torch.Tensor:
    """Micro-averaged F1 (== accuracy for single-label data). Kept for continuity with logged ``*_f1``."""
    return multiclass_f1_score(preds, labels, num_classes=num_classes, average="micro")


# ---------------------------------------------------------------------------
# Additional metrics of the RS-GNN paper (App. "Evaluation metrics")
# ---------------------------------------------------------------------------
def binary_auprc(scores: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Average precision for detecting ``targets == 1`` (OOD)."""
    from torchmetrics.functional.classification import binary_average_precision

    return binary_average_precision(scores.detach().float(), targets.long())


def fpr_at_95_tpr(scores: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """False-positive rate on ID nodes at the first threshold reaching 95% TPR on OOD nodes."""
    from torchmetrics.functional.classification import binary_roc

    fpr, tpr, _ = binary_roc(scores.detach().float(), targets.long())
    return fpr[torch.searchsorted(tpr, torch.tensor(0.95, device=tpr.device))]


def expected_calibration_error(probs: torch.Tensor, labels: torch.Tensor, num_bins: int = 15) -> torch.Tensor:
    """Top-label ECE with ``num_bins`` equal-width confidence bins (last bin closed)."""
    probs = probs.detach().to(torch.float64)
    confidences, predictions = probs.max(dim=1)
    correct = (predictions == labels).to(torch.float64)
    bins = torch.clamp((confidences * num_bins).long(), max=num_bins - 1)
    ece = torch.zeros((), dtype=torch.float64, device=probs.device)
    for b in range(num_bins):
        in_bin = bins == b
        if in_bin.any():
            ece += (correct[in_bin].mean() - confidences[in_bin].mean()).abs() * in_bin.to(
                torch.float64
            ).mean()
    return ece.float()


def negative_log_likelihood(probs: torch.Tensor, labels: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    return -torch.log(probs.gather(1, labels.view(-1, 1)).squeeze(1) + eps).mean()


def brier_score(probs: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    one_hot = torch.nn.functional.one_hot(labels, probs.size(1)).to(probs.dtype)
    return ((probs - one_hot) ** 2).sum(dim=1).mean()


def ood_detection_metrics(scores: torch.Tensor, targets: torch.Tensor, split: str, key: str = "") -> dict:
    """``{split}_{auroc,auprc,fpr95,mean_id,mean_ood}{_key}``; detection metrics need both ID and OOD."""
    suffix = f"_{key}" if key else ""
    out = {}
    is_ood = targets > 0
    if (~is_ood).any():
        out[f"{split}_mean_id{suffix}"] = scores[~is_ood].float().mean()
    if is_ood.any():
        out[f"{split}_mean_ood{suffix}"] = scores[is_ood].float().mean()
    if is_ood.any() and (~is_ood).any():
        out[f"{split}_auroc{suffix}"] = binary_auroc(scores, targets)
        out[f"{split}_auprc{suffix}"] = binary_auprc(scores, targets)
        out[f"{split}_fpr95{suffix}"] = fpr_at_95_tpr(scores, targets)
    return out


def calibration_metrics(probs: torch.Tensor, labels: torch.Tensor, split: str) -> dict:
    """``{split}_{acc,ece,nll,brier}`` on ID nodes."""
    if labels.numel() == 0:
        return {}
    return {
        f"{split}_acc": (probs.argmax(dim=1) == labels).float().mean(),
        f"{split}_ece": expected_calibration_error(probs, labels),
        f"{split}_nll": negative_log_likelihood(probs, labels),
        f"{split}_brier": brier_score(probs, labels),
    }
