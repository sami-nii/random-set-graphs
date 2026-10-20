"""Random-Set Graph Neural Networks (RS-GNN, the paper's method).

A message-passing backbone gives ``z_v = act(GNN(x))``; the belief head predicts
``Bel_v(A) = sigmoid(g(z_v))_A`` for every focal set ``A`` of the family ``F``.

- Masses: Moebius inversion ``m = Bel @ M^T`` (Eq. mobius).
- Inference (Fig. rsgnn_belief_head): clip negative masses, assign the residual
  to the universal set when it is focal, renormalise (uniform BetP for a
  degenerate output), ``BetP = m @ P``.
- Scores: pignistic entropy ``H_v`` (total, Eq. pignistic_entropy) and credal
  width ``W_v = Pl({y}) - Bel({y})`` of the predicted class (epistemic, Eq. credal_width).
- Loss: ``BCE(1{y in A}, Bel) + alpha R_neg + beta R_sum`` (Eq. final_loss).

Logged metrics: ``train_{loss,bce,mr,ms,acc}``, ``val_{loss,f1,auroc_entropy,auroc_credal}``,
``test_{auroc,auprc,fpr95,mean_id,mean_ood}_{entropy,credal}``, ``test_{acc,f1}``,
``test_id_{acc,ece,nll,brier}`` and, with ``isotonic_calibration``,
``isotonic_test_{id_ece,id_nll,id_acc,auroc_entropy}_{before,after}``.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from cgnn import metrics
from cgnn.models.backbones import build_backbone
from cgnn.models.base import TrainableUQModule, id_mask, ood_targets
from cgnn.models.focal_sets import FocalSets, focal_set_matrices

LOSS_ABLATIONS = {
    # name: (BCE, R_neg, R_sum)
    "full": (True, True, True),
    "bce_only": (True, False, False),
    "bce_mr": (True, True, False),
    "bce_ms": (True, False, True),
    "mr_ms": (False, True, True),
}


class RandomSetLoss(nn.Module):
    """``BCE(Bel_target, Bel) + alpha * R_neg + beta * R_sum`` with ``Bel_target(A) = 1{y in A}``."""

    def __init__(
        self, membership: torch.Tensor, moebius: torch.Tensor, alpha=1e-3, beta=1e-3, ablation="full"
    ):
        super().__init__()
        if ablation not in LOSS_ABLATIONS:
            raise ValueError(f"loss_ablation must be one of {sorted(LOSS_ABLATIONS)}, got {ablation!r}")
        self.register_buffer("membership", membership)
        self.register_buffer("moebius", moebius)
        self.alpha, self.beta = alpha, beta
        self.use_bce, self.use_mr, self.use_ms = LOSS_ABLATIONS[ablation]

    def forward(self, beliefs: torch.Tensor, targets: torch.Tensor):
        target_beliefs = self.membership[:, targets].t().to(beliefs.dtype)
        beliefs = torch.clamp(beliefs, 1e-7, 1.0 - 1e-7)
        bce = F.binary_cross_entropy(beliefs, target_beliefs)
        masses = beliefs @ self.moebius.t()
        mr = torch.relu(-masses).mean()
        ms = torch.abs(masses.sum(dim=1) - 1.0).mean()
        loss = torch.zeros((), device=beliefs.device, dtype=beliefs.dtype)
        if self.use_bce:
            loss = loss + bce
        if self.use_mr:
            loss = loss + self.alpha * mr
        if self.use_ms:
            loss = loss + self.beta * ms
        return loss, bce, mr, ms


class RandomSetGNN(TrainableUQModule):
    def __init__(
        self,
        gnn_type: str,
        in_channels: int,
        hidden_channels: int,
        num_layers: int,
        out_channels: int,
        focal_sets: FocalSets,
        num_classes: int | None = None,
        lr: float = 0.001,
        weight_decay: float = 0.0,
        alpha: float = 1e-3,
        beta: float = 1e-3,
        loss_ablation: str = "full",
        ood_in_val: bool = True,
        isotonic_calibration: bool = False,
        **kwargs,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        # out_channels: ID classes (labels); num_classes: classes of BetP, which also
        # covers OOD singletons when the family spans the full class space.
        self.C = out_channels
        self.num_classes = num_classes or out_channels
        self.focal_sets = [sorted(int(c) for c in s) for s in focal_sets]
        self.lr = lr
        self.weight_decay = weight_decay
        self.ood_in_val = ood_in_val
        self.isotonic_calibration = isotonic_calibration

        moebius, pignistic, membership, singleton = focal_set_matrices(self.focal_sets, self.num_classes)
        self.register_buffer("moebius", moebius)
        self.register_buffer("pignistic", pignistic)
        self.register_buffer("membership", membership)
        self.register_buffer("singleton", singleton)
        universal = list(range(self.num_classes))
        self.universal_index = next((i for i, s in enumerate(self.focal_sets) if s == universal), None)

        self.gnn_model = build_backbone(
            gnn_type, in_channels, hidden_channels, num_layers, hidden_channels, **kwargs
        )
        self.belief_head = nn.Linear(hidden_channels, len(self.focal_sets))
        self.criterion = RandomSetLoss(membership, moebius, alpha, beta, loss_ablation)
        self.apply(self.weights_init)
        self._val_calibration = None

    # -- belief head --------------------------------------------------------------
    def forward(self, data) -> torch.Tensor:
        z = self.gnn_model.act(self.gnn_model(data.x, data.edge_index))
        return torch.sigmoid(self.belief_head(z))

    def valid_masses(self, beliefs: torch.Tensor):
        """Moebius inversion, clip, residual -> universal set, renormalise. Returns ``(m, degenerate)``."""
        masses = torch.clamp(beliefs @ self.moebius.t(), min=0.0)
        if self.universal_index is not None:
            residual = torch.clamp(1.0 - masses.sum(dim=1), min=0.0)
            masses[:, self.universal_index] = masses[:, self.universal_index] + residual
        total = masses.sum(dim=1, keepdim=True)
        degenerate = total.squeeze(1) <= 1e-8
        masses = torch.where(degenerate.unsqueeze(1), masses, masses / total.clamp_min(1e-8))
        return masses, degenerate

    def predict(self, beliefs: torch.Tensor):
        """``(BetP, H, W)``: pignistic probabilities, pignistic entropy, credal width."""
        masses, degenerate = self.valid_masses(beliefs)
        betp = masses @ self.pignistic
        betp = torch.where(degenerate.unsqueeze(1), torch.full_like(betp, 1.0 / self.num_classes), betp)
        entropy = -(betp * torch.log(betp + 1e-8)).sum(dim=1)
        width = (masses @ self.membership - masses @ self.singleton).gather(1, betp.argmax(1, keepdim=True))
        return betp, entropy, width.squeeze(1)

    # -- training -------------------------------------------------------------------
    def training_step(self, batch, batch_idx):
        beliefs = self(batch)
        mask = self.split_mask(batch, "train")
        if not mask.any():
            return None
        targets = torch.argmax(batch.y[mask], dim=1)
        loss, bce, mr, ms = self.criterion(beliefs[mask], targets)
        with torch.no_grad():
            betp, _, _ = self.predict(beliefs[mask].detach())
        n = int(mask.sum())
        self.log("train_loss", loss, batch_size=n)
        self.log("train_bce", bce, batch_size=n)
        self.log("train_mr", mr, batch_size=n)
        self.log("train_ms", ms, batch_size=n)
        self.log("train_acc", (betp.argmax(dim=1) == targets).float().mean(), batch_size=n)
        return loss

    # -- validation -----------------------------------------------------------------
    def on_validation_epoch_start(self):
        self.buffer("val").clear()

    def validation_step(self, batch, batch_idx):
        beliefs = self(batch)
        mask = self.split_mask(batch, "val")
        beliefs, y = beliefs[mask].detach(), batch.y[mask].detach()
        betp, entropy, width = self.predict(beliefs)
        is_id = id_mask(y)
        if is_id.any():
            labels = torch.argmax(y[is_id], dim=1)
            loss, _, _, _ = self.criterion(beliefs[is_id], labels)
            self.log("val_loss", loss, batch_size=int(is_id.sum()), on_step=False, on_epoch=True)
        self.buffer("val").add(
            entropy=entropy,
            width=width,
            ood_targets=ood_targets(y),
            id_probs=betp[is_id],
            id_labels=torch.argmax(y[is_id], dim=1),
        )

    def on_validation_epoch_end(self):
        buf = self.buffer("val")
        if not buf:
            return
        id_probs, id_labels = buf.cat("id_probs"), buf.cat("id_labels")
        if id_labels.numel() > 0:
            self.log("val_f1", self.f1(id_probs.argmax(dim=1), id_labels), prog_bar=True)
        targets = buf.cat("ood_targets")
        if self.ood_in_val and 0 < int(targets.sum()) < targets.numel():
            self.log("val_auroc_entropy", self.auroc(buf.cat("entropy"), targets), prog_bar=True)
            self.log("val_auroc_credal", self.auroc(buf.cat("width"), targets))
        # Labelled ID validation nodes of the last epoch fit the isotonic calibrator.
        self._val_calibration = (id_probs.cpu(), id_labels.cpu())
        buf.clear()

    # -- test -----------------------------------------------------------------------
    def on_test_epoch_start(self):
        self.buffer("test").clear()

    def test_step(self, batch, batch_idx):
        mask = self.split_mask(batch, "test")
        betp, entropy, width = self.predict(self(batch)[mask].detach())
        self.buffer("test").add(probs=betp, entropy=entropy, width=width, y=batch.y[mask].detach())

    def on_test_epoch_end(self):
        buf = self.buffer("test")
        if not buf:
            return
        probs, y = buf.cat("probs"), buf.cat("y")
        targets, is_id = ood_targets(y), id_mask(y)
        id_labels = torch.argmax(y[is_id], dim=1)
        out = {}
        out.update(metrics.ood_detection_metrics(buf.cat("entropy"), targets, "test", "entropy"))
        out.update(metrics.ood_detection_metrics(buf.cat("width"), targets, "test", "credal"))
        out.update(metrics.calibration_metrics(probs[is_id], id_labels, "test_id"))
        if id_labels.numel() > 0:
            out["test_acc"] = (probs[is_id].argmax(dim=1) == id_labels).float().mean()
            out["test_f1"] = self.f1(probs[is_id].argmax(dim=1), id_labels)
        if self.isotonic_calibration:
            out.update(self._isotonic_metrics(probs, targets, is_id, id_labels))
        self.log_dict(out)
        buf.clear()

    def _isotonic_metrics(self, probs, targets, is_id, id_labels) -> dict:
        """One-vs-rest isotonic calibration fitted on labelled ID validation nodes."""
        from cgnn.uncertainty.isotonic import OneVsRestIsotonicCalibrator

        if self._val_calibration is None or self._val_calibration[1].numel() == 0:
            raise RuntimeError("isotonic_calibration needs a validation epoch with labelled ID nodes")
        val_probs, val_labels = self._val_calibration
        calibrator = OneVsRestIsotonicCalibrator().fit(val_probs.numpy(), val_labels.numpy())
        calibrated = torch.as_tensor(calibrator.predict_proba(probs.cpu().numpy()), dtype=probs.dtype)
        out = {}
        for tag, p in (("before", probs.cpu()), ("after", calibrated)):
            cal = metrics.calibration_metrics(p[is_id.cpu()], id_labels.cpu(), "test_id")
            for name in ("acc", "ece", "nll"):
                if f"test_id_{name}" in cal:
                    out[f"isotonic_test_id_{name}_{tag}"] = cal[f"test_id_{name}"]
            entropy = -(p * torch.log(p.clamp_min(1e-12))).sum(dim=1)
            if 0 < int(targets.sum()) < targets.numel():
                out[f"isotonic_test_auroc_entropy_{tag}"] = self.auroc(entropy, targets.cpu())
        return out
