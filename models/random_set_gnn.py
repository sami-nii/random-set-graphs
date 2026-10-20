import lightning as L
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics import AUROC, Accuracy, F1Score

from models.backbones import build_backbone
from utils.metrics import calibration_metrics, ood_metrics


def generate_matrices(focal_sets, num_classes):
    """
    Pre-computes the matrices of the focal family F:
      - Moebius inversion matrix (Belief -> Mass), Eq. (mobius)
      - Pignistic matrix (Mass -> BetP)
      - Membership matrix, membership[i, c] = 1 if c in A_i (Plausibility of singletons)
      - Singleton matrix, singleton[i, c] = 1 if A_i == {c} (Belief of singletons)
    """
    num_sets = len(focal_sets)

    # M[i, j] = (-1)^(len(A_i) - len(A_j)) if A_j is subset of A_i, else 0
    moebius_mat = torch.zeros((num_sets, num_sets))
    for i, set_a in enumerate(focal_sets):
        for j, set_b in enumerate(focal_sets):
            if set_b.issubset(set_a):
                moebius_mat[i, j] = (-1) ** (len(set_a) - len(set_b))

    # P[i, c] = 1/|A_i| if c in A_i, else 0
    pignistic_mat = torch.zeros((num_sets, num_classes))
    membership_mat = torch.zeros((num_sets, num_classes))
    singleton_mat = torch.zeros((num_sets, num_classes))
    for i, input_set in enumerate(focal_sets):
        cardinality = len(input_set)
        for class_idx in input_set:
            if class_idx < num_classes:
                pignistic_mat[i, class_idx] = 1.0 / cardinality
                membership_mat[i, class_idx] = 1.0
                if cardinality == 1:
                    singleton_mat[i, class_idx] = 1.0

    return moebius_mat, pignistic_mat, membership_mat, singleton_mat


class RandomSetLayer(nn.Module):
    def __init__(self, input_dim, num_focal_sets):
        super().__init__()
        self.linear = nn.Linear(input_dim, num_focal_sets)

    def forward(self, x):
        return torch.sigmoid(self.linear(x))


class RandomSetLoss(nn.Module):
    """L = BCE(Bel_target, Bel) + alpha * R_neg + beta * R_sum, Eq. (final_loss)."""

    def __init__(
        self,
        focal_sets,
        moebius_mat,
        alpha=1e-3,
        beta=1e-3,
        use_bce=True,
        use_mr=True,
        use_ms=True,
        num_classes=None,
    ):
        super().__init__()
        self.focal_sets = focal_sets
        self.register_buffer("moebius_mat", moebius_mat)
        num_classes = num_classes or (max(max(s) for s in focal_sets) + 1)
        membership = torch.zeros((len(focal_sets), num_classes))
        for i, focal_set in enumerate(focal_sets):
            membership[i, list(focal_set)] = 1.0
        self.register_buffer("membership_mat", membership)
        self.alpha = alpha
        self.beta = beta
        self.use_bce = use_bce
        self.use_mr = use_mr
        self.use_ms = use_ms

    def forward(self, pred_beliefs, target_indices):
        """
        pred_beliefs: (Batch, Num_Sets) - Output of the model
        target_indices: (Batch) - Integer ID labels
        """
        # Bel_target(A) = 1{y in A}
        target_beliefs = self.membership_mat[:, target_indices.long()].t().to(pred_beliefs.dtype)

        pred_beliefs = torch.clamp(pred_beliefs, min=1e-7, max=1.0 - 1e-7)
        bce_loss = F.binary_cross_entropy(pred_beliefs, target_beliefs)

        masses = torch.matmul(pred_beliefs, self.moebius_mat.t())
        mr_loss = torch.relu(-masses).mean()
        ms_loss = torch.abs(masses.sum(dim=1) - 1.0).mean()

        total_loss = torch.zeros((), device=pred_beliefs.device, dtype=pred_beliefs.dtype)
        if self.use_bce:
            total_loss = total_loss + bce_loss
        if self.use_mr:
            total_loss = total_loss + (self.alpha * mr_loss)
        if self.use_ms:
            total_loss = total_loss + (self.beta * ms_loss)

        return total_loss, bce_loss, mr_loss, ms_loss


class RandomSetGNN(L.LightningModule):
    def __init__(
        self,
        gnn_type: str,
        in_channels: int,
        hidden_channels: int,
        num_layers: int,
        focal_sets: list,
        num_classes: int,
        lr: float = 0.001,
        weight_decay: float = 0.0,
        alpha: float = 1e-3,
        beta: float = 1e-3,
        loss_ablation: str = "full",
        use_bce_loss: bool = True,
        use_mr_loss: bool = True,
        use_ms_loss: bool = True,
        ood_in_val: bool = True,
        heads: int = 4,
        dropout: float = 0.2,
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["focal_sets"])

        self.focal_sets = focal_sets
        self.num_classes = num_classes
        self.lr = lr
        self.weight_decay = weight_decay
        self.ood_in_val = ood_in_val

        moebius_mat, pignistic_mat, membership_mat, singleton_mat = generate_matrices(focal_sets, num_classes)
        self.register_buffer("moebius_mat", moebius_mat)
        self.register_buffer("pignistic_mat", pignistic_mat)
        self.register_buffer("membership_mat", membership_mat)
        self.register_buffer("singleton_mat", singleton_mat)

        universal_set = set(range(num_classes))
        self.universal_index = next(
            (i for i, focal_set in enumerate(focal_sets) if focal_set == universal_set),
            None,
        )

        self.gnn_model = build_backbone(
            gnn_type,
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            num_layers=num_layers,
            heads=heads,
            dropout=dropout,
        )

        self.rs_layer = RandomSetLayer(hidden_channels, len(focal_sets))
        self.criterion = RandomSetLoss(
            focal_sets,
            moebius_mat,
            alpha,
            beta,
            use_bce=use_bce_loss,
            use_mr=use_mr_loss,
            use_ms=use_ms_loss,
            num_classes=num_classes,
        )

        self.train_acc = Accuracy(task="multiclass", num_classes=num_classes)
        self.train_f1 = F1Score(task="multiclass", num_classes=num_classes)
        self.val_f1 = F1Score(task="multiclass", num_classes=num_classes)
        self.val_auroc_metric = AUROC(task="binary")
        self.val_auroc_credal_metric = AUROC(task="binary")
        self._test_outputs = []

    def forward(self, data):
        h = self.gnn_model(data.x.float(), data.edge_index)
        beliefs = self.rs_layer(h)
        return beliefs

    def _get_step_mask(self, batch, split):
        mask = getattr(batch, f"{split}_mask")

        # NeighborLoader places the seed nodes first in sampled batches. Restrict
        # losses to those nodes only for sampled subgraphs, not for full-graph
        # DataLoader batches where `batch_size` counts graphs.
        if (
            hasattr(batch, "batch_size")
            and batch.batch_size is not None
            and (hasattr(batch, "n_id") or hasattr(batch, "input_id"))
        ):
            seed_mask = torch.zeros(mask.size(0), dtype=torch.bool, device=mask.device)
            seed_mask[:batch.batch_size] = True
            return mask & seed_mask

        return mask

    def get_valid_masses(self, beliefs):
        """
        Belief head steps 3-8 (Fig. rsgnn_belief_head): Moebius inversion, clip
        negative masses, assign the residual to the universal set when it is a
        focal set, and renormalise. Returns the masses and a mask of degenerate
        rows (total mass <= eps) for which BetP falls back to uniform.
        """
        masses = torch.matmul(beliefs, self.moebius_mat.t())
        masses = torch.clamp(masses, min=0.0)
        if self.universal_index is not None:
            residual_mass = torch.clamp(1.0 - masses.sum(dim=1), min=0.0)
            masses[:, self.universal_index] = masses[:, self.universal_index] + residual_mass

        mass_sum = masses.sum(dim=1, keepdim=True)
        degenerate = mass_sum.squeeze(1) <= 1e-8
        masses = torch.where(degenerate.unsqueeze(1), masses, masses / mass_sum.clamp_min(1e-8))
        return masses, degenerate

    def betp_from_masses(self, masses, degenerate):
        betp = torch.matmul(masses, self.pignistic_mat)
        uniform = torch.full_like(betp, 1.0 / self.num_classes)
        return torch.where(degenerate.unsqueeze(1), uniform, betp)

    def get_pignistic_probs(self, beliefs):
        """Beliefs -> valid masses -> pignistic probabilities BetP."""
        masses, degenerate = self.get_valid_masses(beliefs)
        return self.betp_from_masses(masses, degenerate)

    def credal_width(self, masses, predictions):
        """W_v = Pl({y_hat}) - Bel({y_hat}), Eq. (credal_width)."""
        plausibility = torch.matmul(masses, self.membership_mat)
        belief = torch.matmul(masses, self.singleton_mat)
        width = plausibility - belief
        return width.gather(1, predictions.unsqueeze(1)).squeeze(1)

    def predict_uncertainty(self, beliefs):
        """Returns BetP, pignistic entropy H_v (Eq. pignistic_entropy) and credal width W_v."""
        masses, degenerate = self.get_valid_masses(beliefs)
        betp = self.betp_from_masses(masses, degenerate)
        entropy_scores = -torch.sum(betp * torch.log(betp + 1e-8), dim=1)
        width = self.credal_width(masses, betp.argmax(dim=1))
        return betp, entropy_scores, width

    def training_step(self, batch, batch_idx):
        beliefs = self(batch)

        train_mask = self._get_step_mask(batch, "train")
        num_train = int(train_mask.sum().item())
        if num_train == 0:
            return None

        preds = beliefs[train_mask]
        targets = batch.y[train_mask]

        if targets.dim() > 1 and targets.size(1) > 1:
            target_indices = torch.argmax(targets, dim=1)
        else:
            target_indices = targets

        loss, bce, mr, ms = self.criterion(preds, target_indices)

        with torch.no_grad():
            betp = self.get_pignistic_probs(preds)
            pred_classes = torch.argmax(betp, dim=1)
            acc = self.train_acc(pred_classes, target_indices)
            f1 = self.train_f1(pred_classes, target_indices)

        self.log("train_loss", loss, batch_size=num_train)
        self.log("train_bce", bce, batch_size=num_train)
        self.log("train_mr", mr, batch_size=num_train)
        self.log("train_ms", ms, batch_size=num_train)
        self.log("train_acc", acc, batch_size=num_train)

        return loss

    def validation_step(self, batch, batch_idx):
        beliefs = self(batch)

        if self.ood_in_val:
            val_mask = self._get_step_mask(batch, "val")
            val_count = int(val_mask.sum().item())
            if val_count > 0:
                _, entropy_scores, width = self.predict_uncertainty(beliefs[val_mask])
                ood_targets = 1 - batch.y[val_mask].sum(dim=1).long()

                if ood_targets.numel() > 0 and torch.unique(ood_targets).numel() > 1:
                    self.val_auroc_metric.update(entropy_scores, ood_targets)
                    self.val_auroc_credal_metric.update(width, ood_targets)
                    self.log("val_auroc_entropy", self.val_auroc_metric, on_step=False, on_epoch=True, batch_size=val_count)
                    self.log("val_auroc_credal", self.val_auroc_credal_metric, on_step=False, on_epoch=True, batch_size=val_count)

        y_val_full = batch.y
        is_id = y_val_full.sum(dim=1) == 1
        id_val_mask = self._get_step_mask(batch, "val") & is_id

        if id_val_mask.sum() > 0:
            id_beliefs = beliefs[id_val_mask]
            id_targets = batch.y[id_val_mask]
            if id_targets.dim() > 1:
                id_target_indices = torch.argmax(id_targets, dim=1)
            else:
                id_target_indices = id_targets

            loss, _, _, _ = self.criterion(id_beliefs, id_target_indices)
            id_betp = self.get_pignistic_probs(id_beliefs)
            pred_classes = torch.argmax(id_betp, dim=1)
            f1 = self.val_f1(pred_classes, id_target_indices)

            self.log("val_loss", loss, batch_size=int(id_val_mask.sum().item()))
            self.log("val_f1", f1, batch_size=int(id_val_mask.sum().item()))

    def on_test_epoch_start(self):
        self._test_outputs = []

    def test_step(self, batch, batch_idx):
        beliefs = self(batch)
        test_mask = self._get_step_mask(batch, "test")
        if not test_mask.any():
            return

        betp, entropy_scores, width = self.predict_uncertainty(beliefs[test_mask])
        self._test_outputs.append((
            betp.detach().cpu(),
            entropy_scores.detach().cpu(),
            width.detach().cpu(),
            batch.y[test_mask].detach().cpu(),
        ))

    def on_test_epoch_end(self):
        if not self._test_outputs:
            return
        betp, entropy_scores, width, y = (torch.cat(parts).numpy() for parts in zip(*self._test_outputs))
        self._test_outputs = []

        is_id = y.sum(axis=1) == 1
        is_ood = (~is_id).astype(np.int64)
        metrics = {}
        metrics.update(ood_metrics(entropy_scores, is_ood, "test_entropy"))
        metrics.update(ood_metrics(width, is_ood, "test_credal"))
        metrics.update(calibration_metrics(betp[is_id], y[is_id].argmax(axis=1), "test_id"))

        # Names used by the existing sweeps and result scripts.
        aliases = {
            "test_auroc_entropy": "test_entropy_auroc",
            "test_auroc_credal": "test_credal_auroc",
            "test_acc_id": "test_id_acc",
            "test_ece_id": "test_id_ece",
        }
        for alias, key in aliases.items():
            if key in metrics:
                metrics[alias] = metrics[key]
        self.log_dict(metrics)

    def configure_optimizers(self):
        return torch.optim.Adam(
            self.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
