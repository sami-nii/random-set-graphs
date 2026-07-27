import os
import sys

import lightning as L
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.models import EdgeCNN, GAT, GCN, GIN, GraphSAGE
from torchmetrics import AUROC, Accuracy, F1Score
from utils.isotonic_calibration import expected_calibration_error


def generate_matrices(focal_sets, num_classes):
    """
    Pre-computes matrices for fast Moebius inversion and Pignistic transformation.
    """
    num_sets = len(focal_sets)

    # 1. Moebius Inversion Matrix (Belief -> Mass)
    # M[i, j] = (-1)^(len(A_i) - len(A_j)) if A_j is subset of A_i, else 0
    moebius_mat = torch.zeros((num_sets, num_sets))
    for i, set_a in enumerate(focal_sets):
        for j, set_b in enumerate(focal_sets):
            if set_b.issubset(set_a):
                moebius_mat[i, j] = (-1) ** (len(set_a) - len(set_b))

    # 2. Pignistic Transformation Matrix (Mass -> Class Probability)
    # P[i, c] = 1/|A_i| if c in A_i, else 0
    pignistic_mat = torch.zeros((num_sets, num_classes))
    for i, input_set in enumerate(focal_sets):
        cardinality = len(input_set)
        if cardinality > 0:
            for class_idx in input_set:
                if class_idx < num_classes:
                    pignistic_mat[i, class_idx] = 1.0 / cardinality

    return moebius_mat, pignistic_mat


class RandomSetLayer(nn.Module):
    def __init__(self, input_dim, num_focal_sets):
        super().__init__()
        self.linear = nn.Linear(input_dim, num_focal_sets)

    def forward(self, x):
        return torch.sigmoid(self.linear(x))


class RandomSetLoss(nn.Module):
    def __init__(
        self,
        focal_sets,
        moebius_mat,
        alpha=1e-3,
        beta=1e-3,
        use_bce=True,
        use_mr=True,
        use_ms=True,
    ):
        super().__init__()
        self.focal_sets = focal_sets
        self.register_buffer("moebius_mat", moebius_mat)
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
        device = pred_beliefs.device
        target_beliefs = torch.zeros_like(pred_beliefs)

        for i, input_set in enumerate(self.focal_sets):
            member_check = torch.tensor([t.item() in input_set for t in target_indices], device=device)
            target_beliefs[:, i] = member_check.float()

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


models_map = {
    "GCN": GCN,
    "SAGE": GraphSAGE,
    "GAT": GAT,
    "GIN": GIN,
    "EdgeCNN": EdgeCNN,
}


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
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["focal_sets"])

        self.focal_sets = focal_sets
        self.num_classes = num_classes
        self.lr = lr
        self.weight_decay = weight_decay
        self.ood_in_val = ood_in_val

        moebius_mat, pignistic_mat = generate_matrices(focal_sets, num_classes)
        self.register_buffer("moebius_mat", moebius_mat)
        self.register_buffer("pignistic_mat", pignistic_mat)

        self.gnn_model = models_map[gnn_type](
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            num_layers=num_layers,
            out_channels=hidden_channels,
            act=F.relu,
            **kwargs,
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
        )

        self.train_acc = Accuracy(task="multiclass", num_classes=num_classes)
        self.train_f1 = F1Score(task="multiclass", num_classes=num_classes)
        self.val_f1 = F1Score(task="multiclass", num_classes=num_classes)
        self.val_auroc_metric = AUROC(task="binary")
        self.test_auroc_metric = AUROC(task="binary")

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

    def get_pignistic_probs(self, beliefs):
        """
        Converts predicted Beliefs -> Masses -> Pignistic Probabilities
        """
        masses = torch.matmul(beliefs, self.moebius_mat.t())
        masses = torch.clamp(masses, min=0.0)
        residual_mass = torch.clamp(1.0 - masses.sum(dim=1, keepdim=True), min=0.0)

        universal_set = set(range(self.num_classes))
        universal_index = next(
            (i for i, focal_set in enumerate(self.focal_sets) if focal_set == universal_set),
            None,
        )
        if universal_index is not None:
            masses[:, universal_index] = masses[:, universal_index] + residual_mass.squeeze(1)

        mass_sum = masses.sum(dim=1, keepdim=True)
        zero_mass_rows = mass_sum.squeeze(1) <= 1e-8
        if zero_mass_rows.any():
            betp = torch.full(
                (beliefs.size(0), self.num_classes),
                1.0 / self.num_classes,
                dtype=beliefs.dtype,
                device=beliefs.device,
            )
            non_zero_rows = ~zero_mass_rows
            if non_zero_rows.any():
                normalized_masses = masses[non_zero_rows] / mass_sum[non_zero_rows]
                betp[non_zero_rows] = torch.matmul(normalized_masses, self.pignistic_mat)
            return betp

        masses = masses / mass_sum
        betp = torch.matmul(masses, self.pignistic_mat)
        return betp

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
                val_beliefs = beliefs[val_mask]
                val_betp = self.get_pignistic_probs(val_beliefs)
                val_betp = torch.clamp(val_betp, min=1e-8)
                entropy = -torch.sum(val_betp * torch.log(val_betp), dim=1)

                y_val_all = batch.y[val_mask]
                ood_targets = 1 - y_val_all.sum(dim=1).long()

                if ood_targets.numel() > 0 and torch.unique(ood_targets).numel() > 1:
                    self.val_auroc_metric.update(entropy, ood_targets)
                    self.log("val_auroc_entropy", self.val_auroc_metric, on_step=False, on_epoch=True, batch_size=val_count)

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

    def test_step(self, batch, batch_idx):
        beliefs = self(batch)
        test_mask = self._get_step_mask(batch, "test")
        test_count = int(test_mask.sum().item())
        if test_count == 0:
            return

        test_beliefs = beliefs[test_mask]
        y_test = batch.y[test_mask]

        betp = self.get_pignistic_probs(test_beliefs)
        eps = 1e-8
        entropy = -torch.sum(betp * torch.log(betp + eps), dim=1)

        ood_targets = 1 - y_test.sum(dim=1).long()
        if ood_targets.numel() > 0 and torch.unique(ood_targets).numel() > 1:
            self.test_auroc_metric.update(entropy, ood_targets)
            self.log("test_auroc_entropy", self.test_auroc_metric, on_step=False, on_epoch=True, batch_size=test_count)

        is_id = y_test.sum(dim=1) == 1
        if is_id.sum() > 0:
            id_betp = betp[is_id]
            id_targets = torch.argmax(y_test[is_id], dim=1)
            id_preds = torch.argmax(id_betp, dim=1)

            acc = (id_preds == id_targets).float().mean()
            self.log("test_acc_id", acc, batch_size=int(is_id.sum().item()))
            test_ece_id = expected_calibration_error(
                id_betp.detach().cpu().numpy(),
                id_targets.detach().cpu().numpy(),
            )
            self.log("test_ece_id", test_ece_id, batch_size=int(is_id.sum().item()))

    def configure_optimizers(self):
        return torch.optim.Adam(
            self.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
