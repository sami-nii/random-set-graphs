"""Standard GNN classifier; also the frozen backbone of every post-hoc method.

OOD score: negative maximum softmax probability (MSP).

``head="conv"`` (CGNN protocol): the last backbone layer emits the logits.
``head="linear"`` (RS-GNN paper): the backbone emits the node embedding
``z_v = act(GNN(x))`` and a linear classifier gives ``softmax(W z_v)``.
Logged metrics: train_{loss,acc,f1}, val_{loss,acc,f1,auroc}, test_{auroc,acc,f1}.

Checkpoints (``checkpoints/``) load with ``VanillaGNN.load_from_checkpoint``.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from cgnn import metrics
from cgnn.models.backbones import build_backbone, joint_representation, layerwise_embeddings
from cgnn.models.base import TrainableUQModule, id_mask, ood_targets


class VanillaGNN(TrainableUQModule):
    def __init__(
        self,
        gnn_type: str,
        in_channels: int,
        hidden_channels: int,
        num_layers: int,
        out_channels: int,
        lr: float = 0.001,
        weight_decay: float = 0.0,
        ood_in_val: bool = True,
        head: str = "conv",
        **kwargs,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        if head not in ("conv", "linear"):
            raise ValueError(f"head must be 'conv' or 'linear', got {head!r}")

        self.head = head
        self.gnn_model = build_backbone(
            gnn_type,
            in_channels,
            hidden_channels,
            num_layers,
            out_channels if head == "conv" else hidden_channels,
            **kwargs,
        )
        if head == "linear":
            self.classifier = nn.Linear(hidden_channels, out_channels)
        self.C = out_channels
        self.criterion = nn.CrossEntropyLoss()
        self.lr = lr
        self.weight_decay = weight_decay
        self.ood_in_val = ood_in_val
        self.apply(self.weights_init)

    def forward(self, data) -> torch.Tensor:
        if self.head == "linear":
            return self.classifier(self.gnn_model.act(self.gnn_model(data.x, data.edge_index)))
        return self.gnn_model(data.x, data.edge_index)  # logits [N, C]

    # -- representations used by the post-hoc detectors -------------------------
    def embedding(self, data) -> torch.Tensor:
        """Penultimate representation: ``z_v`` (linear head) or the second-to-last layer (conv head)."""
        gnn = self.gnn_model
        if self.head == "linear":
            return gnn.act(gnn(data.x, data.edge_index))
        if gnn.num_layers <= 1:
            return self(data)
        return layerwise_embeddings(
            gnn, data.x, data.edge_index, num_layers=gnn.num_layers - 1, act_on_last=True, include_input=False
        )[-1]

    def joint_latent(self, data) -> torch.Tensor:
        """``[x || z^1 || ... || z^L (|| logits)]`` of the backbone (JLDE, GEBM, CGNN post train)."""
        if self.head == "linear":
            hidden = layerwise_embeddings(
                self.gnn_model, data.x, data.edge_index, act_on_last=True, include_input=True
            )
            return torch.cat([*hidden, self.classifier(hidden[-1])], dim=1)
        return joint_representation(
            self.gnn_model, data.x, data.edge_index, act_on_last=False, include_input=True
        )

    @property
    def joint_latent_dim(self) -> int:
        hp = self.hparams
        hidden_layers = hp["num_layers"] if self.head == "linear" else max(hp["num_layers"] - 1, 0)
        return int(hp["in_channels"] + hidden_layers * hp["hidden_channels"] + hp["out_channels"])

    def training_step(self, batch, batch_idx):
        logits = self(batch)
        train_mask = self.split_mask(batch, "train")
        logits_train = logits[train_mask]
        target = torch.argmax(batch.y[train_mask], dim=1)

        loss = self.criterion(logits_train, target)
        preds = torch.argmax(logits_train, dim=1)

        n = train_mask.sum()
        self.log("train_loss", loss, batch_size=n)
        self.log("train_acc", (preds == target).float().mean(), batch_size=n)
        self.log("train_f1", self.f1(preds, target), batch_size=n)
        return loss

    # -- validation -----------------------------------------------------------
    def on_validation_epoch_start(self):
        self.buffer("val").clear()

    def validation_step(self, batch, batch_idx):
        logits = self(batch)
        val_mask = self.split_mask(batch, "val")
        logits_val = logits[val_mask].detach()
        y_val = batch.y[val_mask].detach()

        is_id = id_mask(y_val)
        loss = None
        if is_id.any():
            id_logits = logits_val[is_id]
            id_labels = torch.argmax(y_val[is_id], dim=1)
            loss = self.criterion(id_logits, id_labels)
            id_preds = torch.argmax(id_logits, dim=1)
            n = is_id.sum()
            self.log("val_loss", loss, prog_bar=True, batch_size=n, on_step=False, on_epoch=True)
            self.log(
                "val_acc", (id_preds == id_labels).float().mean(), batch_size=n, on_step=False, on_epoch=True
            )
        else:
            id_labels = torch.empty(0, dtype=torch.long, device=self.device)
            id_preds = torch.empty(0, dtype=torch.long, device=self.device)

        buf = self.buffer("val")
        buf.add(id_labels=id_labels, id_preds=id_preds)
        if self.ood_in_val:
            msp = F.softmax(logits_val, dim=1).max(dim=1).values
            buf.add(ood_scores=-msp, ood_targets=ood_targets(y_val))
        return loss

    def on_validation_epoch_end(self):
        buf = self.buffer("val")
        if not buf:
            return
        id_labels = buf.cat("id_labels")
        if id_labels.numel() > 0:
            self.log("val_f1", self.f1(buf.cat("id_preds"), id_labels), prog_bar=True)
        if self.ood_in_val and "ood_scores" in buf:
            self.log("val_auroc", self.auroc(buf.cat("ood_scores"), buf.cat("ood_targets")), prog_bar=True)
        buf.clear()

    # -- test -----------------------------------------------------------------
    def on_test_epoch_start(self):
        self.buffer("test").clear()

    def test_step(self, batch, batch_idx):
        logits = self(batch)
        test_mask = self.split_mask(batch, "test")
        logits_test = logits[test_mask].detach()
        y_test = batch.y[test_mask].detach()

        probs = F.softmax(logits_test, dim=1)
        msp = probs.max(dim=1).values
        is_id = id_mask(y_test)
        self.buffer("test").add(
            ood_scores=-msp,
            entropy=-(probs * torch.log(probs + 1e-8)).sum(dim=1),
            ood_targets=ood_targets(y_test),
            id_labels=torch.argmax(y_test[is_id], dim=1),
            id_preds=torch.argmax(logits_test[is_id], dim=1),
            id_probs=probs[is_id],
        )
        return -msp

    def on_test_epoch_end(self):
        buf = self.buffer("test")
        if not buf:
            return
        targets = buf.cat("ood_targets")
        self.log("test_auroc", self.auroc(buf.cat("ood_scores"), targets))
        id_labels, id_preds = buf.cat("id_labels"), buf.cat("id_preds")
        if id_labels.numel() > 0:
            self.log("test_acc", (id_preds == id_labels).float().mean())
            self.log("test_f1", self.f1(id_preds, id_labels))
        # RS-GNN paper road-scene tables: softmax entropy and calibration on ID nodes.
        extra = metrics.ood_detection_metrics(buf.cat("entropy"), targets, "test", "entropy")
        extra.update(metrics.calibration_metrics(buf.cat("id_probs"), id_labels, "test_id"))
        self.log_dict(extra)
        buf.clear()
