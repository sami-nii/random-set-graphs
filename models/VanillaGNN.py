import torch
import lightning as L
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from torchmetrics import Accuracy, F1Score, AUROC 
from sklearn.metrics import f1_score
from models.backbones import build_backbone
from utils.metrics import calibration_metrics, entropy, ood_metrics


class VanillaGNN(L.LightningModule):
    """Shared backbone + linear classifier + softmax, p_v = softmax(W z_v)."""

    def __init__(
        self,
        gnn_type: str,
        in_channels: int,
        hidden_channels: int,
        num_layers: int,
        out_channels: int,
        lr: float = 0.001,
        weight_decay: float = 0.0,
        ood_in_val: bool = True, # Flag to control validation behavior
        heads: int = 4,
        dropout: float = 0.2,
        **kwargs,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        
        self.gnn_model = build_backbone(
            gnn_type,
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            num_layers=num_layers,
            heads=heads,
            dropout=dropout,
        )
        # Linear classifier over the node embedding; outputs logits for the ID classes
        self.head = nn.Linear(hidden_channels, out_channels)
        self.C = out_channels

        # Use PyTorch's built-in CrossEntropyLoss for stability
        self.criterion = nn.CrossEntropyLoss()

        # Initialize metrics
        self.accuracy_metric = Accuracy(task="multiclass", num_classes=out_channels)
        self.f1_metric = F1Score(task="multiclass", num_classes=out_channels)
        self.auroc_metric = AUROC(task="binary")
        
        self.lr = lr
        self.weight_decay = weight_decay
        self.ood_in_val = ood_in_val
        self._test_outputs = []

        self.apply(self.weights_init)

    def embed(self, data):
        """Node embedding z_v (input of the classifier)."""
        return self.gnn_model(data.x.float(), data.edge_index)

    def layer_representations(self, data):
        """[x^0, H^(1), ..., H^(L), logits], used by the joint-latent baselines."""
        x0 = data.x.float()
        hidden = self.gnn_model.layer_outputs(x0, data.edge_index)
        return [x0] + hidden + [self.head(hidden[-1])]

    def forward(self, data):
        # The model should return raw logits
        return self.head(self.embed(data))

    def training_step(self, batch, batch_idx):
        logits = self(batch)
        
        # Apply train mask to get logits and labels for training nodes
        logits_train = logits[batch.train_mask]
        y_train = batch.y[batch.train_mask]
        
        # The loss function expects class indices, not one-hot vectors
        loss = self.criterion(logits_train, torch.argmax(y_train, dim=1))
        
        # Calculate metrics
        preds = torch.argmax(logits_train, dim=1)
        target = torch.argmax(y_train, dim=1)
        accuracy = self.accuracy_metric(preds, target)
        f1 = self.f1_metric(preds, target)
        
        num_train_nodes = int(batch.train_mask.sum().item())
        self.log("train_loss", loss, batch_size=num_train_nodes)
        self.log("train_acc", accuracy, batch_size=num_train_nodes)
        self.log("train_f1", f1, batch_size=num_train_nodes)
        return loss

    def validation_step(self, batch, batch_idx):
        logits = self(batch)
        
        # Get all predictions and labels for the validation set
        logits_val = logits[batch.val_mask].detach()
        y_val = batch.y[batch.val_mask].detach()
        num_val_nodes = int(batch.val_mask.sum().item())

        # OOD Detection using Maximum Softmax Probability (MSP)
        if self.ood_in_val:
            probs = F.softmax(logits_val, dim=1)
            msp_scores, _ = torch.max(probs, dim=1)
            ood_scores = -msp_scores # Lower confidence -> higher score
            
            ood_targets = 1 - y_val.sum(axis=1)
            val_auroc = self.auroc_metric(ood_scores, ood_targets)
            self.log("val_auroc", val_auroc, prog_bar=True, batch_size=num_val_nodes)

        # Classification Metrics on ID nodes within the validation set
        id_mask_in_val = (y_val.sum(axis=1) == 1)
        
        id_logits = logits_val[id_mask_in_val]
        id_labels = torch.argmax(y_val[id_mask_in_val], dim=1)
        num_id_val_nodes = int(id_mask_in_val.sum().item())
        
        loss = self.criterion(id_logits, id_labels)
        self.log("val_loss", loss, prog_bar=True, batch_size=num_id_val_nodes)
        
        id_preds = torch.argmax(id_logits, dim=1)
        val_acc = self.accuracy_metric(id_preds, id_labels)
        val_f1 = self.f1_metric(id_preds, id_labels)
        
        self.log("val_acc", val_acc, batch_size=num_id_val_nodes)
        self.log("val_f1", val_f1, prog_bar=True, batch_size=num_id_val_nodes)
        return loss
    
    def on_test_epoch_start(self):
        self._test_outputs = []

    def test_step(self, batch, batch_idx):
        logits = self(batch)
        probs = F.softmax(logits[batch.test_mask], dim=1)
        self._test_outputs.append((probs.detach().cpu(), batch.y[batch.test_mask].detach().cpu()))

    def on_test_epoch_end(self):
        if not self._test_outputs:
            return
        probs, y = (torch.cat(parts).numpy() for parts in zip(*self._test_outputs))
        self._test_outputs = []

        is_id = y.sum(axis=1) == 1
        is_ood = (~is_id).astype(np.int64)
        id_labels = y[is_id].argmax(axis=1)
        metrics = {}
        # OOD scores: maximum softmax probability s = 1 - max_i p_v(i), and softmax entropy
        metrics.update(ood_metrics(1.0 - probs.max(axis=1), is_ood, "test_msp"))
        metrics.update(ood_metrics(entropy(probs), is_ood, "test_entropy"))
        metrics.update(calibration_metrics(probs[is_id], id_labels, "test_id"))
        if is_id.any():
            metrics["test_f1"] = float(f1_score(id_labels, probs[is_id].argmax(axis=1), average="micro"))

        # Names used by the existing sweeps and result scripts.
        aliases = {
            "test_auroc": "test_msp_auroc",
            "test_auroc_entropy": "test_entropy_auroc",
            "test_acc": "test_id_acc",
            "test_ece_id": "test_id_ece",
        }
        for alias, key in aliases.items():
            if key in metrics:
                metrics[alias] = metrics[key]
        self.log_dict(metrics)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        return optimizer
     
    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
