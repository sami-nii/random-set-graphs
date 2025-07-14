import torch
import lightning as L
from torch_geometric.nn.models import GCN, GraphSAGE, GAT, GIN, EdgeCNN
import torch.nn as nn
import torch.nn.functional as F
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# Assuming these are available and work as before
from ood_metrics import fpr_at_95_tpr
from torchmetrics import Accuracy, F1Score, AUROC 

models_map = {
    "GCN": GCN, "SAGE": GraphSAGE, "GAT": GAT, "GIN": GIN, "EdgeCNN": EdgeCNN
}

class VanillaGNN(L.LightningModule):
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
        **kwargs,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        
        # GNN outputs raw logits for the number of ID classes
        self.gnn_model = models_map[gnn_type](
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            num_layers=num_layers,
            out_channels=out_channels, # This is the number of ID classes
            act=F.sigmoid, 
            **kwargs,
        )
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

        self.apply(self.weights_init)

    def forward(self, data):
        # The model should return raw logits
        logits = self.gnn_model(data.x, data.edge_index)
        return logits

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
        
        num_train_nodes = batch.train_mask.sum()
        self.log("train_loss", loss, batch_size=num_train_nodes)
        self.log("train_acc", accuracy, batch_size=num_train_nodes)
        self.log("train_f1", f1, batch_size=num_train_nodes)
        return loss

    def validation_step(self, batch, batch_idx):
        logits = self(batch)
        
        # Get all predictions and labels for the validation set
        logits_val = logits[batch.val_mask].detach()
        y_val = batch.y[batch.val_mask].detach()

        # OOD Detection using Maximum Softmax Probability (MSP)
        if self.ood_in_val:
            probs = F.softmax(logits_val, dim=1)
            msp_scores, _ = torch.max(probs, dim=1)
            ood_scores = -msp_scores # Lower confidence -> higher score
            
            ood_targets = 1 - y_val.sum(axis=1)
            val_auroc = self.auroc_metric(ood_scores, ood_targets)
            self.log("val_auroc", val_auroc, prog_bar=True)

        # Classification Metrics on ID nodes within the validation set
        id_mask_in_val = (y_val.sum(axis=1) == 1)
        
        id_logits = logits_val[id_mask_in_val]
        id_labels = torch.argmax(y_val[id_mask_in_val], dim=1)
        
        loss = self.criterion(id_logits, id_labels)
        self.log("val_loss", loss, prog_bar=True)
        
        id_preds = torch.argmax(id_logits, dim=1)
        val_acc = self.accuracy_metric(id_preds, id_labels)
        val_f1 = self.f1_metric(id_preds, id_labels)
        
        self.log("val_acc", val_acc)
        self.log("val_f1", val_f1, prog_bar=True)
        return loss
    
    def test_step(self, batch, batch_idx):
        logits = self(batch)

        logits_test = logits[batch.test_mask].detach()
        y_test = batch.y[batch.test_mask].detach()

        # OOD Detection using MSP
        probs = F.softmax(logits_test, dim=1)
        msp_scores, _ = torch.max(probs, dim=1)
        ood_scores = -msp_scores

        ood_targets = 1 - y_test.sum(axis=1)
        test_auroc = self.auroc_metric(ood_scores, ood_targets)
        test_fpr95 = fpr_at_95_tpr(ood_scores, ood_targets)

        self.log("test_auroc", test_auroc)
        self.log("test_fpr95", test_fpr95)

        # Classification Metrics on ID nodes
        id_mask_in_test = (y_test.sum(axis=1) == 1)
        
        id_logits = logits_test[id_mask_in_test]
        id_labels = torch.argmax(y_test[id_mask_in_test], dim=1)
        
        id_preds = torch.argmax(id_logits, dim=1)
        test_acc = self.accuracy_metric(id_preds, id_labels)
        test_f1 = self.f1_metric(id_preds, id_labels)
        
        self.log("test_acc", test_acc)
        self.log("test_f1", test_f1)
        return test_auroc

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