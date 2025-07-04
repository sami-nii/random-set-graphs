import torch
import lightning as L
from torch_geometric.nn.models import GCN, GraphSAGE, GAT, GIN, EdgeCNN
import torch.nn as nn
import torch.nn.functional as F
import os
import sys
import numpy as np
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.math import compute_uncertainties
from ood_metrics import fpr_at_95_tpr
from torchmetrics import AUROC
from torchmetrics.classification import F1Score
from models.credal_layer import CredalLayer
from models.credal_loss import CreNetLoss

models_map = {
    "GCN": GCN, "SAGE": GraphSAGE, "GAT": GAT, "GIN": GIN, "EdgeCNN": EdgeCNN
}

class credal_GNN_t(L.LightningModule):
    def __init__(
        self,
        gnn_type: str,
        in_channels: int,
        hidden_channels: int,
        num_layers: int,
        out_channels: int,
        lr: float = 0.001,
        weight_decay: float = 0.0,
        delta = 0.5,
        **kwargs,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        
        self.gnn_model = models_map[gnn_type](
            in_channels=in_channels, hidden_channels=hidden_channels,
            num_layers=num_layers, out_channels=hidden_channels,
            act=F.sigmoid, **kwargs
        )
        self.C = out_channels
        self.credal_layer_model = CredalLayer(input_dim=hidden_channels, C=out_channels)
        self.criterion = CreNetLoss(delta=delta)
        self.lr = lr
        self.weight_decay = weight_decay
        
        self.f1_score_metric = F1Score(task="multiclass", num_classes=self.C)

        self.apply(self.weights_init)

    def forward(self, data):
        representation_layer = self.gnn_model(data.x, data.edge_index)
        q_L, q_U = self.credal_layer_model(representation_layer)
        return q_L, q_U

    def training_step(self, batch, batch_idx):
        q_L, q_U = self(batch)
        
        y_train = batch.y[batch.train_mask]
        q_U_train = q_U[batch.train_mask]
        q_L_train = q_L[batch.train_mask]
        
        loss = self.criterion(q_L_train, q_U_train, y_train) 
        
        train_preds_U = torch.argmax(q_U_train, dim=1)
        train_preds_L = torch.argmax(q_L_train, dim=1)
        train_labels = torch.argmax(y_train, dim=1)
        
        accuracy_U = (train_preds_U == train_labels).float().mean()
        accuracy_L = (train_preds_L == train_labels).float().mean()
        
        ### MODIFIED ###
        f1_U = self.f1_score_metric(train_preds_U, train_labels)
        f1_L = self.f1_score_metric(train_preds_L, train_labels)
        
        num_train_nodes = batch.train_mask.sum()
        self.log("train_loss", loss, batch_size=num_train_nodes)
        self.log("train_acc_U", accuracy_U, batch_size=num_train_nodes)
        self.log("train_acc_L", accuracy_L, batch_size=num_train_nodes)
        self.log("train_f1_U", f1_U, batch_size=num_train_nodes)
        self.log("train_f1_L", f1_L, batch_size=num_train_nodes)
        return loss

    def validation_step(self, batch, batch_idx):
        q_L, q_U = self(batch)

        y_val = batch.y[batch.val_mask]
        q_U_val = q_U[batch.val_mask]
        q_L_val = q_L[batch.val_mask]

        loss = self.criterion(q_L_val, q_U_val, y_val)
        
        val_preds_U = torch.argmax(q_U_val, dim=1)
        val_preds_L = torch.argmax(q_L_val, dim=1)
        val_labels = torch.argmax(y_val, dim=1)

        accuracy_U = (val_preds_U == val_labels).float().mean()
        accuracy_L = (val_preds_L == val_labels).float().mean()

        ### MODIFIED ###
        f1_U = self.f1_score_metric(val_preds_U, val_labels)
        f1_L = self.f1_score_metric(val_preds_L, val_labels)

        num_val_nodes = batch.val_mask.sum()
        self.log("val_loss", loss, batch_size=num_val_nodes, prog_bar=True)
        self.log("val_acc_U", accuracy_U, batch_size=num_val_nodes)
        self.log("val_acc_L", accuracy_L, batch_size=num_val_nodes)
        self.log("val_f1_U", f1_U, batch_size=num_val_nodes, prog_bar=True)
        self.log("val_f1_L", f1_L, batch_size=num_val_nodes)
        return loss

    def test_step(self, batch, batch_idx):
        q_L, q_U = self(batch)

        q_L_test = q_L[batch.test_mask].detach().cpu()
        q_U_test = q_U[batch.test_mask].detach().cpu()
        y_test = batch.y[batch.test_mask].detach().cpu()

        # Uncertainty Metrics (OOD Detection)
        TU, AU, EU = compute_uncertainties(q_L_test.numpy(), q_U_test.numpy()) 
        targets = 1 - y_test.sum(axis=1)
        auroc_calculator = AUROC(task="binary")
        auroc_EU = auroc_calculator(torch.from_numpy(EU), targets)
        self.log("test_auroc_EU", auroc_EU)
        self.log("test_auroc_AU", auroc_calculator(torch.from_numpy(AU), targets))
        self.log("test_auroc_TU", auroc_calculator(torch.from_numpy(TU), targets))
        self.log("test_fpr_95", fpr_at_95_tpr(torch.from_numpy(EU), targets))

        # Classification Metrics (for ID nodes only)
        id_mask_in_test = (y_test.sum(axis=1) == 1)
        
        if id_mask_in_test.sum() > 0:
            id_labels = torch.argmax(y_test[id_mask_in_test], dim=1)
            
            ### MODIFIED ###
            id_preds_U = torch.argmax(q_U_test[id_mask_in_test], dim=1)
            id_preds_L = torch.argmax(q_L_test[id_mask_in_test], dim=1)
            
            id_accuracy_U = (id_preds_U == id_labels).float().mean()
            id_accuracy_L = (id_preds_L == id_labels).float().mean()
            
            id_f1_U = self.f1_score_metric(id_preds_U, id_labels)
            id_f1_L = self.f1_score_metric(id_preds_L, id_labels)
            
            self.log("test_id_accuracy_U", id_accuracy_U)
            self.log("test_id_accuracy_L", id_accuracy_L)
            self.log("test_id_f1_U", id_f1_U)
            self.log("test_id_f1_L", id_f1_L)
        
        return auroc_EU

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