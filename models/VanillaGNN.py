import torch
import lightning as L
from torch_geometric.nn.models import GCN, GraphSAGE, GAT, GIN, EdgeCNN
import torch.nn as nn
import torch.nn.functional as F
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.math import cross_entropy_loss
from torchmetrics import Accuracy, F1Score, AUROC 


models_map = {
    "GCN": GCN,  # GCN is built-in in torch_geometric
    "SAGE": GraphSAGE,  # SAGE is built-in in torch_geometric
    "GAT": GAT,  # GAT is built-in in torch_geometric
    "GIN": GIN,  # GIN is built-in in torch_geometric
    "EdgeCNN": EdgeCNN,  # EdgeCNN is built-in in torch_geometric
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
        **kwargs,
    ) -> None:
        super().__init__()
        
        # Dynamically initialize the GNN model
        self.gnn_model = models_map[gnn_type](
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            num_layers=num_layers,
            out_channels=out_channels,
            act=F.sigmoid,
            **kwargs,
        )

        self.C = out_channels

        # Cross-Entropy Loss for multi-class classification
        self.criterion = cross_entropy_loss

        self.accuracy = Accuracy(task="multiclass", num_classes=out_channels)
        self.f1 = F1Score(task="multiclass", num_classes=out_channels)
        
        # Learning parameters
        self.lr = lr
        self.weight_decay = weight_decay

        # Save hyperparameters
        self.save_hyperparameters()

        self.apply(self.weights_init)

    def forward(self, data):
        # Forward method to process node features and edges
        out =  F.softmax(self.gnn_model(data.x, data.edge_index)) # shape (num_nodes * hidden_dim)

        assert len(out.shape) == 2, f"Representation layer must be 2D tensor, but got shape {out.shape}"
        assert out.shape[1] == self.gnn_model.out_channels, f"Representation layer must have {self.gnn_model.out_channels} channels, but got {out.shape[1]} channels"

        return out 

        
    def training_step(self, batch, batch_idx):
        
        out = self(batch)  # shape: [num_nodes, C]

        loss = self.criterion(out, batch.y).mean()

        pred = torch.argmax(out, dim=1)
        target = torch.argmax(batch.y, dim=1)

        accuracy = self.accuracy(pred, target)
        f1 = self.f1(pred, target)

        batch_size = batch.y.shape[0]
        
        # Logging
        self.log("train_loss", loss, batch_size=batch_size)
        self.log("train_acc", accuracy, batch_size=batch_size)
        self.log("train_f1", f1, batch_size=batch_size)

        return loss

    def validation_step(self, batch, batch_idx):
        
        out = self(batch)  # shape: [num_nodes, C]

        loss = self.criterion(out, batch.y).mean()

        pred = torch.argmax(out, dim=1)
        target = torch.argmax(batch.y, dim=1)

        accuracy = self.accuracy(pred, target)
        f1 = self.f1(pred, target)

        batch_size = batch.y.shape[0]
        
        # Logging
        self.log("val_loss", loss, batch_size=batch_size)
        self.log("val_acc", accuracy, batch_size=batch_size)
        self.log("val_f1", f1, batch_size=batch_size)

        return loss
    
    def test_step(self, batch, batch_idx):
        
        out = self(batch)  # shape: [num_nodes, C]

        loss = self.criterion(out, batch.y).mean()

        pred = torch.argmax(out, dim=1)
        target = torch.argmax(batch.y, dim=1)

        accuracy = self.accuracy(pred, target)
        f1 = self.f1(pred, target)

        batch_size = batch.y.shape[0]
        
        # Logging
        self.log("test_loss", loss, batch_size=batch_size)
        self.log("test_acc", accuracy, batch_size=batch_size)
        self.log("test_f1", f1, batch_size=batch_size)

        return loss
    

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        return optimizer
     
    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            nn.init.zeros_(m.bias)

