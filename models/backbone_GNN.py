import torch
import lightning as L
from torch_geometric.nn.models import GCN, GraphSAGE, GAT, GIN, EdgeCNN
import torch.nn as nn
import torch.nn.functional as F
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.math import compute_uncertainties
from ood_metrics import fpr_at_95_tpr
from torchmetrics import AUROC
from models.credal_layer import CredalLayer
from models.credal_loss import CreNetLoss

models_map = {
    "GCN": GCN,  # GCN is built-in in torch_geometric
    "SAGE": GraphSAGE,  # SAGE is built-in in torch_geometric
    "GAT": GAT,  # GAT is built-in in torch_geometric
    "GIN": GIN,  # GIN is built-in in torch_geometric
    "EdgeCNN": EdgeCNN,  # EdgeCNN is built-in in torch_geometric
}


class UncertaintyGNN(L.LightningModule):
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
        
        # Dynamically initialize the GNN model
        self.gnn_model = models_map[gnn_type](
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            num_layers=num_layers,
            out_channels=hidden_channels,
            act=F.sigmoid,
            **kwargs,
        )

        self.C = out_channels

        # credal wrapper will be a MLP 
        self.credal_layer_model = CredalLayer(input_dim=hidden_channels, C=out_channels)

        # Cross-Entropy Loss for multi-class classification
        self.criterion = CreNetLoss(delta=delta) # TODO add as hyperparameter
        
        # Learning parameters
        self.lr = lr
        self.weight_decay = weight_decay

        # Save hyperparameters
        self.save_hyperparameters()

        self.apply(self.weights_init)

    def forward(self, data):
        # Forward method to process node features and edges
        representation_layer = self.gnn_model(data.x, data.edge_index) # shape (num_nodes * hidden_dim)
        q_L, q_U = self.credal_layer_model(representation_layer)

        assert len(q_L.shape) == 2, "Output of credal_layer_model must be a 2D tensor"
        assert q_U.shape == q_L.shape, f"q_L and q_U must have the same shape, but got {q_L.shape} and {q_U.shape}"
        assert q_L.shape[1] == self.C, f"Output shape must be (num_nodes, {self.C}), but got {q_L.shape}"
        assert q_U.shape[1] == self.C, f"Output shape must be (num_nodes, {self.C}), but got {q_U.shape}"

        return q_L, q_U
        

    def training_step(self, batch, batch_idx):
        
        q_L, q_U = self(batch)  # shape: [num_nodes, 2*C]

        loss = self.criterion(q_L, q_U, batch.y) 

        accuracy_U = (torch.argmax(q_U, dim=1) == torch.argmax(batch.y, dim=1)).float().mean()
        accuracy_L = (torch.argmax(q_L, dim=1) == torch.argmax(batch.y, dim=1)).float().mean()

        batch_size = batch.y.shape[0]
        # Logging
        self.log("train_loss", loss, batch_size=batch_size)
        self.log("train_acc_U", accuracy_U, batch_size=batch_size)
        self.log("train_acc_L", accuracy_L, batch_size=batch_size)
        return loss

    def validation_step(self, batch, batch_idx):
        
        q_L, q_U = self(batch)  # shape: [num_nodes, C] each

        loss = self.criterion(q_L, q_U, batch.y) 

        accuracy_U = (torch.argmax(q_U, dim=1) == torch.argmax(batch.y, dim=1)).float().mean()
        accuracy_L = (torch.argmax(q_L, dim=1) == torch.argmax(batch.y, dim=1)).float().mean()

        batch_size = batch.y.shape[0]
        # Logging
        self.log("val_loss", loss, batch_size=batch_size)
        self.log("val_acc_U", accuracy_U, batch_size=batch_size)
        self.log("val_acc_L", accuracy_L, batch_size=batch_size)
        return loss

    def test_step(self, batch, batch_idx):
        
        q_L, q_U = self(batch)  # shape: [num_nodes, C] each

        loss = self.criterion(q_L, q_U, batch.y)  

        q_L = q_L.detach().cpu().numpy()
        q_U = q_U.detach().cpu().numpy()
        batch.y = batch.y.detach().cpu().numpy()

        TU, AU, EU = compute_uncertainties(q_L, q_U) 

        print(f"AU: {AU.mean()}, TU: {TU.mean()}, EU: {EU.mean()}")


        auroc = AUROC(task="binary")


        auroc_score_EU = auroc(torch.from_numpy(EU), torch.from_numpy(1 - batch.y.sum(axis=1)))
        auroc_score_AU = auroc(torch.from_numpy(AU), torch.from_numpy(1 - batch.y.sum(axis=1)))
        auroc_score_TU = auroc(torch.from_numpy(TU), torch.from_numpy(1 - batch.y.sum(axis=1)))

        fpr_95 = fpr_at_95_tpr(-EU, batch.y.sum(axis=1))
        
        # Logging
        self.log("test_auroc", auroc_score_EU) # the "standard" auroc is the one computed with epistemic uncertainty
        self.log("test_auroc_AU", auroc_score_AU)
        self.log("test_auroc_TU", auroc_score_TU)
        self.log("test_fpr_95", fpr_95)
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

