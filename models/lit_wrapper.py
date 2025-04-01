import torch
from torch_geometric.data.lightning import LightningDataset
import lightning as L
from torch_geometric.nn.models import GCN
from typing import Optional

models_map = {
    "GCN": GCN,  # GCN is built-in in torch_geometric
}

def convert_to_lit_dataset(data):
    return LightningDataset(data)

class LitGraphNN(L.LightningModule):
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
        self.model = models_map[gnn_type](
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            num_layers=num_layers,
            out_channels=out_channels,
            **kwargs,
        )

        # Cross-Entropy Loss for multi-class classification
        self.criterion = torch.nn.CrossEntropyLoss()
        
        # Learning parameters
        self.lr = lr
        self.weight_decay = weight_decay

        # Save hyperparameters
        self.save_hyperparameters()

    def forward(self, data):
        # Forward method to process node features and edges
        return self.model(data.x, data.edge_index)

    def training_step(self, batch, batch_idx):
        logits = self(batch)  # logits shape: [batch_size, num_classes]
        loss = self.criterion(logits, batch.y)

        # Calculate accuracy
        preds = torch.argmax(logits, dim=1)
        accuracy = (preds == batch.y).float().mean()

        # Logging
        self.log("train_loss", loss)
        self.log("train_acc", accuracy)
        return loss

    def validation_step(self, batch, batch_idx):
        logits = self(batch)
        loss = self.criterion(logits, batch.y)

        # Calculate accuracy
        preds = torch.argmax(logits, dim=1)
        accuracy = (preds == batch.y).float().mean()

        # Logging
        self.log("val_loss", loss, on_epoch=True)
        self.log("val_acc", accuracy, on_epoch=True)
        return loss

    def test_step(self, batch, batch_idx):
        logits = self(batch)
        loss = self.criterion(logits, batch.y)

        # Calculate accuracy
        preds = torch.argmax(logits, dim=1)
        accuracy = (preds == batch.y).float().mean()

        # Logging
        self.log("test_loss", loss, on_epoch=True)
        self.log("test_acc", accuracy, on_epoch=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        return optimizer
