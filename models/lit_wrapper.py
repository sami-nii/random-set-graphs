import torch
from torch_geometric.data.lightning import LightningDataset
import lightning as L
from torch_geometric.nn.models import GCN
from typing import Optional


models_map = {
    "GCN": GCN, # GCN is built-in in torch_geometric
}


def convert_to_lit_dataset(data):
    return LightningDataset(data)


class LitGraphNN(L.LightningModule):
    def __init__(
        self,
        gnn_type: str,
        in_channels: int,
        out_channels: int,
        hidden_channels: Optional[int] = None,
        num_layers: int = 1,
        lr: float = 0.001,
        weight_decay: float = 0.0,
        scaling_factor: float = 1.0,
        **kwargs,
    ) -> None:
        super().__init__()

        self.model = models_map[gnn_type](
            in_channels=in_channels,
            out_channels=out_channels,
            hidden_channels=hidden_channels,
            num_layers=num_layers,
            **kwargs,
        )

        self.criterion = torch.nn.MSELoss()
        
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=lr, weight_decay=weight_decay
        )
        self.scaling_factor = scaling_factor

        # save hyperparameters
        self.save_hyperparameters()

    def forward(self, data):
        return self.model(data)

    def training_step(self, batch, batch_idx):
        out: torch.Tensor = self.model(batch).squeeze(-1)
        loss = self.criterion(out, batch.y)

        self.log(
            "train_loss",
            loss,
            sync_dist=True,
            batch_size=batch.y.size(0),
        )
        self.log(
            "train_mae",
            torch.nn.functional.l1_loss(
                out.detach() * self.scaling_factor, batch.y * self.scaling_factor
            ),
            sync_dist=True,
            batch_size=batch.y.size(0),
        )
        self.log(
            "train_mse",
            torch.nn.functional.mse_loss(
                out.detach() * self.scaling_factor, batch.y * self.scaling_factor
            ),
            sync_dist=True,
            batch_size=batch.y.size(0),
        )
        return loss

    def validation_step(self, batch, batch_idx):
        out = self.model(batch).squeeze(-1)
        loss = self.criterion(out, batch.y)

        self.log(
            "val_loss", loss, sync_dist=True, batch_size=batch.y.size(0)
        )
        self.log(
            "val_mae",
            torch.nn.functional.l1_loss(
                out.detach() * self.scaling_factor, batch.y * self.scaling_factor
            ),
            sync_dist=True,
            batch_size=batch.y.size(0),
        )
        self.log(
            "val_mse",
            torch.nn.functional.mse_loss(
                out.detach() * self.scaling_factor, batch.y * self.scaling_factor
            ),
            sync_dist=True,
            batch_size=batch.y.size(0),
        )
        return loss

    def test_step(self, batch, batch_idx):
        out = self.model(batch).squeeze(-1)
        loss = self.criterion(out, batch.y)

        self.log(
            "test_loss", loss, sync_dist=True, batch_size=batch.y.size(0)
        )
        self.log(
            "test_mae",
            torch.nn.functional.l1_loss(
                out.detach() * self.scaling_factor, batch.y * self.scaling_factor
            ),
            sync_dist=True,
            batch_size=batch.y.size(0),
        )
        self.log(
            "test_mse",
            torch.nn.functional.mse_loss(
                out.detach() * self.scaling_factor, batch.y * self.scaling_factor
            ),
            sync_dist=True,
            batch_size=batch.y.size(0),
        )
        return loss

    def configure_optimizers(self):
        return self.optimizer
