import torch
from torch_geometric.data.lightning import LightningDataset
import lightning as L
from torch_geometric.nn.models import GCN
from typing import Optional
import torch.nn as nn
import torch.nn.functional as F
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.solvers import calculate_entropy
from utils.math import interval_softmax, actually_reachable, cross_entropy_loss, checker

models_map = {
    "GCN": GCN,  # GCN is built-in in torch_geometric
}

def convert_to_lit_dataset(data):
    return LightningDataset(data)

class CredalGNN(L.LightningModule):
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
            out_channels=hidden_channels,
            **kwargs,
        )

        self.C = out_channels

        # credal wrapper will be a MLP 
        self.credal_layer_model = CredalLayer(input_dim=hidden_channels, C=out_channels)

        # Cross-Entropy Loss for multi-class classification
        self.criterion = CreNetLoss(eta=0.6) # TODO add as hyperparameter
        
        # Learning parameters
        self.lr = lr
        self.weight_decay = weight_decay

        # Save hyperparameters
        self.save_hyperparameters()

    def forward(self, data):
        # Forward method to process node features and edges
        representation_layer = self.gnn_model(data.x, data.edge_index) # shape (num_nodes * hidden_dim)
        q_L, q_U = self.credal_layer_model(representation_layer)

        assert len(q_L.shape) == 2, "Output of credal_layer_model must be a 2D tensor"
        assert q_U.shape == q_L.shape, f"q_L and q_U must have the same shape, but got {q_L.shape} and {q_U.shape}"
        assert q_L.shape[1] == self.C, f"Output shape must be (num_nodes, {self.C}), but got {q_L.shape}"
        assert q_U.shape[1] == self.C, f"Output shape must be (num_nodes, {self.C}), but got {q_U.shape}"
        assert torch.all(q_L <= q_U), "Lower bounds must be less than or equal to upper bounds"

        return q_L, q_U
        

    def training_step(self, batch, batch_idx):
        
        q_L, q_U = self(batch)  # shape: [num_nodes, 2*C]

        loss = self.criterion(q_L, q_U, batch.y) 

        accuracy = (torch.argmax(q_U, dim=1) == torch.argmax(batch.y)).float().mean()

        # Logging
        self.log("train_loss", loss)
        self.log("train_acc", accuracy)
        return loss

    def validation_step(self, batch, batch_idx):
        
        q_L, q_U = self(batch)  # shape: [num_nodes, C] each

        loss = self.criterion(q_L, q_U, batch.y) 

        accuracy = (torch.argmax(q_U, dim=1) == torch.argmax(batch.y)).float().mean()

        # Logging
        self.log("val_loss", loss)
        self.log("val_acc", accuracy)
        return loss

    def test_step(self, batch, batch_idx):
        
        q_L, q_U = self(batch)  # shape: [num_nodes, C] each
        
        loss = self.criterion(q_L, q_U, batch.y)  

        accuracy = (torch.argmax(q_U, dim=1) == torch.argmax(batch.y)).float().mean() 

        q_L_star, q_U_star = actually_reachable(q_L, q_U) # shape: [num_nodes, C] each

        assert torch.all(q_L_star <= q_U_star), "Lower bounds must be less than or equal to upper bounds"

        _, TU = calculate_entropy(q_L_star.cpu().numpy(), q_U_star.cpu().numpy(), "maximize") 
        _, AU = calculate_entropy(q_L_star.cpu().numpy(), q_U_star.cpu().numpy(), "minimize")
        EU = TU - AU

        # for each node print the the EU and the last column of the target
        for i in range(len(batch.x)):
            print(f"Node {i}: EU = {EU[i]}, Target = {batch.y[i]}")
        
        # Logging
        self.log("test_loss", loss, on_epoch=True)
        self.log("test_acc", accuracy, on_epoch=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        return optimizer


class CredalLayer(torch.nn.Module):
    """
        Credal layer: the input dim is the size of the representation layer

        Parameters:
        ----------
        input_dim: int
            The size of the representation layer z
        C: int
            The number of classes in output
    """

    def __init__(self, input_dim, C, softplus_bias=1e-6):
        super().__init__()

        self.C = C
        self.softplus_bias = softplus_bias
        
        self.mh_layer = nn.Linear(in_features=input_dim, out_features= 2 * C)
        

    def forward(self, z):
        
        assert len(z.shape) == 2, "Input must be a 2D tensor"
        assert z.shape[1] == self.mh_layer.in_features, f"Input shape must be (num_nodes, {self.mh_layer.in_features})"
        
        C = self.C
        
        # Apply the mh_layer (first layer of the credal structure)
        mh = self.mh_layer(z)

        assert len(mh.shape) == 2, "Output of mh_layer must be a 2D tensor"
        assert mh.shape[1] == 2 * C, f"Output shape must be (num_nodes, {2 * C})"
        
        m = mh[:, :C]  # Interval midpoint
        h = mh[:, C:]  # Half-size of the interval
        
        # Split into m-part and h-part
        m = F.sigmoid(m)  # TODO is sigmoid the right choice?
        h = F.softplus(h) + self.softplus_bias  # Ensure h is positive for numerical stability

        # Calculate the interval boundaries
        a_L = m - h  # Lower bounds
        a_U = m + h  # Upper bounds

        assert torch.all(a_L <= a_U), "Lower bounds must be less than or equal to upper bounds"

        q_L, q_U = interval_softmax(a_L, a_U)
        
        assert torch.all(q_L <= q_U), f"Lower bounds must be less than or equal to upper bounds."
        return q_L, q_U  


class CreNetLoss(nn.Module):
    def __init__(self, eta):
        super().__init__()
        self.eta = eta

    def forward(self, q_L, q_U, target):
        """
        CreNet loss function
        Parameters:
        ----------
        q_L: torch.Tensor
            Lower bounds of the credal set, shape (num_nodes, C)
        q_U: torch.Tensor
            Upper bounds of the credal set, shape (num_nodes, C)
        target: torch.Tensor
            Target labels (one-hot encoded), shape (num_nodes, C)
        Returns:
        ----------
        torch.Tensor
            The loss value        
        """
        
        assert len(q_L.shape) == 2 and len(q_U.shape) == 2, f"q_L and q_U must be 2D tensors, but got shapes {q_L.shape} and {q_U.shape}"
        assert q_L.shape == q_U.shape, f"q_L and q_U must have the same shape, but got {q_L.shape} and {q_U.shape}"
        assert target.shape[0] == q_L.shape[0], f"target and q_L must have the same number of samples, but got {target.shape[0]} and {q_L.shape[0]}"
        assert target.shape[1] == q_L.shape[1], f"target and q_L must have the same number of classes, but got {target.shape[1]} and {q_L.shape[1]}"

        # Number of nodes
        num_nodes = q_L.shape[0]

        # Cross-entropy loss for q_U
        vanilla_component = cross_entropy_loss(q_U, target).mean()

        # Cross-entropy loss for q_L
        dro_component = cross_entropy_loss(q_L, target)

        print(f'shape of dro_component: {dro_component.shape}, shape of target: {target.shape}, shape of q_L: {q_L.shape}')
        assert len(dro_component.shape) == 1, f"dro_component must be a 1D tensor, but got shape {dro_component.shape}"
        assert dro_component.shape[0] == num_nodes, f"dro_component must have the same number of samples as num_nodes, but got {dro_component.shape[0]} and {num_nodes}"

        # Select top eta * num_nodes cross-entropy values from dro_component
        top_eta_values, _ = torch.topk(input=dro_component, k = max(1, int(self.eta * num_nodes)))

        dro_component = top_eta_values.mean()

        # Total loss as the sum of both components
        total_loss = vanilla_component + dro_component

        return total_loss


