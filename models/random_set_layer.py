import torch
import lightning as L
import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torchmetrics import AUROC, Accuracy, F1Score
from models.VanillaGNN import VanillaGNN
from torch_geometric.nn.models import GCN, GraphSAGE, GAT, GIN, EdgeCNN
from utils.math import compute_uncertainties


models_map = {
    "GCN": GCN, "SAGE": GraphSAGE, "GAT": GAT, "GIN": GIN, "EdgeCNN": EdgeCNN
}

class RandomSetLayer(nn.Module):
    # Random set layer that outputs a valid mass function.
    # Include at least all singletons
    def __init__(self, focal_sets):
        super().__init__()
        self.focal_sets = focal_sets
        # Replace final layer with length of new classes
        self.num_sets = len(focal_sets)

        self.linear = nn.Linear(self.num_sets)

    def forward(self, x):
        logits = self.linear(x)
        m = F.softmax(logits, dim=-1)  # Mass function
        return m
    
def pignistic(m, focal_sets, num_classes):
    device = m.device
    N = m.size(0)
    betp = torch.zeros(N, num_classes, device=device)
    
    for i, A in enumerate(focal_sets):
        if len(A) == 0:
            continue
        weight = m[:, i] / len(A)
        for c in A: 
            betp[:, c] += weight
            
            return betp

class BinaryCrossEntropy(nn.Module):
    def __init__(self, focal_sets, num_classes):
        super().__init__
        self.focal_sets = focal_sets
        self.num_classes = num_classes
        self.cross_entropy = nn.CrossEntropyLoss()

    def forward(self, m, y):
        betp = pignistic(m, self.focal_sets, self.num_classes)
        labels = torch.argmax(y, dim=1)
        return self.cross_entropy(betp, labels)
    
    class random_set_GNN(L.LightningModule):
        # Metrics using pignistic probabilities.
        # Forward returns mass. 
        # Final layer size should equal the length of focal sets
        def __init__(
                self,
                gnn_type:str,
                in_channels: int,
                hidden_channels: int,
                num_layers: int,
                focal_sets,
                num_classes: int,
                lr: float = 0.001,
                weight_decay: float = 0.0,
                act=F.sigmoid,
            ):
        
                super().__init__()
                self.save_hyperparameters(ignore=["focal_sets"])
        
                self.gnn_model = models_map[gnn_type](
                    in_channels=in_channels,
                    hidden_channels=hidden_channels,
                    num_layers=num_layers,
                    out_channels=hidden_channels,
                    act=act,
                )
                self.focal_sets = focal_sets
                self.num_classes = num_classes
                self.lr = lr
                self.weight_decay = weight_decay

                self.random_set_layer = RandomSetLayer(
                    input_dim=hidden_channels,
                    focal_sets=focal_sets,
                )
                
                self.f1_score_metric = F1Score(
                    task="multiclass",
                    num_classes=num_classes,
                    )
                
                self.lr = lr
                self.weight_decay = weight_decay

        def forward(self, data):
            h = self.gnn_model(data.x, data.edge_index)
            m = self.random_set_layer(h)
            return m
                
        def test_step(self, batch, batch_idx):
            m = self(batch)
            m_train = m[batch.train_mask]
            y_train = batch.y[batch.train_mask]
            loss = self.criterion(m_train, y_train)

            betp = pignistic(
                 m_train, self.focal_sets, self.num_classes
                 )

            preds = torch.argmax(betp, dim=1)
            labels = torch.argmax(y_train, dim=1)

            acc = (preds == labels).float().mean()
            f1 = self.f1_score_metrics(preds, labels)

            self.log("train_loss", loss)
            self.log("train_acc", acc)
            self.log("train_f1", f1)

            return loss
                
        def validation_step(self, batch, batch_idx):
            m = self(batch)
            m_val = m[batch.val_mask]
            y_val = batch.y[batch.val_mask]

        # --- OOD Detection Metrics ---
            if self.ood_in_val:
                TU, AU, EU = compute_uncertainties(m_val.cpu().numpy(), y_val.cpu().numpy()) 
                targets = 1 - y_val.sum(axis=1) # 1 for OOD, 0 for ID
                auroc_EU = self.auroc_metric(torch.from_numpy(EU).to(self.device), targets)
                auroc_AU = self.auroc_metric(torch.from_numpy(AU).to(self.device), targets)
                auroc_TU = self.auroc_metric(torch.from_numpy(TU).to(self.device), targets)
                self.log("val_auroc_EU", auroc_EU, prog_bar=True) # Monitor this during training
                self.log("val_auroc_AU", auroc_AU)
                self.log("val_auroc_TU", auroc_TU)
                    
                betp = pignistic(
                    m_val, self.focal_sets, self.num_classes
                    )
            val_preds = torch.argmax(betp, dim=1)
            val_labels = torch.argmax(y_val, dim=1)

        def configure_optimizers(self):
            return torch.optim.Adam(
                self.parameters(), lr=self.lr, weight_decay=self.weight_decay)