import torch
import lightning as L
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics import Accuracy, F1Score
from torch_geometric.nn.models import GCN, GraphSAGE, GAT, GIN, EdgeCNN

models_map = {
    "GCN": GCN, "SAGE": GraphSAGE, "GAT": GAT, "GIN": GIN, "EdgeCNN": EdgeCNN
}
# Random set layer that outputs a valid mass function.
# Include at least all singletons
class RandomSetLayer(nn.Module):
    def __init__(self, input_dim, focal_sets):
        super().__init__()
        self.focal_sets = focal_sets
        self.num_sets = len(focal_sets)
        self.linear = nn.Linear(input_dim, self.num_sets)
        # Replace final layer with length of new classes

    def forward(self, x):
        scores = torch.sigmoid(self.linear(x))
        return scores / (scores.sum(dim=-1, keepdim=True) + 1e-8)
    
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

class PignisticCrossEntropy(nn.Module):
    def __init__(self, focal_sets, num_classes):
        super().__init__()
        self.focal_sets = focal_sets
        self.num_classes = num_classes
        self.ce = nn.CrossEntropyLoss()

    def forward(self, m, y):
        betp = pignistic(m, self.focal_sets, self.num_classes)
        return self.ce(betp, y)
    
class RandomSetGNN(L.LightningModule):
    def __init__(self, gnn_type, in_channels, hidden_channels, num_layers, focal_sets, num_classes, lr=0.001, weight_decay=0.0, act=F.relu):
        super().__init__()
        self.save_hyperparameters(ignore=["focal_sets"])

        self.gnn_model = models_map[gnn_type](
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            num_layers=num_layers,
            out_channels=hidden_channels,
            act=act,
        )
        self.random_set_layer = RandomSetLayer(input_dim=hidden_channels, focal_sets=focal_sets)
        self.criterion = PignisticCrossEntropy(focal_sets=focal_sets, num_classes=num_classes)
        self.acc = Accuracy(task="multiclass", num_classes=num_classes)
        self.f1 = F1Score(task="multiclass", num_classes=num_classes)
        self.lr = lr
        self.weight_decay = weight_decay
        self.focal_sets = focal_sets
        self.num_classes = num_classes

    def forward(self, data):
        h = self.gnn_model(data.x, data.edge_index)
        m = self.random_set_layer(h)
        return m
                
    def training_step(self, batch, batch_idx):
        m = self(batch)
        y_train = batch.y[batch.train_mask]
        m_train = m[batch.train_mask]

        loss = self.criterion(m_train, y_train)
        betp = pignistic(m_train, self.focal_sets, self.num_classes)
        preds = betp.argmax(dim=1)

        # Convert labels to indices
        labels_idx = y_train.argmax(dim=1) if y_train.ndim > 1 else y_train

        self.log("train_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log("train_acc", self.acc(preds, labels_idx), on_step=False, on_epoch=True)
        self.log("train_f1", self.f1(preds, labels_idx), on_step=False, on_epoch=True)

        return loss

    
    def validation_step(self, batch, batch_idx):
        m = self(batch)
        y_val = batch.y[batch.val_mask]
        m_val = m[batch.val_mask]

        loss = self.criterion(m_val, y_val)
        betp = pignistic(m_val, self.focal_sets, self.num_classes)
        preds = betp.argmax(dim=1)

        labels_idx = y_val.argmax(dim=1) if y_val.ndim > 1 else y_val

        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log("val_acc", self.acc(preds, labels_idx), on_step=False, on_epoch=True)
        self.log("val_f1", self.f1(preds, labels_idx), on_step=False, on_epoch=True)

        return loss
    
    def test_step(self, batch, batch_idx):
        m = self(batch)
        y_test = batch.y
        betp = pignistic(m, self.focal_sets, self.num_classes)
        preds = betp.argmax(dim=1)

        labels_idx = y_test.argmax(dim=1) if y_test.ndim > 1 else y_test

        acc = self.acc(preds, labels_idx)
        f1 = self.f1(preds, labels_idx)

        self.log("test_acc", acc)
        self.log("test_f1", f1)

        return {"test_acc": acc, "test_f1": f1}
    def configure_optimizers(self):
        return torch.optim.Adam(
        self.parameters(), lr=self.lr, weight_decay=self.weight_decay)