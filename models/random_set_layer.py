import torch
import lightning as L
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics import Accuracy, F1Score
from torch_geometric.nn.models import GCN, GraphSAGE, GAT, GIN, EdgeCNN
from utils.math import compute_uncertainties
import os
import sys
import numpy as np

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
    
# new_classes = focal_sets
# Classes = Classes from squirrel dataset

    
# def pignistic(m, focal_sets, num_classes):
#     device = m.device
#     N = m.size(0)
#     betp = torch.zeros(N, num_classes, device=device)
#     for i, A in enumerate(focal_sets):
#         if len(A) == 0:
#             continue
#         weight = m[:, i] / len(A)
#         for c in A:
#             betp[:, c] += weight
#     return betp

def mass_coeff(new_classes):
    num = len(new_classes)
    mass_co = torch.zeros(num, num)
    for i, A in enumerate(new_classes):
        for j, B in enumerate(new_classes):
            if set(B).issubset(set(A)):
                mass_co[j, i] = (-1) ** (len(A) - len(B))
    return mass_co

    # mass_co = np.zeros((len(new_classes), len(new_classes)))
    # mass_co = np.zeros((len(new_classes_1024), len(new_classes_1024)))

# mass_coeff_matrix = mass_coeff(new_classes)
# mass_coeff_matrix = tf.cast(mass_coeff_matrix, tf.float32)

#Mobius inverse function
def belief_to_mass(belief, focal_sets):
    device = belief.device
    num_sets = len(focal_sets)

    mass_coeff = torch.zeros(num_sets, num_sets, device=device)

    for i, A in enumerate(focal_sets):
        for j, B in enumerate(focal_sets):
            if set(B).issubset(A):
                mass_coeff[j, i] = (-1) ** (len(A) - len(B))

    mass = belief @ mass_coeff
    mass = torch.clamp(mass, min=0)

    # normalize
    mass = mass / (mass.sum(dim=-1, keepdim=True) + 1e-8)
    return mass

def betp_approx(classes, new_classes):
    mask = []
    for n in classes:
        row = []
        for A in new_classes:
            if {n}.issubset(A):
                row.append(1 / len(A))
            else:
                row.append(0)
        mask.append(row)
    return torch.tensor(mask).T

# Pignistic probability
def pignistic(mass, classes, new_classes):
    betp_matrix = torch.zeros((len(new_classes), len(classes)))
    for i,c in enumerate(classes): 
        for j,A in enumerate(new_classes):
            if set([c]).issubset(A):
                betp_matrix[j,i] = 1/len(A)
    
    final_bet_p = mass @ betp_matrix

    return final_bet_p

class BinaryCrossEntropy(nn.Module):
    ALPHA = 0.01
    BETA = 0.01

    def __init__(self, mass_coeff_matrix):
        super().__init__()
        self.mass_coeff_matrix = mass_coeff_matrix

    def forward(self, y_pred, y_true):
        eps = 1e-8

        y_true = y_true.float().clamp(eps, 1)
        y_pred = y_pred.clamp(eps, 1 - eps)

        term_0 = (1 - y_true) * torch.log(1 - y_pred)
        term_1 = y_true * torch.log(y_pred)
        bce_loss = -(term_0 + term_1).mean()

        mass = y_pred @ self.mass_coeff_matrix.to(y_pred.device)

        mass_reg = torch.relu(-mass).mean()
        mass_sum = torch.relu(mass.sum(dim=-1).mean() - 1)

        total_loss = (
            bce_loss
            + self.ALPHA * mass_reg
            + self.BETA * mass_sum
        )

        return total_loss
    
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
        self.criterion = BinaryCrossEntropy(focal_sets=focal_sets, num_classes=num_classes)
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
        q = self(batch)
        
        y_train = batch.y[batch.train_mask]
        q_train = q[batch.train_mask]

        loss = self.criterion(q_train, y_train) 
        # betp = pignistic(m_train, self.focal_sets, self.num_classes)
        # preds = betp.argmax(dim=1)

        # Return belief = model prediction instead of pignistic

        # Convert labels to indices
        train_preds = torch.argmax(q_train, dim=1)
        train_labels = torch.argmax(y_train, dim=1)
        
        accuracy = (train_preds == train_labels).float().mean()
        
        f1 = self.f1_score_metric(train_preds, train_labels)
        
        num_train_nodes = batch.train_mask.sum()
        self.log("train_loss", loss, batch_size=num_train_nodes)
        self.log("train_acc", accuracy, batch_size=num_train_nodes)
        self.log("train_f1", f1, batch_size=num_train_nodes)
        return loss
    
    def validation_step(self, batch, batch_idx):
        q, = self(batch)

        # Select the outputs and labels for the validation nodes
        q_val = q[batch.val_mask].detach()
        y_val = batch.y[batch.val_mask].detach()
        
        # --- OOD Detection Metrics ---
        # if self.ood_in_val:
        #     TU, AU, EU = compute_uncertainties(q_val.cpu().numpy())
        #     targets = 1 - y_val.sum(axis=1) # 1 for OOD, 0 for ID
            
        #     auroc_EU = self.auroc_metric(torch.from_numpy(EU).to(self.device), targets)
        #     auroc_AU = self.auroc_metric(torch.from_numpy(AU).to(self.device), targets)
        #     auroc_TU = self.auroc_metric(torch.from_numpy(TU).to(self.device), targets)
            
        #     self.log("val_auroc_EU", auroc_EU, prog_bar=True) # Monitor this during training
        #     self.log("val_auroc_AU", auroc_AU)
        #     self.log("val_auroc_TU", auroc_TU)
        
        # --- Classification Metrics (for ID nodes within the validation set) ---
        id_mask_in_val = (y_val.sum(axis=1) == 1)
        
        # Calculate validation loss ONLY on ID nodes
        loss = self.criterion(q_val[id_mask_in_val], y_val[id_mask_in_val])
        self.log("val_loss", loss, prog_bar=True)

        # Calculate classification metrics
        val_labels = torch.argmax(y_val[id_mask_in_val], dim=1)
        
        val_preds = torch.argmax(q_val[id_mask_in_val], dim=1)

        val_f1 = self.f1_score_metric(val_preds, val_labels)
        
        self.log("val_f1", val_f1, prog_bar=True)


        return loss
    
    def test_step(self, batch, batch_idx):
        m = self(batch)
        y_test = batch.y
        mass_coeff_tensor = mass_coeff_matrix.to(m.device)
        test_preds_mass = torch.matmul(m, mass_coeff_tensor)

        betp_matrix = betp_approx(y_test)
        betp = torch.matmul(test_preds_mass, betp_matrix) 

        # betp = pignistic(m, self.focal_sets, self.num_classes)
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