import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
from torch_geometric.nn.models import GCN, GraphSAGE, GAT, GIN, EdgeCNN
from torchmetrics import Accuracy, F1Score, AUROC
import numpy as np
import sys
import os

# --- Utils for Matrix Generation ---

def generate_matrices(focal_sets, num_classes):
    """
    Pre-computes matrices for fast Moebius inversion and Pignistic transformation.
    """
    num_sets = len(focal_sets)
    
    # 1. Moebius Inversion Matrix (Belief -> Mass)
    # M[i, j] = (-1)^(len(A_i) - len(A_j)) if A_j is subset of A_i, else 0
    moebius_mat = torch.zeros((num_sets, num_sets))
    for i, A in enumerate(focal_sets):
        for j, B in enumerate(focal_sets):
            if B.issubset(A):
                moebius_mat[i, j] = (-1) ** (len(A) - len(B))
                
    # 2. Pignistic Transformation Matrix (Mass -> Class Probability)
    # P[i, c] = 1/|A_i| if c in A_i, else 0
    pignistic_mat = torch.zeros((num_sets, num_classes))
    for i, A in enumerate(focal_sets):
        cardinality = len(A)
        if cardinality > 0:
            for c in A:
                if c < num_classes: # Safety check
                    pignistic_mat[i, c] = 1.0 / cardinality
                    
    return moebius_mat, pignistic_mat

# --- The Layer ---

class RandomSetLayer(nn.Module):
    def __init__(self, input_dim, num_focal_sets):
        super().__init__()
        # Maps hidden representation to the belief of each focal set
        self.linear = nn.Linear(input_dim, num_focal_sets)

    def forward(self, x):
        # Eq 3.2: Sigmoid activation to get Belief values in [0, 1]
        return torch.sigmoid(self.linear(x))

# --- The Loss Function ---

class RandomSetLoss(nn.Module):
    def __init__(self, focal_sets, moebius_mat, alpha=1e-3, beta=1e-3):
        super().__init__()
        self.focal_sets = focal_sets
        self.register_buffer("moebius_mat", moebius_mat) # Saved with model state
        self.alpha = alpha # Regularization weight for non-negativity
        self.beta = beta   # Regularization weight for sum-to-one

    def forward(self, pred_beliefs, target_indices):
        """
        pred_beliefs: (Batch, Num_Sets) - Output of the model
        target_indices: (Batch) - Integer ID labels
        """
        device = pred_beliefs.device
        batch_size = pred_beliefs.size(0)
        num_sets = len(self.focal_sets)

        # 1. Dynamic Ground Truth Encoding
        # Bel(A) = 1 if true_class is in A, else 0
        target_beliefs = torch.zeros_like(pred_beliefs)
        
        # We iterate over sets to fill the mask (Efficient enough for fixed budget K)
        # For a fully vectorized approach with huge K, one might use sparse tensors, 
        # but for K < 1000 loops are fine.
        for i, input_set in enumerate(self.focal_sets):
            # Create a boolean mask: does the target index belong to this set?
            # We use a CPU check for set membership logic to build the mask tensor
            # (Note: This can be optimized further, but is clear for implementation)
            member_check = torch.tensor([t.item() in input_set for t in target_indices], device=device)
            target_beliefs[:, i] = member_check.float()

        # 2. BCE Loss (Eq. 5)
        # Clamp for numerical stability
        pred_beliefs = torch.clamp(pred_beliefs, min=1e-7, max=1.0 - 1e-7)
        bce_loss = F.binary_cross_entropy(pred_beliefs, target_beliefs)

        # 3. Compute Masses for Regularization (Eq. 6)
        # Mass = Belief * Moebius_Matrix
        # We need to transpose logic: Mass vector m = B * M_mat (depending on definition)
        # Based on matrices: Mass[i] = sum_j (Moebius[i,j] * Belief[j]) is wrong direction.
        # Moebius inversion usually: m(A) = Sum (-1)^... Bel(B).
        # With our matrix definition: mass = moebius_mat @ belief.T (per sample)
        # Shapes: (Num_Sets, Num_Sets) @ (Batch, Num_Sets).T -> (Num_Sets, Batch) -> Transpose back
        
        masses = torch.matmul(pred_beliefs, self.moebius_mat.t())

        # 4. Regularization Terms
        # Mr: Penalize negative masses
        mr_loss = torch.relu(-masses).mean()
        
        # Ms: Penalize sum of masses != 1
        ms_loss = torch.abs(masses.sum(dim=1) - 1.0).mean()

        total_loss = bce_loss + (self.alpha * mr_loss) + (self.beta * ms_loss)
        
        return total_loss, bce_loss, mr_loss, ms_loss

# --- The Lightning Module ---

models_map = {
    "GCN": GCN, "SAGE": GraphSAGE, "GAT": GAT, "GIN": GIN, "EdgeCNN": EdgeCNN
}

class RandomSetGNN(L.LightningModule):
    def __init__(
        self,
        gnn_type: str,
        in_channels: int,
        hidden_channels: int,
        num_layers: int,
        focal_sets: list, # List of sets, e.g. [{0}, {1}, {0,1}]
        num_classes: int,
        lr: float = 0.001,
        weight_decay: float = 0.0,
        alpha: float = 1e-3,
        beta: float = 1e-3,
        ood_in_val: bool = True,
        **kwargs
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["focal_sets"])
        
        self.focal_sets = focal_sets
        self.num_classes = num_classes
        self.lr = lr
        self.weight_decay = weight_decay
        self.ood_in_val = ood_in_val

        # 1. Pre-compute matrices
        moebius_mat, pignistic_mat = generate_matrices(focal_sets, num_classes)
        self.register_buffer("moebius_mat", moebius_mat)
        self.register_buffer("pignistic_mat", pignistic_mat)

        # 2. Backbone
        self.gnn_model = models_map[gnn_type](
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            num_layers=num_layers,
            out_channels=hidden_channels,
            act=F.relu, 
            **kwargs
        )

        # 3. RS Layer
        self.rs_layer = RandomSetLayer(hidden_channels, len(focal_sets))

        # 4. Loss
        self.criterion = RandomSetLoss(focal_sets, moebius_mat, alpha, beta)

        # 5. Metrics
        self.train_acc = Accuracy(task="multiclass", num_classes=num_classes)
        self.train_f1 = F1Score(task="multiclass", num_classes=num_classes)
        self.val_f1 = F1Score(task="multiclass", num_classes=num_classes)
        self.auroc_metric = AUROC(task="binary")

    def forward(self, data):
        h = self.gnn_model(data.x, data.edge_index)
        beliefs = self.rs_layer(h) # Shape: [N, num_focal_sets]
        return beliefs

    def get_pignistic_probs(self, beliefs):
        """
        Converts predicted Beliefs -> Masses -> Pignistic Probabilities
        """
        # 1. Belief -> Mass
        masses = torch.matmul(beliefs, self.moebius_mat.t())
        
        # 2. Post-processing (Paper Section 3.2 end): 
        # "Set any negative masses to zero... add universal set... ensure sum is 1"
        # For simplicity/speed in training, we rely on the Loss Regularization.
        # For Inference, we clamp and normalize.
        masses = torch.clamp(masses, min=0.0)
        mass_sum = masses.sum(dim=1, keepdim=True) + 1e-8
        masses = masses / mass_sum
        
        # 3. Mass -> BetP (Pignistic Probability)
        betp = torch.matmul(masses, self.pignistic_mat) # Shape: [N, num_classes]
        return betp

    def training_step(self, batch, batch_idx):
        beliefs = self(batch)
        
        # Filter masks
        train_mask = batch.train_mask
        preds = beliefs[train_mask]
        targets = batch.y[train_mask] # Assuming y is one-hot or index
        
        # Convert One-Hot to Index if necessary
        if targets.dim() > 1 and targets.size(1) > 1:
            target_indices = torch.argmax(targets, dim=1)
        else:
            target_indices = targets

        # Calculate Loss
        loss, bce, mr, ms = self.criterion(preds, target_indices)
        
        # Calculate Metrics (using Pignistic transform)
        with torch.no_grad():
            betp = self.get_pignistic_probs(preds)
            pred_classes = torch.argmax(betp, dim=1)
            acc = self.train_acc(pred_classes, target_indices)
            f1 = self.train_f1(pred_classes, target_indices)

        self.log("train_loss", loss, batch_size=train_mask.sum())
        self.log("train_bce", bce, batch_size=train_mask.sum())
        self.log("train_mr", mr, batch_size=train_mask.sum())
        self.log("train_acc", acc, batch_size=train_mask.sum())
        
        return loss

    def validation_step(self, batch, batch_idx):
        beliefs = self(batch)
        
        # --- OOD Detection (Pignistic Entropy) ---
        if self.ood_in_val:
            # We evaluate Uncertainty on VAL mask (which contains ID + OOD)
            val_mask = batch.val_mask
            val_beliefs = beliefs[val_mask]
            
            # Compute Pignistic Probabilities
            val_betp = self.get_pignistic_probs(val_beliefs)
            
            # Entropy = - sum(p * log(p))
            val_betp = torch.clamp(val_betp, min=1e-8)
            entropy = -torch.sum(val_betp * torch.log(val_betp), dim=1)
            
            # Targets: 1 for OOD, 0 for ID
            # In your data loader: ID has sum(y)=1, OOD has sum(y)=0
            y_val_all = batch.y[val_mask]
            ood_targets = 1 - y_val_all.sum(dim=1).long() # 1 if OOD, 0 if ID
            
            # AUROC of Entropy
            try:
                auroc = self.auroc_metric(entropy, ood_targets)
                self.log("val_auroc_entropy", auroc, batch_size=val_mask.sum())
            except:
                pass # Skip if only one class present in batch

        # --- Classification Metrics (ID Only) ---
        # Identify ID nodes in validation
        y_val_full = batch.y
        is_id = (y_val_full.sum(dim=1) == 1)
        id_val_mask = batch.val_mask & is_id
        
        if id_val_mask.sum() > 0:
            id_beliefs = beliefs[id_val_mask]
            id_targets = batch.y[id_val_mask]
            if id_targets.dim() > 1:
                id_target_indices = torch.argmax(id_targets, dim=1)
            else:
                id_target_indices = id_targets

            # Classification Loss
            loss, _, _, _ = self.criterion(id_beliefs, id_target_indices)
            
            # F1 Score
            id_betp = self.get_pignistic_probs(id_beliefs)
            pred_classes = torch.argmax(id_betp, dim=1)
            f1 = self.val_f1(pred_classes, id_target_indices)
            
            self.log("val_loss", loss, batch_size=id_val_mask.sum())
            self.log("val_f1", f1, batch_size=id_val_mask.sum())

    def test_step(self, batch, batch_idx):
        beliefs = self(batch)
        test_mask = batch.test_mask
        
        test_beliefs = beliefs[test_mask]
        y_test = batch.y[test_mask]
        
        # 1. OOD Metrics
        betp = self.get_pignistic_probs(test_beliefs)
        
        # Entropy
        eps = 1e-8
        entropy = -torch.sum(betp * torch.log(betp + eps), dim=1)
        
        ood_targets = 1 - y_test.sum(dim=1).long()
        auroc = self.auroc_metric(entropy, ood_targets)
        self.log("test_auroc_entropy", auroc)

        # 2. ID Classification Metrics
        is_id = (y_test.sum(dim=1) == 1)
        if is_id.sum() > 0:
            id_betp = betp[is_id]
            id_targets = torch.argmax(y_test[is_id], dim=1)
            id_preds = torch.argmax(id_betp, dim=1)
            
            acc = (id_preds == id_targets).float().mean()
            self.log("test_acc_id", acc)

    def configure_optimizers(self):
        return torch.optim.Adam(
            self.parameters(), 
            lr=self.lr, 
            weight_decay=self.weight_decay
        )