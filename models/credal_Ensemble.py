import torch
import lightning as L
from torch_geometric.nn.models import GCN, GraphSAGE, GAT, GIN, EdgeCNN
import torch.nn as nn
import torch.nn.functional as F
import os
import sys
import numpy as np
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.VanillaGNN import VanillaGNN # Important: It loads VanillaGNN models
from utils.math import compute_uncertainties
from torchmetrics import Accuracy, F1Score, AUROC 

class credal_Ensemble(L.LightningModule):
    """
    A post-hoc, inference-only module to estimate credal uncertainty bounds
    from an ensemble of pre-trained VanillaGNN models.

    This module does not require training. It computes q_L and q_U by taking the
    element-wise min and max of the softmax outputs from M vanilla models.
    """
    def __init__(
        self,
        model_class: L.LightningModule,
        checkpoint_paths: list[str],
    ) -> None:
        super().__init__()
        # This model has no trainable parameters, so hyperparameters are for logging/consistency
        self.save_hyperparameters()

        # Load the ensemble of models from their checkpoints
        self.models = nn.ModuleList()
        print(f"Loading {len(checkpoint_paths)} models for the ensemble...")
        for path in checkpoint_paths:
            model = model_class.load_from_checkpoint(path)
            # Freeze the model to be certain no training happens
            model.freeze() 
            self.models.append(model)
        
        if not self.models:
            raise ValueError("checkpoint_paths cannot be empty.")

        # Get the number of output classes from the first model
        self.C = self.models[0].C

        # Initialize metrics for the test step
        self.accuracy_metric = Accuracy(task="multiclass", num_classes=self.C)
        self.f1_metric = F1Score(task="multiclass", num_classes=self.C)
        self.auroc_metric = AUROC(task="binary")

    def forward(self, data):
        """
        Performs a forward pass through all models in the ensemble to compute
        the credal bounds q_L and q_U.
        """
        all_probs = []

        # 1. Get softmax probabilities from each model in the ensemble
        for model in self.models:
            logits = model(data)
            probs = F.softmax(logits, dim=1)
            all_probs.append(probs)

        # 2. Stack probabilities along a new 'ensemble' dimension
        # Shape: [num_models, num_nodes, num_classes]
        stacked_probs = torch.stack(all_probs, dim=0) # TODO check if this is correct

        # 3. Compute q_L and q_U by taking min and max over the ensemble dimension
        q_L = torch.min(stacked_probs, dim=0).values
        q_U = torch.max(stacked_probs, dim=0).values

        return q_L, q_U

    def test_step(self, batch, batch_idx):
        # The logic here is identical to your credal_GNN_t test_step,
        # allowing for a direct and fair comparison.
        q_L, q_U = self(batch)

        q_L_test = q_L[batch.test_mask].detach().cpu()
        q_U_test = q_U[batch.test_mask].detach().cpu()
        y_test = batch.y[batch.test_mask].detach().cpu()

        # Uncertainty Metrics (OOD Detection)
        TU, AU, EU = compute_uncertainties(q_L_test.numpy(), q_U_test.numpy()) 
        targets = 1 - y_test.sum(axis=1)
        auroc_EU = self.auroc_metric(torch.from_numpy(EU), targets)
        self.log("test_auroc_EU", auroc_EU)
        self.log("test_auroc_AU", self.auroc_metric(torch.from_numpy(AU), targets))
        self.log("test_auroc_TU", self.auroc_metric(torch.from_numpy(TU), targets))

        
        return auroc_EU

    # --- Non-Trainable Methods ---
    def training_step(self, batch, batch_idx):
        pass

    def validation_step(self, batch, batch_idx):
        pass

    def configure_optimizers(self):
        # This model has no parameters to optimize
        return None