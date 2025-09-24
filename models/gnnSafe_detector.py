# models/gnnsafe_detector.py

import torch
import lightning as L
from torchmetrics import AUROC
import torch.nn.functional as F

# --- You might need to install torch_sparse ---
# pip install torch_sparse
from torch_sparse import SparseTensor, matmul
from torch_geometric.utils import degree

# --- Import your custom modules ---
from models.VanillaGNN import VanillaGNN

class GNNSafeDetector(L.LightningModule):
    def __init__(self, backbone_ckpt_path: str, K: int = 2, alpha: float = 0.5):
        super().__init__()
        self.save_hyperparameters()

        # Load and freeze the pre-trained backbone
        self.backbone = VanillaGNN.load_from_checkpoint(backbone_ckpt_path)
        self.backbone.eval()
        for param in self.backbone.parameters():
            param.requires_grad = False
        
        # Hyperparameters for the propagation step
        self.K = K
        self.alpha = alpha
        
        # Metrics for validation and testing
        self.val_auroc = AUROC(task="binary")
        self.test_auroc = AUROC(task="binary")

    def _propagate(self, energy_scores, edge_index):
        """
        Applies the energy-based belief propagation for K steps.
        This corresponds to Equation (7) in the GNNSafe paper.
        """
        e = energy_scores.unsqueeze(1) # Shape: [N, 1]
        N = e.shape[0]
        row, col = edge_index
        
        # Calculate D^-1 * A
        d = degree(col, N, dtype=e.dtype)
        d_inv = 1.0 / d
        d_inv[torch.isinf(d_inv)] = 0 # Handle isolated nodes
        
        norm_values = d_inv[col]
        
        # Create the normalized adjacency matrix D^-1 * A
        adj = SparseTensor(row=col, col=row, value=norm_values, sparse_sizes=(N, N))
        
        # Iterative propagation
        for _ in range(self.K):
            e = e * self.alpha + matmul(adj, e) * (1 - self.alpha)
        
        return e.squeeze(1)

    def forward(self, data):
        """
        Calculates the final GNNSafe score by first computing the initial energy
        from logits, and then applying the propagation scheme.
        """
        with torch.no_grad():
            logits = self.backbone(data)

        # 1. Calculate initial energy score (from Liu et al., 2020)
        initial_energy_scores = -torch.logsumexp(logits, dim=1)

        # 2. Apply energy belief propagation (from Wu et al., 2023)
        propagated_energy_scores = self._propagate(initial_energy_scores, data.edge_index)
        
        # The final OOD score is the negative of the propagated energy
        ood_scores = -propagated_energy_scores # Lower energy (higher score) means more ID

        return ood_scores

    # The validation and test steps are standard for post-hoc OOD detectors
    def validation_step(self, batch, batch_idx):
        ood_scores_all = self(batch).cpu()
        val_mask = batch.val_mask
        ood_scores_val = ood_scores_all[val_mask]
        y_val = batch.y[val_mask]
        targets = (y_val.sum(dim=1) == 0).long()
        self.val_auroc.update(ood_scores_val, targets)
        self.log('val_auroc', self.val_auroc, on_step=False, on_epoch=True, prog_bar=True)

    def test_step(self, batch, batch_idx):
        ood_scores_all = self(batch).cpu()
        test_mask = batch.test_mask
        ood_scores_test = ood_scores_all[test_mask]
        y_test = batch.y[test_mask]
        targets = (y_test.sum(dim=1) == 0).long()
        self.test_auroc.update(ood_scores_test, targets)
        self.log('test_auroc', self.test_auroc, on_step=False, on_epoch=True, prog_bar=True)

    def configure_optimizers(self): return None