import torch
import lightning as L
from torchmetrics import AUROC
import numpy as np
import faiss # Requires faiss-cpu or faiss-gpu

# --- Import your custom modules ---
from models.VanillaGNN import VanillaGNN # Assuming your VanillaGNN is here
import torch.nn.functional as F

class KNNDetector(L.LightningModule):
    def __init__(self, backbone_ckpt_path: str, k: int = 50):
        super().__init__()
        self.save_hyperparameters()

        # Load and freeze the pre-trained backbone
        self.backbone = VanillaGNN.load_from_checkpoint(backbone_ckpt_path)
        self.backbone.eval()
        for param in self.backbone.parameters():
            param.requires_grad = False
        
        # Hyperparameter for the method
        self.k = k
        
        # Placeholder for the faiss index, which will store the training embeddings
        self.faiss_index = None
        
        # Metrics for validation and testing
        self.val_auroc = AUROC(task="binary")
        self.test_auroc = AUROC(task="binary")

    def precompute_statistics(self, train_data):
        """
        Pre-computes the faiss index from the ID training data embeddings.
        This corresponds to the preprocessing step in Algorithm 1 of the paper.
        """
        print(f"Building Faiss index for k={self.k}...")
        
        data = train_data.to(self.device)
        self.backbone.to(self.device)

        # Get embeddings for all nodes in the graph
        with torch.no_grad():
            all_embeddings = self.backbone.get_embeddings(data)
        
        # Filter to get embeddings for only the training nodes
        train_embeddings = all_embeddings[data.train_mask]

        # --- CRITICAL STEP: Feature Normalization ---
        # As emphasized in the paper, normalizing the features is key to good performance.
        train_embeddings_norm = F.normalize(train_embeddings, p=2, dim=1)

        # Build the faiss index
        feature_dim = train_embeddings_norm.shape[1]
        self.faiss_index = faiss.IndexFlatL2(feature_dim)
        
        # Add the normalized training embeddings to the index
        self.faiss_index.add(train_embeddings_norm.cpu().numpy())
        
        print(f"Faiss index built with {self.faiss_index.ntotal} vectors.")

    def forward(self, data):
        """
        Calculates the k-th nearest neighbor distance score for all nodes.
        The OOD score is the negative of this distance.
        """
        if self.faiss_index is None:
            raise RuntimeError("Faiss index not built. Please call `precompute_statistics(data)` before running.")
        
        with torch.no_grad():
            all_embeddings = self.backbone.get_embeddings(data)
        
        # Normalize the test embeddings as well
        all_embeddings_norm = F.normalize(all_embeddings, p=2, dim=1)

        # Search for the k nearest neighbors for all nodes
        # The `search` method returns distances and indices
        distances, _ = self.faiss_index.search(all_embeddings_norm.cpu().numpy(), self.k)
        
        # The score is the distance to the k-th nearest neighbor.
        # Faiss returns squared L2 distances, so we can use them directly.
        kth_distances = distances[:, -1] # Get the k-th distance (last column)
        
        # The paper's decision is based on distance, so a higher distance means more OOD.
        # To make it a confidence score (higher = ID), we negate it.
        ood_scores = -torch.from_numpy(kth_distances)

        return ood_scores

    def validation_step(self, batch, batch_idx):
        ood_scores_all = self(batch).to(self.device)
        val_mask = batch.val_mask
        
        ood_scores_val = ood_scores_all[val_mask]
        y_val = batch.y[val_mask]

        targets = (y_val.sum(dim=1) == 0).long()

        self.val_auroc.update(ood_scores_val, targets)
        self.log('val_auroc', self.val_auroc, on_step=False, on_epoch=True, prog_bar=True)

    def test_step(self, batch, batch_idx):
        ood_scores_all = self(batch).to(self.device)
        test_mask = batch.test_mask

        ood_scores_test = ood_scores_all[test_mask]
        y_test = batch.y[test_mask]
        
        targets = (y_test.sum(dim=1) == 0).long()

        self.test_auroc.update(ood_scores_test, targets)
        self.log('test_auroc', self.test_auroc, on_step=False, on_epoch=True, prog_bar=True)

    def configure_optimizers(self): return None