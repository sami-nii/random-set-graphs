import torch
import lightning as L
from torchmetrics import AUROC
import torch.nn.functional as F
import numpy as np
import faiss  # Requires faiss-cpu or faiss-gpu

from models.VanillaGNN import VanillaGNN


class KNNDetector(L.LightningModule):
    """
    kNN-based OOD detector on top of a frozen VanillaGNN.

    - Uses the SECOND-LAST GNN layer (last hidden representation) as embedding f(x):
        * For num_layers >= 2: output of conv[num_layers-2] after activation.
        * For num_layers == 1: falls back to final logits.
    - Builds a FAISS index on TRAIN embeddings (ID nodes).
    - For each node, uses the distance to its k-th nearest neighbor as an OOD score
      (larger distance => more OOD, we log AUROC with 1 = OOD).
    """

    def __init__(self, backbone_ckpt_path: str, k: int = 50):
        super().__init__()
        self.save_hyperparameters()

        # ---- Frozen backbone ----
        self.backbone = VanillaGNN.load_from_checkpoint(backbone_ckpt_path)
        self.backbone.eval()
        for p in self.backbone.parameters():
            p.requires_grad = False

        self.k = int(k)
        self.faiss_index = None

        self.val_auroc = AUROC(task="binary")
        self.test_auroc = AUROC(task="binary")

    # ------------------------------------------------------------------
    # Internal: second-last-layer embeddings
    # ------------------------------------------------------------------
    def _get_embeddings(self, data):
        """Penultimate features of the backbone: the node embedding z_v fed to the classifier."""
        device = next(self.backbone.parameters()).device
        data = data.to(device)
        with torch.no_grad():
            return self.backbone.embed(data)

    # ------------------------------------------------------------------
    # Precomputation: build FAISS index on train embeddings
    # ------------------------------------------------------------------
    def precompute_statistics(self, train_data):
        """
        Pre-computes the FAISS index from the ID training data embeddings.

        Args:
            train_data: PyG Data object with x, edge_index, y, train_mask.
        """
        if not hasattr(train_data, "train_mask"):
            raise RuntimeError("train_data must have a 'train_mask' attribute.")

        print(f"Building FAISS index for k={self.k}...")

        device = next(self.backbone.parameters()).device
        data = train_data.to(device)

        # 1) Embeddings for all nodes
        all_embeddings = self._get_embeddings(data)  # [N, D]

        # 2) Restrict to training nodes
        train_embeddings = all_embeddings[data.train_mask]  # [N_train, D]

        # 3) L2-normalize
        train_embeddings_norm = F.normalize(train_embeddings, p=2, dim=1)

        # 4) Build FAISS index on CPU
        feature_dim = train_embeddings_norm.size(1)
        self.faiss_index = faiss.IndexFlatL2(feature_dim)

        train_np = train_embeddings_norm.detach().cpu().numpy().astype(np.float32)
        self.faiss_index.add(train_np)

        print(f"FAISS index built with {self.faiss_index.ntotal} training vectors (dim={feature_dim}).")

    # ------------------------------------------------------------------
    # Forward: compute kNN-based OOD scores for all nodes
    # ------------------------------------------------------------------
    def forward(self, data):
        """
        Compute kNN-based OOD scores for ALL nodes in the graph.

        Returns:
            scores: [N] where higher values => more ID
                    (we return -distance to the k-th NN).
        """
        if self.faiss_index is None:
            raise RuntimeError(
                "FAISS index not built. Call `precompute_statistics(train_data)` first."
            )

        device = next(self.backbone.parameters()).device
        data = data.to(device)

        # Embeddings from second-last layer
        embeddings = self._get_embeddings(data)          # [N, D]
        embeddings_norm = F.normalize(embeddings, p=2, dim=1)
        embeddings_np = embeddings_norm.detach().cpu().numpy().astype(np.float32)

        # kNN search
        distances, _ = self.faiss_index.search(embeddings_np, self.k)  # [N, k]

        kth_distances = distances[:, -1]  # distance to k-th NN
        ood_scores = -torch.from_numpy(kth_distances).float()  # higher => more ID

        return ood_scores

    # ------------------------------------------------------------------
    # Lightning eval hooks
    # ------------------------------------------------------------------
    def validation_step(self, batch, batch_idx):
        ood_scores_all = self(batch).cpu()
        val_mask = batch.val_mask

        ood_scores_val = ood_scores_all[val_mask]
        y_val = batch.y[val_mask].cpu()

        targets = (y_val.sum(dim=1) == 0).long()  # 1 = OOD, 0 = ID

        self.val_auroc.update(ood_scores_val, targets)
        self.log("val_auroc", self.val_auroc, on_step=False, on_epoch=True, prog_bar=True)

    def test_step(self, batch, batch_idx):
        ood_scores_all = self(batch).cpu()
        test_mask = batch.test_mask

        ood_scores_test = ood_scores_all[test_mask]
        y_test = batch.y[test_mask].cpu()

        targets = (y_test.sum(dim=1) == 0).long()

        self.test_auroc.update(ood_scores_test, targets)
        self.log("test_auroc", self.test_auroc, on_step=False, on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        return None
