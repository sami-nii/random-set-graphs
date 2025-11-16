# models/GEBMModule.py
import torch
import lightning as L
import torch.nn.functional as F
from torchmetrics import AUROC, Accuracy, F1Score

from models.VanillaGNN import VanillaGNN

from graph_ebm.graph_uq.gebm import GraphEBMWrapper


class GEBMModule(L.LightningModule):
    """
    Post-hoc, inference-only LightningModule that:
      1) Loads a trained VanillaGNN checkpoint.
      2) Fits GEBM on training nodes using logits+embeddings computed WITH edges.
      3) Evaluates uncertainty on logits+embeddings computed WITHOUT edges,
         while GEBM's own diffusion still uses the real graph.
    Notes:
      - Embeddings are set to logits (VanillaGNN doesn't expose penultimate features).
        This keeps the interface simple and works out-of-the-box.
    """
    def __init__(self, checkpoint_path: str):
        super().__init__()
        self.save_hyperparameters()
        self.model: VanillaGNN = VanillaGNN.load_from_checkpoint(checkpoint_path)
        self.model.freeze()

        self.gebm = GraphEBMWrapper()
        self.fitted = False

        self.C = self.model.C
        self.auroc_metric = AUROC(task="binary")
        self.acc_metric = Accuracy(task="multiclass", num_classes=self.C)
        self.f1_metric = F1Score(task="multiclass", num_classes=self.C)

    @torch.no_grad()
    def _compute_logits(self, data) -> torch.Tensor:
        return self.model(data)

    @staticmethod
    def _onehot_to_indices(y_onehot: torch.Tensor) -> torch.Tensor:
        # Handles mixed ID/OOD labels: OOD rows are all-zero one-hots.
        return torch.argmax(y_onehot, dim=1)

    def test_step(self, batch, batch_idx):
        """
        We assume a single transductive graph batch with train/val/test masks.
        """
        device = batch.x.device

        # --- 1) FIT PHASE (WITH EDGES / network effects) --------------------
        # Compute training-time logits and use them as "embeddings".
        logits_with_edges = self._compute_logits(batch)                      # [N, C]
        embeddings_with_edges = logits_with_edges                            # [N, C] (see note)
        y_indices = self._onehot_to_indices(batch.y)                         # [N]
        mask_train = batch.train_mask                                        # [N] bool

        # Fit GEBM on training nodes; GEBM uses the real graph structure.
        self.gebm.fit(
            logits_with_edges, embeddings_with_edges,
            batch.edge_index, y_indices, mask_train
        )
        self.fitted = True

        # --- 2) EVAL PHASE (WITHOUT EDGES for the base model) ---------------
        # Make an "edge-less" copy for the base model forward.
        # (GEBM diffusion still receives the real edges below.)
        data_no_edges = batch.clone()
        data_no_edges.edge_index = torch.empty((2, 0), dtype=batch.edge_index.dtype, device=device)

        logits_no_edges = self._compute_logits(data_no_edges)                # [N, C]
        embeddings_no_edges = logits_no_edges                                # [N, C]

        # GEBM diffusion leverages the original graph (batch.edge_index).
        uncertainty_all = self.gebm.get_uncertainty(
            logits_unpropagated=logits_no_edges,
            embeddings_unpropagated=embeddings_no_edges,
            edge_index=batch.edge_index,
        )                                                                    # [N]

        # --- 3) Slice to TEST and compute metrics ---------------------------
        test_mask = batch.test_mask
        y_test = batch.y[test_mask].detach()
        logits_test = logits_with_edges[test_mask].detach()  # for ID metrics
        uq_test = uncertainty_all[test_mask].detach()

        # OOD targets per your convention: 1 = OOD, 0 = ID
        ood_targets = 1 - y_test.sum(dim=1)

        # AUROC: higher uncertainty -> more OOD-like
        auroc_gebm = self.auroc_metric(uq_test, ood_targets)

        # Classification metrics on ID only (optional but matches your style)
        id_mask = (y_test.sum(dim=1) == 1)
        if id_mask.any():
            id_logits = logits_test[id_mask]
            id_labels = torch.argmax(y_test[id_mask], dim=1)
            id_preds = torch.argmax(F.softmax(id_logits, dim=1), dim=1)
            id_acc = self.acc_metric(id_preds, id_labels)
            id_f1 = self.f1_metric(id_preds, id_labels)
            self.log("test_acc", id_acc)
            self.log("test_f1", id_f1)

        self.log("auroc_GEBM", auroc_gebm, prog_bar=True)
        return auroc_gebm

    # No training/validation; this is post-hoc
    def training_step(self, *args, **kwargs): pass
    def validation_step(self, *args, **kwargs): pass
    def configure_optimizers(self): return None
