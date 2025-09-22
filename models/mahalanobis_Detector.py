# models/mahalanobis_detector.py

import torch
import lightning as L
from torchmetrics import AUROC
from models.VanillaGNN import VanillaGNN

class MahalanobisDetector(L.LightningModule):
    def __init__(self, backbone_ckpt_path: str, noise_magnitude: float = 0.0):
        super().__init__()
        self.save_hyperparameters()

        self.backbone = VanillaGNN.load_from_checkpoint(backbone_ckpt_path)
        self.backbone.eval()
        for param in self.backbone.parameters():
            param.requires_grad = False
        
        self.noise_magnitude = noise_magnitude
        self.num_classes = self.backbone.out_channels
        self.class_means = None
        self.shared_covariance_inv = None
        
        # Create separate metric instances for validation and test
        self.val_auroc = AUROC(task="binary")
        self.test_auroc = AUROC(task="binary")

    def precompute_statistics(self, train_data):
        """
        Computes class means and shared covariance from the ID training data.
        This must be called manually before validation or testing.
        """
        print("Pre-computing Mahalanobis statistics...")
        data = train_data.to(self.device)
        self.backbone.to(self.device)
        # ... (The rest of the pre-computation logic remains exactly the same) ...
        with torch.no_grad():
            all_embeddings = self.backbone.get_embeddings(data)
        
        train_mask = data.train_mask
        train_embeddings = all_embeddings[train_mask]
        train_labels = data.y[train_mask]
        
        if train_labels.ndim > 1:
            train_labels = torch.argmax(train_labels, dim=1)

        feature_dim = train_embeddings.shape[1]
        self.class_means = torch.zeros(self.num_classes, feature_dim, device=self.device)
        # ... (rest of the logic)
        for c in range(self.num_classes):
            class_mask = (train_labels == c)
            if class_mask.sum() > 0: self.class_means[c] = train_embeddings[class_mask].mean(dim=0)
        
        shared_covariance = torch.zeros(feature_dim, feature_dim, device=self.device)
        for c in range(self.num_classes):
            class_mask = (train_labels == c)
            if class_mask.sum() > 0:
                diff = train_embeddings[class_mask] - self.class_means[c]
                shared_covariance += diff.t() @ diff
        
        shared_covariance /= train_mask.sum()
        self.shared_covariance_inv = torch.linalg.pinv(shared_covariance)
        print("Mahalanobis statistics pre-computation complete.")

    def forward(self, data):
        # ... (The forward pass with optional input pre-processing remains the same) ...
        if self.class_means is None: raise RuntimeError("Statistics not computed.")
        
        if self.noise_magnitude > 0:
            x_perturbed = data.x.clone().detach().requires_grad_(True)
            # ... (full input pre-processing logic) ...
            embeddings = self.backbone.get_embeddings(x_perturbed, data.edge_index)
            # ... (backward pass, etc.)
            final_embeddings = self.backbone.get_embeddings(data.x - self.noise_magnitude * torch.sign(x_perturbed.grad.data), data.edge_index)
        else:
            final_embeddings = self.backbone.get_embeddings(data.x, data.edge_index)

        diff = final_embeddings.unsqueeze(1) - self.class_means.unsqueeze(0)
        left_term = torch.einsum('ncd,df->ncf', diff, self.shared_covariance_inv)
        mahalanobis_sq = torch.einsum('ncf,ncf->nc', left_term, diff)
        min_dist, _ = torch.min(mahalanobis_sq, dim=1)
        return -min_dist

    def validation_step(self, batch, batch_idx):
        """
        Evaluates the detector on the validation set.
        This is used by the wandb sweep to find the best hyperparameters.
        """
        ood_scores_all = self(batch).cpu()
        val_mask = batch.val_mask
        
        ood_scores_val = ood_scores_all[val_mask]
        y_val = batch.y[val_mask]

        # OOD labels are 1, ID labels are 0
        targets = (y_val.sum(dim=1) == 0).long()

        self.val_auroc.update(ood_scores_val, targets)
        self.log('val_auroc', self.val_auroc, on_step=False, on_epoch=True, prog_bar=True)

    def test_step(self, batch, batch_idx):
        """
        Evaluates the final, best model on the test set.
        This provides the final score for the paper.
        """
        ood_scores_all = self(batch).cpu()
        test_mask = batch.test_mask

        ood_scores_test = ood_scores_all[test_mask]
        y_test = batch.y[test_mask]
        
        targets = (y_test.sum(dim=1) == 0).long()

        self.test_auroc.update(ood_scores_test, targets)
        self.log('test_auroc', self.test_auroc, on_step=False, on_epoch=True, prog_bar=True)

    def configure_optimizers(self): return None