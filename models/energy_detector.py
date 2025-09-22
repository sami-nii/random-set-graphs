# models/energy_detector.py

import torch
import lightning as L
from torchmetrics import AUROC
from models.VanillaGNN import VanillaGNN # Assuming your VanillaGG is here

class EnergyDetector(L.LightningModule):
    def __init__(self, backbone_ckpt_path: str):
        super().__init__()
        self.save_hyperparameters()

        # Load and freeze the pre-trained backbone
        self.backbone = VanillaGNN.load_from_checkpoint(backbone_ckpt_path)
        self.backbone.eval()
        for param in self.backbone.parameters():
            param.requires_grad = False
        
        # Metrics for validation and testing
        self.val_auroc = AUROC(task="binary")
        self.test_auroc = AUROC(task="binary")

    def forward(self, data):
        """
        Calculates the energy score for all nodes in the data object.
        This corresponds to Equation (4) in Liu et al. (2020).
        The OOD score is the negative energy score.
        """
        with torch.no_grad():
            # The energy score is based on the pre-softmax logits
            logits = self.backbone(data)

        # Energy score E(x) = -log(sum_c(exp(logit_c(x))))
        # A higher energy score means more likely to be OOD.
        energy_scores = -torch.logsumexp(logits, dim=1)

        # For OOD detection, a higher score should indicate in-distribution.
        # Since lower energy means more ID, we use the negative energy as the OOD score.
        ood_scores = -energy_scores

        return ood_scores

    def validation_step(self, batch, batch_idx):
        """
        Evaluates the detector on the validation set.
        This is used by the wandb sweep to find the best hyperparameters for other models,
        but for Energy, it's just for logging as there's nothing to tune.
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
        Evaluates the final model on the test set.
        """
        ood_scores_all = self(batch).cpu()
        test_mask = batch.test_mask

        ood_scores_test = ood_scores_all[test_mask]
        y_test = batch.y[test_mask]
        
        targets = (y_test.sum(dim=1) == 0).long()

        self.test_auroc.update(ood_scores_test, targets)
        self.log('test_auroc', self.test_auroc, on_step=False, on_epoch=True, prog_bar=True)

    def configure_optimizers(self): return None