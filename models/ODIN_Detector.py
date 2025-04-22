import torch
import lightning as L
import torch.nn.functional as F
from torchmetrics import AUROC, Accuracy, F1Score
from models.VanillaGNN import VanillaGNN

class ODINDetector(L.LightningModule):
    def __init__(self, backbone_ckpt_path, temperature=1.0, noise_magnitude=0.0):
        super().__init__()

        # Load and freeze trained backbone
        self.backbone = VanillaGNN.load_from_checkpoint(backbone_ckpt_path)
        self.backbone.eval()
        for param in self.backbone.parameters():
            param.requires_grad = False

        # OOD method hyperparameters
        self.temperature = temperature
        self.noise_magnitude = noise_magnitude

        # Metrics
        self.auroc = AUROC(task="binary")
        self.accuracy = Accuracy(task="multiclass", num_classes=self.backbone.C)
        self.f1 = F1Score(task="multiclass", num_classes=self.backbone.C)

        # Save hyperparameters (wandb sweep will control these)
        self.save_hyperparameters(ignore=['backbone'])

    def train_step(self, batch, batch_idx):

        # Extract logits from backbone
        logits = self.backbone(batch)

        # ODIN logic (example implementation)
        scaled_logits = logits / self.temperature
        pred_probs = F.softmax(scaled_logits, dim=1)

        # Compute OOD scores based on maximum softmax probability
        max_softmax_scores, pred_classes = pred_probs.max(dim=1)

        # OOD thresholding (example, based on hyperparameter)
        ood_mask = max_softmax_scores < self.noise_magnitude

        # Ground-truth OOD labels (assuming your batch.y is one-hot, where all zeros = OOD)
        target_ood = torch.sum(batch.y, dim=1).bool()

        # Compute metrics
        auroc_score = self.auroc(max_softmax_scores, target_ood)
        accuracy_score = self.accuracy(pred_classes[~ood_mask], torch.argmax(batch.y[~ood_mask], dim=1))
        f1_score = self.f1(pred_classes[~ood_mask], torch.argmax(batch.y[~ood_mask], dim=1))

        batch_size = batch.y.size(0)

        # Logging
        self.log("test_auroc", auroc_score, batch_size=batch_size)
        self.log("test_accuracy", accuracy_score, batch_size=batch_size)
        self.log("test_f1", f1_score, batch_size=batch_size)

        return auroc_score
    
    def test_step(self, batch, batch_idx):
        pass
        # the same
