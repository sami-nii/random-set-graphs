# trainers/mahalanobis_tester.py

import wandb
import lightning as L
from lightning.pytorch.loggers import WandbLogger
import os
import sys

# --- Add project root to path ---
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# --- Import your custom modules ---
from models.mahalanobis_Detector import MahalanobisDetector
from dataset_loader.dataset_loader import dataset_loader
from utils.model_manager import search_best_model

def mahalanobis_test(project_name, dataset_name, save_path, **kwargs):
    wandb.init(project=project_name, job_type="test_mahalanobis")
    config = wandb.config

    # --- 1. Find the Best Backbone Checkpoint ---
    vanilla_model_dir = os.path.join(save_path, f"vanilla_{dataset_name}")
    backbone_ckpt_path = search_best_model(vanilla_model_dir)
    if not backbone_ckpt_path:
        raise FileNotFoundError(f"No backbone checkpoint found in {vanilla_model_dir}")
    print(f"Using backbone checkpoint: {backbone_ckpt_path}")

    # --- 2. Instantiate the Mahalanobis Model with Hyperparameters ---
    mahalanobis_model = MahalanobisDetector(
        backbone_ckpt_path=backbone_ckpt_path,
        noise_magnitude=config.get("noise_magnitude", 0.0) # From sweep config
    )

    # --- 3. Set up Trainer ---
    wandb_logger = WandbLogger(project=project_name)
    trainer = L.Trainer(
        devices="auto",
        accelerator="auto",
        logger=wandb_logger,
        enable_checkpointing=False,
        log_every_n_steps=1,
    )

    # --- 4. Load Datasets ---
    # We need the train_loader for pre-computation, val_loader for tuning, and test_loader for final eval.
    # The loader must return the full graph in a single batch.
    train_loader, val_loader, _, test_loader = dataset_loader(
        dataset_name, config={}, ood_test=True, batch_size=1, shuffle=False
    )
    
    # --- 5. Pre-compute Statistics ---
    print(f"\n--- Pre-computing Mahalanobis statistics on {dataset_name} training data ---")
    mahalanobis_model.precompute_statistics(train_loader.dataset.data)
    
    # --- 6. Run Validation for Hyperparameter Tuning ---
    print(f"\n--- Validating Mahalanobis on {dataset_name} ---")
    trainer.validate(mahalanobis_model, val_loader)

    # --- 7. Run the Final Test ---
    print(f"\n--- Testing Mahalanobis on {dataset_name} ---")
    trainer.test(mahalanobis_model, test_loader)

    wandb.finish()