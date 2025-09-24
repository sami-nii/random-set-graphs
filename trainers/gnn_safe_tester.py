# trainers/gnnsafe_tester.py

import wandb
import lightning as L
from lightning.pytorch.loggers import WandbLogger
import os
import sys

# --- Add project root to path ---
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# --- Import your custom modules ---
from models.gnnsafe_detector import GNNSafeDetector
from dataset_loader.dataset_loader import dataset_loader
from utils.model_manager import search_best_model

def gnnsafe_test(project_name, dataset_name, save_path, **kwargs):
    """
    Tests a GNNSafeDetector, which uses energy-based belief propagation.
    """
    wandb.init(project=project_name, job_type="test_gnnsafe")
    config = wandb.config

    # --- 1. Find the Best Backbone Checkpoint ---
    vanilla_model_dir = os.path.join(save_path, f"vanilla_{dataset_name}")
    backbone_ckpt_path = search_best_model(vanilla_model_dir)
    if not backbone_ckpt_path:
        raise FileNotFoundError(f"No backbone checkpoint found in {vanilla_model_dir}")
    print(f"Using backbone checkpoint: {backbone_ckpt_path}")

    # --- 2. Instantiate the GNNSafe Detector with Hyperparameters ---
    gnnsafe_model = GNNSafeDetector(
        backbone_ckpt_path=backbone_ckpt_path,
        K=config.get("K", 2),
        alpha=config.get("alpha", 0.5)
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
    _, val_loader, _, test_loader = dataset_loader(
        dataset_name, config={}, ood_test=True, batch_size=1, shuffle=False
    )
    
    # --- 5. Run Validation and Test ---
    print(f"\n--- Validating GNNSafe on {dataset_name} ---")
    trainer.validate(gnnsafe_model, val_loader)

    print(f"\n--- Testing GNNSafe on {dataset_name} ---")
    trainer.test(gnnsafe_model, test_loader)

    wandb.finish()