import wandb
import lightning as L
from lightning.pytorch.loggers import WandbLogger
import os
import sys

# --- Add project root to path ---
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# --- Import your custom modules ---
from models.knn_detector import KNNDetector
from dataset_loader.dataset_loader import dataset_loader
from utils.model_manager import search_best_model

def knn_test(project_name, dataset_name, save_path, **kwargs):
    """
    Tests a KNNDetector, a post-hoc method that uses k-th nearest neighbor
    distance in the latent space for OOD detection.
    """
    wandb.init(project=project_name, job_type="test_knn")
    config = wandb.config

    # --- 1. Find the Best Backbone Checkpoint ---
    vanilla_model_dir = os.path.join(save_path, f"vanilla_{dataset_name}")
    backbone_ckpt_path = search_best_model(vanilla_model_dir)
    if not backbone_ckpt_path:
        raise FileNotFoundError(f"No backbone checkpoint found in {vanilla_model_dir}")
    print(f"Using backbone checkpoint: {backbone_ckpt_path}")

    # --- 2. Instantiate the KNN Detector with Hyperparameters ---
    knn_model = KNNDetector(
        backbone_ckpt_path=backbone_ckpt_path,
        k=config.get("k", 50) # Get k from the sweep config
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
    train_loader, val_loader, _, test_loader = dataset_loader(
        dataset_name, config={}, ood_test=True, batch_size=1, shuffle=False
    )
    
    # --- 5. Pre-compute Statistics (Build Faiss Index) ---
    print(f"\n--- Pre-computing KNN index on {dataset_name} training data ---")
    knn_model.precompute_statistics(train_loader.dataset.data)
    
    # --- 6. Run Validation for Hyperparameter Tuning ---
    print(f"\n--- Validating KNN on {dataset_name} ---")
    trainer.validate(knn_model, val_loader)

    # --- 7. Run the Final Test ---
    print(f"\n--- Testing KNN Detector on {dataset_name} ---")
    trainer.test(knn_model, test_loader)

    wandb.finish()