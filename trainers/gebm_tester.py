
import wandb
import lightning as L
from lightning.pytorch.loggers import WandbLogger
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.GEBM_detector import GEBMModule  
from dataset_loader.dataset_loader import dataset_loader
from utils.model_manager import find_best_checkpoints

def gebm_test(project_name, dataset_name, **kwargs):
    """
    Tests GEBM as a post-hoc uncertainty estimator on a single best VanillaGNN.
    - Fits GEBM on training nodes (with edges).
    - Evaluates uncertainty on base-model outputs computed without edges.
    """
    wandb.init(project=project_name, job_type="test_gebm")
    config = wandb.config

    # 1) Pick best VanillaGNN checkpoint
    ckpt_path = find_best_checkpoints(dataset_name, num_models=1)[0]

    # 2) Wrap it into our GEBM LightningModule
    model = GEBMModule(checkpoint_path=ckpt_path)

    # 3) Get loaders (we only need test_loader; the batch carries all masks)
    _, _, test_loader = dataset_loader(dataset_name, config)

    # 4) Standard Lightning test run
    wandb_logger = WandbLogger(project=project_name)
    trainer = L.Trainer(
        devices="auto",
        accelerator="auto",
        logger=wandb_logger,
        log_every_n_steps=1,
    )

    print(f"\n--- Testing GEBM on {dataset_name} ---")
    trainer.test(model, dataloaders=test_loader)

    wandb.finish()
    print("GEBM testing complete.")
