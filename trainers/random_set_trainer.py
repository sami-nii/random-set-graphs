import wandb
import lightning as L
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import EarlyStopping
import os
import sys
import torch
import gc
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.random_set_layer import RandomSetGNN
from dataset_loader.dataset_loader import dataset_loader


def random_set_train(project_name, dataset_name, **kwargs):
    wandb.init(project=project_name, config=kwargs)
    config = wandb.config
    wandb_logger = WandbLogger(project=project_name)
    
    # Load the dataset
    train_loader, val_loader, test_loader = dataset_loader(dataset_name, config)

    focal_sets = [
        {0}, {1}, {2},
        {0,1}, {0,2}, {1,2},
        {0,1,2}
    ]
    # Instantiate the model
    model = RandomSetGNN(
        gnn_type="GCN",
        in_channels=config.in_channels,
        hidden_channels=config.hidden_channels,
        num_layers=config.num_layers,
        focal_sets=focal_sets,
        num_classes=3,
        lr=config.lr,
        weight_decay=config.get("weight_decay", 0.0)
    )

    # Trainer setup
    trainer = L.Trainer(
        devices="auto",
        accelerator="auto",
        logger=wandb_logger,
        max_epochs=config.get("max_epochs", 200),
        log_every_n_steps=1,
        callbacks=[
            EarlyStopping(monitor="val_loss", patience=config.get("patience", 20))
        ]
    )

    # Train and validate the model
    trainer.fit(model, train_loader, val_loader)

    # Test the model
    trainer.test(model, test_loader)

    # Finalize the wandb run
    wandb.finish()
    
    del model, trainer, train_loader, val_loader, test_loader
    gc.collect()
    torch.cuda.empty_cache()