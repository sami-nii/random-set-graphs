import wandb
import lightning as L
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import EarlyStopping
from models.VanillaGNN import VanillaGNN
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from dataset_loader.dataset_loader import dataset_loader



def vanilla_train(project_name, dataset_name):
    
    # Initialize wandb at the start of the experiment
    wandb.init(project=project_name)

    config = wandb.config

    wandb_logger = WandbLogger(project=project_name)

    # Instantiate the model
    model = CredalGNN(
        gnn_type=config["gnn_type"],
        in_channels=config["in_channels"],
        out_channels=config["out_channels"],
        hidden_channels=config["hidden_channels"],
        num_layers=config["num_layers"],
        lr=config["lr"],
        weight_decay=config["weight_decay"],
        delta=config["delta"],
    )

    # Trainer setup
    trainer = L.Trainer(
        devices="auto",
        accelerator="auto",
        logger=wandb_logger,
        log_every_n_steps=1,
        callbacks=[
            EarlyStopping(monitor="val_loss", patience=config["patience"]),
        ],
    )

    # Load the dataset
    train_loader, val_loader, test_loader = dataset_loader(dataset_name, config)

    # Train and validate the model
    trainer.fit(model, train_loader, val_loader)

    # Test the model
    trainer.test(model, test_loader)

    # Finalize the wandb run
    wandb.finish()