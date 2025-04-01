import subprocess 
import wandb
import lightning as L
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lit_wrapper import LitGraphNN
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.dataset_loader import dataset_loader

# Set the project name for wandb
PROJECT_NAME = "graph-uncertainty"


def train():
    # Initialize wandb at the start of the experiment
    wandb.init(project=PROJECT_NAME)

    config = wandb.config

    wandb_logger = WandbLogger(project=PROJECT_NAME)

    # Instantiate the model
    model = LitGraphNN(
        gnn_type=config["gnn_type"],
        in_channels=config["in_channels"],
        out_channels=config["out_channels"],
        hidden_channels=config["hidden_channels"],
        num_layers=config["num_layers"],
        lr=config["lr"],
        weight_decay=config["weight_decay"]
    )

    # Trainer setup
    trainer = L.Trainer(
        devices="auto",
        accelerator="auto",
        logger=wandb_logger,
        callbacks=[
            EarlyStopping(monitor="val_loss", patience=config["patience"]),
            # ModelCheckpoint(monitor="val_loss", save_top_k=1, mode="min", filename="{epoch:02d}-{val_loss:.4f}"),
        ],
    )

    # Load the dataset
    train_loader, val_loader, test_loader = dataset_loader("cora")

    # Train and validate the model
    trainer.fit(model, train_loader, val_loader)

    # Run final validation to log the best accuracy
    val_acc = trainer.validate(model, val_loader)[0]["val_acc"] # TODO load the best model instead of the last one
    wandb_logger.log_metrics({"sweep_score": float(val_acc)})  

    # Run testing and log results
    test_results = trainer.test(model, test_loader)
    wandb_logger.log_metrics({"test_loss": test_results[0]["test_loss"], "test_acc": test_results[0]["test_acc"]})

    # Finalize the wandb run
    wandb.finish()

# Execute the experiment
if __name__ == "__main__":
    from sweeps.sweep_cora import sweep_cora
    sweep_id = wandb.sweep(sweep=sweep_cora, project=PROJECT_NAME)
    wandb.agent(sweep_id, function=train)

    
    
