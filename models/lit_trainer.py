import subprocess 
import wandb
import lightning as L
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lit_wrapper import CredalGNN
import os
import sys
from torch_geometric.loader import DataLoader
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from dataset_loader.dataset_loader import dataset_loader
from sweeps.sweeps import sweep_cora, sweep_ogb_arxiv_year, sweep_squirrel, sweep_chameleon, sweep_snap_patents
import argparse

# Set the project name for wandb
PROJECT_NAME = "graph-uncertainty"

# Set the dataset name with argument parsing
parser = argparse.ArgumentParser(description="Run a sweep for a specific dataset.")

parser.add_argument(
    "-d", "--dataset",
    type=str,
    choices=["cora", "ogb_arxiv_year", "squirrel", "chameleon", "snap_patents"],
    default="squirrel",
    help="Dataset to run the sweep on.",
)

args = parser.parse_args()

DATASET = args.dataset


def train():
    # Initialize wandb at the start of the experiment
    wandb.init(project=PROJECT_NAME)

    config = wandb.config

    wandb_logger = WandbLogger(project=PROJECT_NAME)

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
            # ModelCheckpoint(monitor="val_loss", save_top_k=1, mode="min", filename="{epoch:02d}-{val_loss:.4f}"),
        ],
        limit_train_batches=config["batch_size"],
        limit_val_batches=config["batch_size"],
        limit_test_batches=config["batch_size"],
    )

    # Load the dataset
    train_loader, val_loader, test_loader = dataset_loader(DATASET, config)

    # Train and validate the model
    trainer.fit(model, train_loader, val_loader)

    # Test the model
    trainer.test(model, test_loader)

    # Finalize the wandb run
    wandb.finish()

# Execute the experiment
if __name__ == "__main__":
    if DATASET == "cora":
        sweep = sweep_cora
    elif DATASET == "ogb_arxiv_year":
        sweep = sweep_ogb_arxiv_year
    elif DATASET == "squirrel":
        sweep = sweep_squirrel
    elif DATASET == "chameleon":
        sweep = sweep_chameleon
    elif DATASET == "snap_patents":
        sweep = sweep_snap_patents
    else:
        raise ValueError(f"Unsupported dataset: {DATASET}")

    sweep_id = wandb.sweep(sweep=sweep, project=PROJECT_NAME)
    wandb.agent(sweep_id, function=train)

    
    
