from typing import Optional, Union, List, Tuple, Dict
import torch
import wandb
from torch_geometric.loader import DataLoader
import lightning as L
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpointp
from lit_wrapper import LitGraphNN


# torch.set_float32_matmul_precision("high")

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default="sssp")
parser.add_argument("--node_level_task", type=bool, default=True)
parser.add_argument("--alpha", type=float, default=None)
parser.add_argument("--node_layer", type=str, default="GCNConv")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument('--search_method', type=str, default='bayes')
parser.add_argument('--project_name', type=str, default='graph-uncertainty')

args = parser.parse_args()

TASK = args.task
NODE_LEVEL_TASK = args.node_level_task
ALPHA = args.alpha
NODE_LAYER = args.node_layer
SEARCH_METHOD = args.search_method
PROJECT_NAME = args.project_name

EXP_CONFIG = {
    "seed": args.seed,
    "batch_size": args.batch_size,
    "conf_id": wandb.util.generate_id(),
}


def run_experiment():
    """Train and validate the model"""
    run = wandb.init(project=PROJECT_NAME)
    config = dict(run.config)

    # manually added
    config["conf_id"] = EXP_CONFIG["conf_id"]
    config["seed"] = EXP_CONFIG["seed"]
    config["node_level_task"] = NODE_LEVEL_TASK
    config["alpha"] = ALPHA
    config["conv_layer"] = NODE_LAYER

    L.seed_everything(EXP_CONFIG["seed"])

    batch_size = EXP_CONFIG["batch_size"]
    
    data_train, data_val, data_test, num_feat, num_class = get_dataset(
        root="./data/",
        task=TASK,  # TODO: Add from config
        pre_transform=None,  # TODO: Add pre_transform
    )

    # data_train = convert_to_lit_dataset(data_train)
    # data_val   = convert_to_lit_dataset(data_val)
    # data_test  = convert_to_lit_dataset(data_test)

    config["input_dim"] = num_feat
    config["output_dim"] = num_class

    train_dataloader = DataLoader(data_train, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(data_val, batch_size=batch_size, shuffle=False)
    test_dataloader = DataLoader(data_test, batch_size=batch_size, shuffle=False)

    wandb_logger = WandbLogger(
        project="lrgb2",
        name=f"experiment_{config['conf_id']}",
        config=config,
        save_dir="/dev/shm/vornao",
    )

    # log hyperparameters
    wandb_logger.log_hyperparams(config)

    model = LitGraphNN(GNN_GraphProp, config)

    trainer = L.Trainer(
        max_epochs=1000,
        devices=1,
        #strategy="ddp_spawn",
        logger=wandb_logger,
        callbacks=[
            EarlyStopping(monitor="val_loss", patience=10),
            ModelCheckpoint(monitor="val_loss", save_top_k=1),
        ],
    )

    trainer.fit(model, train_dataloader, val_dataloader)
    val_loss = trainer.validate(model, val_dataloader)[0]["val_loss"]

    wandb_logger.log_metrics({"sweep_score": float(val_loss)})
    wandb.finish()

