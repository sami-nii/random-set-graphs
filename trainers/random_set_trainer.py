import wandb
import lightning as L
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import EarlyStopping
import os
import sys
import torch
import gc
import itertools # <--- Added for power set generation

# Adjust import paths as needed
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.random_set_gnn import RandomSetGNN
from dataset_loader.dataset_loader import dataset_loader

def random_set_train(project_name, dataset_name, **kwargs):
    
    # Initialize WandB
    wandb.init(project=project_name, config=kwargs)
    config = wandb.config
    wandb_logger = WandbLogger(project=project_name)

    # torch.set_float32_matmul_precision('medium')
    
    # 1. Load Dataset
    # Assumes transductive setting where loaders return the same graph object
    train_loader, val_loader, test_loader = dataset_loader(dataset_name, config)
    print(f"Number of batches per epoch: {len(train_loader)}")
    
    # Get metadata from the first batch
    data_sample = next(iter(train_loader))
    num_features = data_sample.x.shape[1]
    
    # Determine number of ID classes
    # If y is one-hot [Nodes, Classes], shape[1] is the number of classes.
    num_id_classes = data_sample.y.shape[1]
    
    # 2. Generate Full Power Set (2^N - 1)
    # We generate all non-empty subsets of the ID classes.
    # e.g., for 3 classes: {0}, {1}, {2}, {0,1}, {0,2}, {1,2}, {0,1,2}

    # warning if num_id_classes is large
    if num_id_classes > 10:
        print(f"Warning: Number of ID classes is {num_id_classes}, generating full power set may be computationally expensive.")
    
    class_indices = list(range(num_id_classes))
    focal_sets_tuples = itertools.chain.from_iterable(
        itertools.combinations(class_indices, r) for r in range(1, num_id_classes + 1)
    )
    
    # Convert tuples to python sets
    focal_sets = [set(x) for x in focal_sets_tuples]
    
    print(f"--- RS-NN Configuration ---")
    print(f"Number of ID Classes: {num_id_classes}")
    print(f"Focal Sets Strategy: Full Power Set")
    print(f"Total Output Heads (2^N - 1): {len(focal_sets)}")
    
    # 3. Instantiate the Model
    model = RandomSetGNN(
        gnn_type=config.get("gnn_type", "GCN"),
        in_channels=num_features,
        hidden_channels=config.get("hidden_channels", 64),
        num_layers=config.get("num_layers", 2),
        focal_sets=focal_sets,     
        num_classes=num_id_classes,
        lr=config.get("lr", 0.001),
        weight_decay=config.get("weight_decay", 1e-4),
        alpha=config.get("alpha", 1e-3),
        beta=config.get("beta", 1e-3)
    )

    # 4. Trainer Setup
    trainer = L.Trainer(
        devices="auto",
        accelerator="auto",
        logger=wandb_logger,
        max_epochs=config.get("max_epochs", 200),
        log_every_n_steps=1,
        callbacks=[
            EarlyStopping(monitor="val_loss", patience=config.get("patience", 50), mode="min")
        ]
    )

    # 5. Execution
    trainer.fit(model, train_loader, val_loader)
    trainer.test(model, test_loader)

    # 6. Cleanup
    wandb.finish()
    
    del model, trainer, train_loader, val_loader, test_loader
    gc.collect()
    torch.cuda.empty_cache()