import wandb
import lightning as L
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import EarlyStopping
import os
import sys
import torch
import gc
import itertools # <--- Added for power set generation
from torch_geometric.utils import subgraph
from .budgeting import train_embeddings, fit_gmm, overlaps, ellipse
from utils.wandb_utils import init_wandb_run


# Adjust import paths as needed
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.random_set_gnn import RandomSetGNN
from dataset_loader.dataset_loader import dataset_loader

def random_set_train(project_name, dataset_name, **kwargs):
    
    # Initialize WandB
    init_wandb_run(project=project_name, config=kwargs)
    config = wandb.config
    wandb_logger = WandbLogger(experiment=wandb.run)

    # torch.set_float32_matmul_precision('medium')
    
    # 1. Load Dataset
    # Assumes transductive setting where loaders return the same graph object
    train_loader, val_loader, test_loader = dataset_loader(dataset_name, config)

    def _get_base_data(loader):
        if hasattr(loader, "data") and hasattr(loader.data, "x") and hasattr(loader.data, "y"):
            return loader.data

        dataset = getattr(loader, "dataset", None)
        if dataset is not None:
            try:
                if len(dataset) == 1:
                    candidate = dataset[0]
                    if hasattr(candidate, "x") and hasattr(candidate, "y"):
                        return candidate
            except Exception:
                pass

        return None

    base_data = _get_base_data(train_loader)

    try:
        print(f"Number of batches per epoch: {len(train_loader)}")
    except Exception as exc:
        print(f"Skipping batch-count precomputation: {exc}")

    # Prefer reading metadata from the base graph to avoid triggering an eager
    # sampled batch on large NeighborLoader datasets.
    if base_data is None:
        print("Falling back to a sampled batch to recover train-loader metadata.")
        data_sample = next(iter(train_loader))
        base_data = data_sample
    else:
        data_sample = base_data

    num_features = base_data.x.shape[1]

    # If y is one-hot [Nodes, Classes], shape[1] is the number of classes.
    if base_data.y.dim() > 1 and base_data.y.size(1) > 1:
        num_id_classes = base_data.y.shape[1]
    else:
        train_mask = base_data.train_mask.bool() if hasattr(base_data, "train_mask") else torch.ones(base_data.y.size(0), dtype=torch.bool)
        num_id_classes = int(base_data.y[train_mask].max().item()) + 1
    
    # 2. Generate Full Power Set (2^N - 1)
    # We generate all non-empty subsets of the ID classes.
    # e.g., for 3 classes: {0}, {1}, {2}, {0,1}, {0,2}, {1,2}, {0,1,2}

    # warning if num_id_classes is large

    # 2. (edited) Generate Focal Sets (Power Set OR Budgeted)
    class_indices = list(range(num_id_classes))

    print(f"--- RS-NN Configuration ---")
    print(f"Number of ID Classes: {num_id_classes}")

    if num_id_classes >= 10:
        print(
            f"Skipping full power set construction for {num_id_classes} ID classes; "
            "using budgeted focal sets instead."
        )
        focal_sets = [{c} for c in class_indices]
    else:
        focal_sets_tuples = itertools.chain.from_iterable(
            itertools.combinations(class_indices, r) for r in range(1, num_id_classes + 1)
        )
        
        # Convert tuples to python sets
        focal_sets = [set(x) for x in focal_sets_tuples]
        
        print(f"Focal Sets Strategy: Full Power Set")
        print(f"Total Output Heads (2^N - 1): {len(focal_sets)}")

    
    # Budget Override
    if num_id_classes >= 10:
           print("Constructing budgeted focal sets")
           aux_model = RandomSetGNN(
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
           ).gnn_model
           device = 'cuda' if torch.cuda.is_available() else 'cpu'

           base_data = train_loader.data if hasattr(train_loader, "data") else data_sample
           if base_data.y.dim() > 1:
               is_id = base_data.y.sum(dim=1) == 1
               y_indices = base_data.y.argmax(dim=-1)
           else:
               is_id = torch.ones_like(base_data.y, dtype=torch.bool)
               y_indices = base_data.y

           train_mask = base_data.train_mask.bool()
           budget_mask = train_mask & is_id
           budget_samples_per_class = int(config.get("budget_samples_per_class", 200))

           print(
               f"Sampling up to {budget_samples_per_class} training nodes per class "
               "for budgeted focal set construction."
           )

           sampled_node_indices = []
           found_classes = []
           for class_idx in range(num_id_classes):
               class_nodes = torch.nonzero(budget_mask & (y_indices == class_idx), as_tuple=False).view(-1)
               if class_nodes.numel() == 0:
                   continue
               if class_nodes.numel() > budget_samples_per_class:
                   perm = torch.randperm(class_nodes.numel())[:budget_samples_per_class]
                   class_nodes = class_nodes[perm]
               sampled_node_indices.append(class_nodes)
               found_classes.append(class_idx)

           if len(sampled_node_indices) != num_id_classes:
               missing_classes = sorted(set(range(num_id_classes)) - set(found_classes))
               raise ValueError(
                   "Budgeted focal set construction could not find training samples "
                   f"for all ID classes. Missing classes: {missing_classes}"
               )

           sampled_node_indices = torch.cat(sampled_node_indices).cpu()
           sampled_node_mask = torch.zeros(base_data.num_nodes, dtype=torch.bool)
           sampled_node_mask[sampled_node_indices] = True

           sampled_edge_index, _ = subgraph(
               sampled_node_mask,
               base_data.edge_index,
               relabel_nodes=True,
           )
           sampled_x = base_data.x[sampled_node_mask].to(device)
           y_train = y_indices[sampled_node_mask]

           print(
               f"Budgeting subgraph has {sampled_x.size(0)} nodes and "
               f"{sampled_edge_index.size(1)} edges."
           )
           print("Computing auxiliary embeddings for budgeting...")
           train_embedded = train_embeddings(
               aux_model,
               sampled_x,
               batch_size=config.get("batch_size", 256),
               device=device,
               edge_index=sampled_edge_index.to(device),
           )
           
           # 2. Fit GMMs
           classes = list(range(num_id_classes))
           print("Fitting Gaussian mixtures for budgeted focal sets...")
           individual_gms = fit_gmm(classes, train_embedded, y_train)
           
           # 3. Ellipsoids
           print("Computing class ellipsoids...")
           regions, means, max_len = ellipse(individual_gms, len(classes), device=device)
           
           print("Computing focal set overlaps...")
           budget_max_cardinality = int(config.get("budget_max_cardinality", 3))
           print(f"Limiting overlap search to focal sets of size <= {budget_max_cardinality}.")
           focal_sets = overlaps(
                k=config.get("budget_k", 32),
                classes=[str(c) for c in classes],
                num_clusters=num_id_classes,
                classes_dict={str(c): c for c in classes},
                regions=regions,
                means=means,
                max_len=max_len,
                max_cardinality=budget_max_cardinality,
                )
           focal_sets = [set(int(c) for c in s) for s in focal_sets]
           
           print(f"Budgeted focal sets size: {len(focal_sets)}")


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
    uses_sampled_batches = int(config.get("batch_size", -1)) > 0
    sanity_val_steps = 0 if uses_sampled_batches else 2
    if sanity_val_steps == 0:
        print(
            "Skipping Lightning sanity validation for sampled mini-batch runs "
            "to avoid large pre-training validation passes on datasets such as Reddit2."
        )

    trainer = L.Trainer(
        devices="auto",
        accelerator="auto",
        logger=wandb_logger,
        max_epochs=config.get("max_epochs", 200),
        log_every_n_steps=1,
        num_sanity_val_steps=sanity_val_steps,
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

