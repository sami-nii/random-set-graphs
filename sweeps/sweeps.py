

# --- Small Datasets (Full-Batch Training) ---
# batch_size = -1 signals the loader to use the full graph.
metadata_chameleon = {
    "in_channels": {"values": [2325]}, 
    "out_channels": {"values": [3]},
    "batch_size": {"values": [-1]},
    "num_neighbors": {"values": [-1]},
}
metadata_squirrel  = {
    "in_channels": {"values": [2089]}, 
    "out_channels": {"values": [3]},
    "batch_size": {"values": [-1]},
    "num_neighbors": {"values": [-1]},
}
metadata_cora      = {
    "in_channels": {"values": [1433]}, 
    "out_channels": {"values": [3]},
    "batch_size": {"values": [-1]},
    "num_neighbors": {"values": [-1]},
}


metadata_patents   = {
    "in_channels": {"values": [269]},  
    "out_channels": {"values": [3]},
    "batch_size": {"values": [256]},
    "num_neighbors": {"values": [10]},
}
metadata_arxiv     = {
    "in_channels": {"values": [128]},  
    "out_channels": {"values": [3]},
    "batch_size": {"values": [256]},
    "num_neighbors": {"values": [10]},
}
metadata_reddit2   = {
    "in_channels": {"values": [602]},  
    "out_channels": {"values": [30]},
    "batch_size": {"values": [256]},
    "num_neighbors": {"values": [10]},
}
metadata_coauthor  = {
    "in_channels": {"values": [6805]}, 
    "out_channels": {"values": [11]},
    "batch_size": {"values": [-1]},
    "num_neighbors": {"values": [10]},
}
metadata_amazon_ratings = {
    "in_channels":  {"values": [300]},  
    "out_channels": {"values": [3]},    
    "batch_size":   {"values": [-1]},   
    "num_neighbors":{"values": [10]},
}
metadata_roman_empire = {
    "in_channels":  {"values": [300]},  
    "out_channels": {"values": [13]},   # ID classes = {5..17}
    "batch_size":   {"values": [-1]},   
    "num_neighbors":{"values": [10]},
}

metadata_road = {
    "in_channels":  {"values": [4]},   # x, y, width, height
    "out_channels": {"values": [3]},   # 3 ID classes
    "batch_size": {"values": [-1]},
    "num_neighbors": {"values": [-1]},  # full-batch
}


sweep_vanilla = {
    "method": "bayes",
    "metric": {"name": "val_f1", "goal": "maximize"},
    "parameters": {
        "lr": {"distribution": "uniform", "min": 1e-5, "max": 1e-1},
        "hidden_channels": {"values": [64, 128, 256]},
        "num_layers": {"values": [2, 3]},
        "weight_decay": {"distribution": "uniform", "min": 1e-7, "max": 1e-1},
        "gnn_type": {"values": ["GCN", "SAGE"]},
        "patience": {"values": [30]},
    }
}

sweep_ensemble = {
    "method": "grid", 
    "metric": {
        "name": "test_auroc_EU", 
        "goal": "maximize"
    },
    "parameters": {
        "M": {"values": [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]},
    },
}


sweep_credal = {
    "method": "bayes",
    "metric": {
        "name": "val_auroc_EU", 
        "goal": "maximize"
    },
    "parameters": {
        "lr": {"distribution": "uniform", "min": 1e-5, "max": 1e-1},
        "hidden_channels": {"values": [64, 128, 256]},
        "num_layers": {"values": [2, 3]},
        "weight_decay": {"distribution": "uniform", "min": 1e-7, "max": 1e-1},
        "delta": {"distribution": "uniform", "min": 0.5, "max": 1.0},
        "gnn_type": {"values": ["GCN", "SAGE"]},
        "patience": {"values": [10]},
    },
}


sweep_credal_LJ = {
    "method": "bayes",
    "metric": {
        "name": "val_auroc_EU", 
        "goal": "maximize"
    },
    "parameters": {
        "lr": {"distribution": "uniform", "min": 1e-5, "max": 1e-1},
        "hidden_channels": {"values": [64, 128, 256]},
        "num_layers": {"values": [2, 3]},
        "weight_decay": {"distribution": "uniform", "min": 1e-7, "max": 1e-1},
        "delta": {"distribution": "uniform", "min": 0.5, "max": 1.0},
        "gnn_type": {"values": ["GCN", "SAGE"]},
        "patience": {"values": [10]},
    },
}


sweep_mahalanobis = {
    "method": "grid",  
    "metric": {
        "name": "val_auroc", 
        "goal": "maximize"
    },
    "parameters": {
        "noise_magnitude": {
            "values": [0.0, 0.001, 0.005, 0.01, 0.05]
        }
    },
}



sweep_knn = {
    "method": "grid",
    "metric": {
        "name": "val_auroc", 
        "goal": "maximize"
    },
    "parameters": {
        "k": {
            "values": [5, 10, 20, 50, 100, 200]
        }
    },
}

sweep_energy = {
    "method": "grid",
    "metric": {
        "name": "val_auroc", 
        "goal": "maximize"
    },
    "parameters": {
        "dummy_run_id": {"values": [1]} # A dummy parameter to ensure the agent runs exactly once.
    },
}

sweep_knn_LJ = {
    "method": "grid",
    "metric": {
        "name": "val_auroc", 
        "goal": "maximize"
    },
    "parameters": {
        "k": {
            "values": [5, 10, 20, 50, 100, 200]
        }
    },
}

sweep_gnnsafe = {
    "method": "grid",
    "metric": {
        "name": "val_auroc", 
        "goal": "maximize"
    },
    "parameters": {
        "K": {"values": [1, 2, 4, 8, 16]},
        "alpha": {"values": [0.1, 0.3, 0.5, 0.7, 0.9]}
    },
}

sweep_gebm = {
    "method": "grid",
    "metric": {"name": "auroc_GEBM", "goal": "maximize"},
    "parameters": {
        # keep consistent with your other sweeps
        "seed": {"values": [0, 1, 2, 3, 4]},
    },
}

sweep_frozen = {
    "method": "bayes",
    "metric": {"name": "val_auroc_EU", "goal": "maximize"},
    "parameters": {
        "lr": {"distribution": "uniform", "min": 1e-5, "max": 1e-1},
        "weight_decay": {"distribution": "uniform", "min": 1e-7, "max": 1e-1},
        "seed": {"values": [0, 1, 2, 3, 4]},
        "delta": {"distribution": "uniform", "min": 0.5, "max": 1.0},
        "patience": {"values": [10]},
    },
}

sweep_cagcn = {
    "method": "bayes",
    "metric": {"name": "val_auroc", "goal": "maximize"},
    "parameters": {
        "seed": {"values": [0,1,2,3,4]},
        "lambda_cal": {"values": [0.25, 0.5, 0.75]},
        "calib_hidden": {"values": [8,16,32]},
        "calib_layers": {"values": [1,2]},
        "lr": {"distribution": "uniform", "min": 1e-5, "max": 1e-1},
        "weight_decay": {"distribution": "uniform", "min": 1e-7, "max": 1e-1},
        "max_epochs": {"values": [200]},
        "patience": {"values": [10]},
    },
}

sweep_random_set = {
    "method": "bayes",
    "metric": {"name": "val_auroc_entropy", "goal": "maximize"},
    "parameters": {
        "gnn_type": {"values": ["GAT"]},
        "lr": {"distribution": "uniform", "min": 1e-5, "max": 1e-1},
        "hidden_channels": {"values": [64, 128, 256]},
        "num_layers": {"values": [2, 3]},
        "weight_decay": {"distribution": "uniform", "min": 1e-7, "max": 1e-1},
        "singletons_only": {"values": [False]},
        "loss_ablation": {"values": ["full"]},
        "use_bce_loss": {"values": [True]},
        "use_mr_loss": {"values": [True]},
        "use_ms_loss": {"values": [True]},
    },
}


def _clone_random_set_sweep(**parameter_overrides):
    sweep = {
        "method": sweep_random_set["method"],
        "metric": dict(sweep_random_set["metric"]),
        "parameters": {
            key: dict(value) for key, value in sweep_random_set["parameters"].items()
        },
    }
    for key, value in parameter_overrides.items():
        sweep["parameters"][key] = value
    return sweep


sweep_random_set_ablation_singletons_only = _clone_random_set_sweep(
    singletons_only={"values": [True]},
    loss_ablation={"values": ["full"]},
)

sweep_random_set_ablation_bce_only = _clone_random_set_sweep(
    singletons_only={"values": [False]},
    loss_ablation={"values": ["bce_only"]},
    use_bce_loss={"values": [True]},
    use_mr_loss={"values": [False]},
    use_ms_loss={"values": [False]},
)

sweep_random_set_ablation_bce_mr = _clone_random_set_sweep(
    singletons_only={"values": [False]},
    loss_ablation={"values": ["bce_mr"]},
    use_bce_loss={"values": [True]},
    use_mr_loss={"values": [True]},
    use_ms_loss={"values": [False]},
)

sweep_random_set_ablation_bce_ms = _clone_random_set_sweep(
    singletons_only={"values": [False]},
    loss_ablation={"values": ["bce_ms"]},
    use_bce_loss={"values": [True]},
    use_mr_loss={"values": [False]},
    use_ms_loss={"values": [True]},
)

sweep_random_set_ablation_singletons_bce_only = _clone_random_set_sweep(
    singletons_only={"values": [True]},
    loss_ablation={"values": ["bce_only"]},
    use_bce_loss={"values": [True]},
    use_mr_loss={"values": [False]},
    use_ms_loss={"values": [False]},
)

sweep_random_set_ablation_focal_and_loss_grid = {
    "method": "grid",
    "metric": {"name": "val_auroc_entropy", "goal": "maximize"},
    "parameters": {
        "gnn_type": {"values": ["GAT"]},
        "lr": {"values": [1e-3, 1e-2]},
        "hidden_channels": {"values": [64, 128]},
        "num_layers": {"values": [2, 3]},
        "weight_decay": {"values": [1e-4, 1e-2]},
        "singletons_only": {"values": [False, True]},
        "loss_ablation": {"values": ["full", "bce_only", "bce_mr", "bce_ms"]},
        "use_bce_loss": {"values": [True]},
        "use_mr_loss": {"values": [True]},
        "use_ms_loss": {"values": [True]},
    },
}

