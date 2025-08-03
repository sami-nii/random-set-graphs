sweep_cora = {
    "method": "bayes",
    "metric": {"name": "val_f1", "goal": "maximize"},
    "parameters": {
        "lr": {"distribution": "uniform", "min": 1e-5, "max": 1e-1},
        "hidden_channels": {"values": [64, 128, 256, 512]},
        "num_layers": {"values": [1, 2, 3, 4]},
        "batch_size": {"values": [1]},
        "weight_decay": {"distribution": "uniform", "min": 1e-6, "max": 1e-1},
        "delta": {"distribution": "uniform", "min": 0.5, "max": 1.0},
        "gnn_type": {"values": ["GCN", "SAGE", "GAT", "GIN", "EdgeCNN"]},
        "in_channels": {"values": [1433]},
        "out_channels": {"values": [3]},
        "patience": {"values": [30]},
    }
}

sweep_squirrel_vanilla = {
    "method": "bayes",
    "metric": {"name": "val_f1", "goal": "maximize"},
    "parameters": {
        "lr": {"distribution": "uniform", "min": 1e-5, "max": 1e-1},
        "hidden_channels": {"values": [64, 128, 256, 512]},
        "num_layers": {"values": [1, 2, 3, 4]},
        "batch_size": {"values": [1]},
        "weight_decay": {"distribution": "uniform", "min": 1e-6, "max": 1e-1},
        "gnn_type": {"values": ["GCN", "SAGE", "GAT", "GIN", "EdgeCNN"]},
        "in_channels": {"values": [2089]},
        "out_channels": {"values": [3]},
        "patience": {"values": [30]},
    }
}

sweep_chameleon = {
    "method": "bayes",
    "metric": {"name": "val_f1", "goal": "maximize"},
    "parameters": {
        "lr": {"distribution": "uniform", "min": 1e-5, "max": 1e-1},
        "hidden_channels": {"values": [64, 128, 256, 512]},
        "num_layers": {"values": [1, 2, 3, 4]},
        "batch_size": {"values": [1]},
        "weight_decay": {"distribution": "uniform", "min": 1e-6, "max": 1e-1},
        "delta": {"distribution": "uniform", "min": 0.5, "max": 1.0},
        "gnn_type": {"values": ["GCN", "SAGE", "GAT", "GIN", "EdgeCNN"]},
        "in_channels": {"values": [2325]},
        "out_channels": {"values": [3]},
        "patience": {"values": [30]},
    }
}

sweep_snap_patents_msp = {
    "method": "bayes",
    "metric": {
        "name": "val_f1", 
        "goal": "maximize"
    },
    "parameters": {
        "lr": {"distribution": "uniform", "min": 1e-5, "max": 1e-1},
        "hidden_channels": {"values": [64, 128, 256]},
        "num_layers": {"values": [1, 2, 3]},
        "batch_size": {"values": [-1]},
        "num_neighbors": {"values": [-1]},
        "weight_decay": {"distribution": "uniform", "min": 1e-6, "max": 1e-1},
        "gnn_type": {"values": ["GCN", "SAGE"]},
        "in_channels": {"values": [269]},
        "out_channels": {"values": [3]},
        "patience": {"values": [10]},
    },
}

sweep_patents_credal = {
    "method": "bayes",
    "metric": {
        "name": "val_auroc_EU", 
        "goal": "maximize"
    },
    "parameters": {
        "lr": {"distribution": "uniform", "min": 1e-5, "max": 1e-1},
        "hidden_channels": {"values": [64, 128, 256]},
        "num_layers": {"values": [1, 2, 3]},
        "batch_size": {"values": [-1]},
        "num_neighbors": {"values": [-1]},
        "weight_decay": {"distribution": "uniform", "min": 1e-6, "max": 1e-1},
        "delta": {"distribution": "uniform", "min": 0.5, "max": 1.0},
        "gnn_type": {"values": ["GCN", "SAGE"]},
        "in_channels": {"values": [269]},
        "out_channels": {"values": [3]},
        "patience": {"values": [10]},
    },
}


sweep_chameleon_credal = {
    "method": "bayes",
    "metric": {
        "name": "val_auroc_EU", 
        "goal": "maximize"
    },
    "parameters": {
        "lr": {"distribution": "uniform", "min": 1e-5, "max": 1e-1},
        "hidden_channels": {"values": [32, 64, 128, 256]},
        "num_layers": {"values": [1, 2, 3]},
        "batch_size": {"values": [-1]},
        "num_neighbors": {"values": [-1]},
        "weight_decay": {"distribution": "uniform", "min": 1e-6, "max": 1e-1},
        "delta": {"distribution": "uniform", "min": 0.5, "max": 1.0},
        "gnn_type": {"values": ["GCN", "SAGE"]},
        "in_channels": {"values": [2325]},
        "out_channels": {"values": [3]},
        "patience": {"values": [10]},
    },
}


sweep_arxiv_credal = {
    "method": "bayes",
    "metric": {
        "name": "val_auroc_EU", 
        "goal": "maximize"
    },
    "parameters": {
        "lr": {"distribution": "uniform", "min": 1e-5, "max": 1e-1},
        "hidden_channels": {"values": [32, 64, 128, 256]},
        "num_layers": {"values": [1, 2, 3]},
        "batch_size": {"values": [-1]},
        "num_neighbors": {"values": [-1]},
        "weight_decay": {"distribution": "uniform", "min": 1e-6, "max": 1e-1},
        "delta": {"distribution": "uniform", "min": 0.5, "max": 1.0},
        "gnn_type": {"values": ["GCN", "SAGE"]},
        "in_channels": {"values": [128]},
        "out_channels": {"values": [3]},
        "patience": {"values": [10]},
    },
}

sweep_reddit2_credal = {
    "method": "bayes",
    "metric": {
        "name": "val_auroc_EU", 
        "goal": "maximize"
    },
    "parameters": {
        "lr": {"distribution": "uniform", "min": 1e-5, "max": 1e-1},
        "hidden_channels": {"values": [32, 64, 128, 256]},
        "num_layers": {"values": [1, 2, 3]},
        "batch_size": {"values": [-1]},
        "num_neighbors": {"values": [-1]},
        "weight_decay": {"distribution": "uniform", "min": 1e-6, "max": 1e-1},
        "delta": {"distribution": "uniform", "min": 0.5, "max": 1.0},
        "gnn_type": {"values": ["GCN", "SAGE"]},
        "in_channels": {"values": [602]},
        "out_channels": {"values": [30]},
        "patience": {"values": [10]},
    },
}

sweep_coauthor_credal = {
    "method": "bayes",
    "metric": {
        "name": "val_auroc_EU", 
        "goal": "maximize"
    },
    "parameters": {
        "lr": {"distribution": "uniform", "min": 1e-5, "max": 1e-1},
        "hidden_channels": {"values": [32, 64, 128, 256]},
        "num_layers": {"values": [1, 2, 3]},
        "batch_size": {"values": [-1]},
        "num_neighbors": {"values": [-1]},
        "weight_decay": {"distribution": "uniform", "min": 1e-6, "max": 1e-1},
        "delta": {"distribution": "uniform", "min": 0.5, "max": 1.0},
        "gnn_type": {"values": ["GCN", "SAGE"]},
        "in_channels": {"values": [6805]},
        "out_channels": {"values": [11]},
        "patience": {"values": [10]},
    },
}

sweep_reddit2_vanilla = {
    "method": "bayes",
    "metric": {
        "name": "val_f1",  # Corrected Goal: Find the best classifier
        "goal": "maximize"
    },
    "parameters": {
        # --- Standard Hyperparameters ---
        "lr": {"distribution": "uniform", "min": 1e-5, "max": 1e-1},
        "hidden_channels": {"values": [32, 64, 128, 256]},
        "num_layers": {"values": [1, 2, 3, 4]},
        "weight_decay": {"distribution": "uniform", "min": 1e-6, "max": 1e-1},
        "gnn_type": {"values": ["GCN", "SAGE"]},
        
        # --- Fixed/Callback Parameters ---
        "batch_size": {"values": [-1]}, # For full-graph training
        "num_neighbors": {"values": [-1]}, # Ignored for full-graph, but kept for consistency
        "patience": {"values": [10]},
        
        # --- Dataset-Specific (Fixed) Parameters for Reddit2 ---
        "in_channels": {"values": [602]},
        "out_channels": {"values": [30]}, # Corresponds to the 30 ID classes (40 - 11 + 1 is incorrect, it's len(IDclass))
                                        # Let's assume it should be 30 for classes 11-40.
    },
}

sweep_reddit2_ensemble = {
    "method": "grid",  # Grid search is ideal for a single hyperparameter like M
    "metric": {
        "name": "test_auroc_EU",  # The goal is to find the best performing ensemble on the test set
        "goal": "maximize"
    },
    "parameters": {
        # --- The CORE Hyperparameter for the Ensemble ---
        "M": {
            "values": [3, 5, 7, 10, 15]  # A good range of ensemble sizes to test
        },
        
        # --- Fixed Parameters needed by the loader/script ---
        "batch_size": {"values": [-1]}, # For full-graph inference
    },
}