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

sweep_ogb_arxiv_year = {
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
        "in_channels": {"values": [128]},
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

sweep_snap_patents_credal = {
    "method": "bayes",
    "metric": {
        "name": "test_auroc", 
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