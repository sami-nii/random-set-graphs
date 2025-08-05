
metadata_chameleon = {"in_channels": {"values": [2325]}, "out_channels": {"values": [3]}}
metadata_squirrel  = {"in_channels": {"values": [2089]}, "out_channels": {"values": [3]}}
metadata_cora      = {"in_channels": {"values": [1433]}, "out_channels": {"values": [3]}}
metadata_patents   = {"in_channels": {"values": [269]},  "out_channels": {"values": [3]}}
metadata_arxiv     = {"in_channels": {"values": [128]},  "out_channels": {"values": [3]}}
metadata_reddit2   = {"in_channels": {"values": [602]},  "out_channels": {"values": [30]}}


sweep_vanilla = {
    "method": "bayes",
    "metric": {"name": "val_f1", "goal": "maximize"},
    "parameters": {
        "lr": {"distribution": "uniform", "min": 1e-5, "max": 1e-1},
        "hidden_channels": {"values": [64, 128, 256, 512]},
        "num_layers": {"values": [1, 2, 3, 4]},
        "batch_size": {"values": [1]},
        "weight_decay": {"distribution": "uniform", "min": 1e-6, "max": 1e-1},
        "gnn_type": {"values": ["GCN", "SAGE", "GAT", "GIN", "EdgeCNN"]},
        "patience": {"values": [30]},
    }
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
        "num_layers": {"values": [1, 2, 3]},
        "batch_size": {"values": [-1]},
        "num_neighbors": {"values": [-1]},
        "weight_decay": {"distribution": "uniform", "min": 1e-6, "max": 1e-1},
        "delta": {"distribution": "uniform", "min": 0.5, "max": 1.0},
        "gnn_type": {"values": ["GCN", "SAGE"]},
        "patience": {"values": [10]},
    },
}


sweep_ensemble = {
    "method": "grid", 
    "metric": {
        "name": "test_auroc_EU", 
        "goal": "maximize"
    },
    "parameters": {

        "M": {
            "values": [3, 5, 7, 10, 15]  
        },

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
        "hidden_channels": {"values": [32, 64, 128, 256]},
        "num_layers": {"values": [1, 2, 3]},
        "batch_size": {"values": [-1]},
        "num_neighbors": {"values": [-1]},
        "weight_decay": {"distribution": "uniform", "min": 1e-6, "max": 1e-1},
        "delta": {"distribution": "uniform", "min": 0.5, "max": 1.0},
        "gnn_type": {"values": ["GCN", "SAGE"]},
        "patience": {"values": [10]},
    },
}