sweep_cora = {
    "method": "random",
    "metric": {"name": "sweep_score", "goal": "maximize"},
    "parameters": {
        "lr": {"distribution": "uniform", "min": 1e-4, "max": 1e-1},
        "hidden_channels": {"values": [64, 128, 256]},
        "num_layers": {"values": [2, 3]},
        "batch_size": {"values": [16, 32, 64]},
        "weight_decay": {"distribution": "uniform", "min": 1e-7, "max": 1e-4},
        "gnn_type": {"values": ["GCN"]},
        "in_channels": {"values": [1433]},
        "out_channels": {"values": [7]},
        "patience": {"values": [30]},
    },
}
