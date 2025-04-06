sweep_cora = {
    "method": "random",
    "metric": {"name": "sweep_score", "goal": "maximize"},
    "parameters": {
        "lr": {"distribution": "uniform", "min": 1e-5, "max": 1e-1},
        "hidden_channels": {"values": [64, 128, 256, 512]},
        "num_layers": {"values": [2]},
        "batch_size": {"values": [1]},
        "weight_decay": {"distribution": "uniform", "min": 1e-4, "max": 1e-2},
        "delta": {"distribution": "uniform", "min": 0.1, "max": 1.0},
        "gnn_type": {"values": ["GCN"]},
        "in_channels": {"values": [1433]},
        "out_channels": {"values": [3]},
        "patience": {"values": [30]},
    },
}
