# Post-hoc baseline overhead

Dataset: `amazon_ratings`  
Checkpoints: nj4h2q2c_amazon_ratings_val_f1=0.6369.ckpt, ewzbmecp_amazon_ratings_val_f1=0.6365.ckpt, j9quvbdi_amazon_ratings_val_f1=0.6365.ckpt  
Device: `cpu`; repetitions: 3.

Setup is one-time post-hoc work. Additional forward time is detector forward time minus a vanilla GNN forward time on the same graph.

| Method | Checkpoints | Setup mean ± sample std (s) | Vanilla forward mean ± sample std (s) | Detector forward mean ± sample std (s) | Additional forward time (s) | Status |
|---|---:|---:|---:|---:|---:|---|
| energy | 3 | 0.000 ± 0.000 | 0.023 ± 0.000 | 0.024 ± 0.001 | 0.001 ± 0.001 | ok |
| knn | 3 | 0.021 ± 0.002 | 0.032 ± 0.008 | 0.261 ± 0.039 | 0.229 ± 0.032 | ok |
| mahalanobis | 3 | 0.029 ± 0.002 | 0.024 ± 0.002 | 0.026 ± 0.001 | 0.002 ± 0.001 | ok |
| odin | 3 | 0.000 ± 0.000 | 0.024 ± 0.003 | 0.085 ± 0.007 | 0.060 ± 0.004 | ok |
