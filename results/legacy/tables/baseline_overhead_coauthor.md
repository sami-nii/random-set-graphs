# Post-hoc baseline overhead

Dataset: `coauthor`  
Checkpoints: iqtvi6dk_coauthor_val_f1=0.9537.ckpt, 96lsiwam_coauthor_val_f1=0.9470.ckpt, mn0y80cl_coauthor_val_f1=0.9424.ckpt  
Device: `cpu`; repetitions: 3.

Setup is one-time post-hoc work. Additional forward time is detector forward time minus a vanilla GNN forward time on the same graph.

| Method | Checkpoints | Setup mean ± sample std (s) | Vanilla forward mean ± sample std (s) | Detector forward mean ± sample std (s) | Additional forward time (s) | Status |
|---|---:|---:|---:|---:|---:|---|
| energy | 3 | 0.000 ± 0.000 | 0.128 ± 0.017 | 0.134 ± 0.023 | 0.006 ± 0.007 | ok |
| knn | 3 | 0.121 ± 0.011 | 0.128 ± 0.007 | 0.394 ± 0.023 | 0.266 ± 0.017 | ok |
| mahalanobis | 3 | 0.131 ± 0.003 | 0.124 ± 0.006 | 0.132 ± 0.006 | 0.007 ± 0.001 | ok |
| odin | 3 | 0.000 ± 0.000 | 0.126 ± 0.004 | 0.840 ± 0.358 | 0.714 ± 0.355 | ok |
