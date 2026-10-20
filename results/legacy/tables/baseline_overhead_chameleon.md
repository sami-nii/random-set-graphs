# Post-hoc baseline overhead

Dataset: `chameleon`  
Checkpoints: e9pwnd6f_chameleon_val_f1=0.5662.ckpt, lxz52z7e_chameleon_val_f1=0.5594.ckpt, zflp8axw_chameleon_val_f1=0.5114.ckpt  
Device: `cpu`; repetitions: 3.

Setup is one-time post-hoc work. Additional forward time is detector forward time minus a vanilla GNN forward time on the same graph.

| Method | Checkpoints | Setup mean ± sample std (s) | Vanilla forward mean ± sample std (s) | Detector forward mean ± sample std (s) | Additional forward time (s) | Status |
|---|---:|---:|---:|---:|---:|---|
| energy | 3 | 0.000 ± 0.000 | 0.009 ± 0.001 | 0.009 ± 0.000 | 0.001 ± 0.001 | ok |
| knn | 3 | 0.009 ± 0.001 | 0.011 ± 0.001 | 0.017 ± 0.001 | 0.006 ± 0.001 | ok |
| mahalanobis | 3 | 0.014 ± 0.004 | 0.010 ± 0.001 | 0.011 ± 0.001 | 0.001 ± 0.001 | ok |
| odin | 3 | 0.000 ± 0.000 | 0.009 ± 0.000 | 0.037 ± 0.001 | 0.029 ± 0.001 | ok |
