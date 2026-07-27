# Post-hoc baseline overhead

Dataset: `squirrel`  
Checkpoints: 5vlg5fsc_squirrel_val_f1=0.4033.ckpt, ruynhnvu_squirrel_val_f1=0.4033.ckpt, m8kh94xe_squirrel_val_f1=0.3798.ckpt  
Device: `cpu`; repetitions: 3.

Setup is one-time post-hoc work. Additional forward time is detector forward time minus a vanilla GNN forward time on the same graph.

| Method | Checkpoints | Setup mean ± sample std (s) | Vanilla forward mean ± sample std (s) | Detector forward mean ± sample std (s) | Additional forward time (s) | Status |
|---|---:|---:|---:|---:|---:|---|
| energy | 3 | 0.000 ± 0.000 | 0.038 ± 0.004 | 0.037 ± 0.000 | -0.002 ± 0.005 | ok |
| knn | 3 | 0.030 ± 0.001 | 0.040 ± 0.002 | 0.057 ± 0.002 | 0.017 ± 0.000 | ok |
| mahalanobis | 3 | 0.042 ± 0.005 | 0.036 ± 0.001 | 0.037 ± 0.001 | 0.001 ± 0.001 | ok |
| odin | 3 | 0.000 ± 0.000 | 0.035 ± 0.001 | 0.125 ± 0.001 | 0.091 ± 0.001 | ok |
