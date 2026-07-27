# Post-hoc baseline overhead

Dataset: `cora`  
Checkpoints: rz9l4nlx_cora_val_f1=0.9521.ckpt, i0ecwdnk_cora_val_f1=0.9461.ckpt, ft72qzvd_cora_val_f1=0.9401.ckpt  
Device: `cpu`; repetitions: 3.

Setup is one-time post-hoc work. Additional forward time is detector forward time minus a vanilla GNN forward time on the same graph.

| Method | Checkpoints | Setup mean ± sample std (s) | Vanilla forward mean ± sample std (s) | Detector forward mean ± sample std (s) | Additional forward time (s) | Status |
|---|---:|---:|---:|---:|---:|---|
| energy | 3 | 0.000 ± 0.000 | 0.005 ± 0.000 | 0.006 ± 0.001 | 0.000 ± 0.000 | ok |
| knn | 3 | 0.005 ± 0.001 | 0.007 ± 0.001 | 0.015 ± 0.003 | 0.008 ± 0.002 | ok |
| mahalanobis | 3 | 0.009 ± 0.002 | 0.007 ± 0.001 | 0.008 ± 0.001 | 0.001 ± 0.000 | ok |
| odin | 3 | 0.000 ± 0.000 | 0.005 ± 0.000 | 0.030 ± 0.008 | 0.024 ± 0.008 | ok |
