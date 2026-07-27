# Post-hoc baseline overhead

Dataset: `roman_empire`  
Checkpoints: 3qu0v04a_roman_empire_val_f1=0.2649.ckpt, gzuvnddx_roman_empire_val_f1=0.2568.ckpt, dqmedoyu_roman_empire_val_f1=0.2100.ckpt  
Device: `cpu`; repetitions: 3.

Setup is one-time post-hoc work. Additional forward time is detector forward time minus a vanilla GNN forward time on the same graph.

| Method | Checkpoints | Setup mean ± sample std (s) | Vanilla forward mean ± sample std (s) | Detector forward mean ± sample std (s) | Additional forward time (s) | Status |
|---|---:|---:|---:|---:|---:|---|
| energy | 3 | 0.000 ± 0.000 | 0.017 ± 0.000 | 0.019 ± 0.000 | 0.001 ± 0.000 | ok |
| knn | 3 | 0.014 ± 0.000 | 0.019 ± 0.001 | 0.244 ± 0.010 | 0.225 ± 0.010 | ok |
| mahalanobis | 3 | 0.025 ± 0.001 | 0.018 ± 0.001 | 0.030 ± 0.001 | 0.012 ± 0.000 | ok |
| odin | 3 | 0.000 ± 0.000 | 0.017 ± 0.001 | 0.067 ± 0.001 | 0.050 ± 0.001 | ok |
