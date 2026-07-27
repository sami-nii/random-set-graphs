# Post-hoc baseline overhead

Dataset: `arxiv`  
Checkpoints: 4fd33b8n_arxiv_val_f1=0.5201.ckpt, 69havk2g_arxiv_val_f1=0.5195.ckpt, wylldzr9_arxiv_val_f1=0.5195.ckpt  
Device: `cpu`; repetitions: 3.

Setup is one-time post-hoc work. Additional forward time is detector forward time minus a vanilla GNN forward time on the same graph.

| Method | Checkpoints | Setup mean ± sample std (s) | Vanilla forward mean ± sample std (s) | Detector forward mean ± sample std (s) | Additional forward time (s) | Status |
|---|---:|---:|---:|---:|---:|---|
| energy | 3 | 0.000 ± 0.000 | 0.202 ± 0.010 | 0.213 ± 0.006 | 0.011 ± 0.010 | ok |
| knn | 3 | 0.188 ± 0.026 | 0.214 ± 0.006 | 16.546 ± 0.229 | 16.332 ± 0.225 | ok |
| mahalanobis | 3 | 0.214 ± 0.005 | 0.204 ± 0.004 | 0.214 ± 0.003 | 0.010 ± 0.006 | ok |
| odin | 3 | 0.000 ± 0.000 | 0.200 ± 0.001 | 0.641 ± 0.023 | 0.441 ± 0.022 | ok |
