# Task 2: RS-GNN random-set layer inference overhead

The shared GNN backbone is excluded. The table compares a standard linear class head with RandomSetLayer on the same node embeddings.
Values are mean ± sample standard deviation across three vanilla-checkpoint hidden dimensions; each checkpoint uses repeated layer calls.

| Dataset | Checkpoints | Nodes | Classes | Focal sets | Standard head (ms) | Random-set layer (ms) | Additional RS layer (ms) |
|---|---:|---:|---:|---:|---:|---:|---:|
| amazon_ratings | 3 | 24492 | 3 | 7 | 0.472 ± 0.015 | 0.842 ± 0.065 | 0.370 ± 0.049 |
| arxiv | 3 | 169343 | 3 | 7 | 3.030 ± 0.328 | 5.735 ± 0.050 | 2.705 ± 0.279 |
| chameleon | 3 | 2277 | 3 | 7 | 0.068 ± 0.012 | 0.138 ± 0.020 | 0.069 ± 0.008 |
| coauthor | 3 | 18333 | 11 | 43 | 0.865 ± 0.030 | 1.352 ± 0.024 | 0.487 ± 0.007 |
| cora | 3 | 2708 | 3 | 7 | 0.071 ± 0.010 | 0.150 ± 0.023 | 0.080 ± 0.021 |
| roman_empire | 3 | 22662 | 13 | 45 | 0.912 ± 0.033 | 1.563 ± 0.054 | 0.651 ± 0.069 |
| squirrel | 3 | 5201 | 3 | 7 | 0.135 ± 0.026 | 0.245 ± 0.007 | 0.110 ± 0.025 |
