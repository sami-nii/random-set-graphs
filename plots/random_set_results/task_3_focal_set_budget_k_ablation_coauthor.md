# Focal-set budget K ablation

Setup time is the shared, one-time embedding/GMM/ellipsoid preparation. Overlap time is measured independently for each K; total is their sum.

| Dataset | K | Repetitions | Focal sets | Setup (s) | Overlap mean ± sample std (s) | Estimated total (s) |
|---|---:|---:|---:|---:|---:|---:|
| coauthor | 4 | 3 | 15 | 20.045 | 1.635 ± 0.030 | 21.680 |
| coauthor | 8 | 3 | 19 | 20.045 | 1.607 ± 0.003 | 21.652 |
| coauthor | 16 | 3 | 27 | 20.045 | 1.611 ± 0.014 | 21.656 |
| coauthor | 32 | 3 | 43 | 20.045 | 1.605 ± 0.004 | 21.650 |
| coauthor | 64 | 3 | 75 | 20.045 | 1.607 ± 0.005 | 21.652 |
