# Focal-set budget K ablation

Setup time is the shared, one-time embedding/GMM/ellipsoid preparation. Overlap time is measured independently for each K; total is their sum.

| Dataset | K | Repetitions | Focal sets | Setup (s) | Overlap mean ± sample std (s) | Estimated total (s) |
|---|---:|---:|---:|---:|---:|---:|
| reddit2 | 4 | 3 | 34 | 36.836 | 5.921 ± 0.024 | 42.757 |
| reddit2 | 8 | 3 | 38 | 36.836 | 5.839 ± 0.080 | 42.675 |
| reddit2 | 16 | 3 | 46 | 36.836 | 5.869 ± 0.024 | 42.705 |
| reddit2 | 32 | 3 | 62 | 36.836 | 5.869 ± 0.060 | 42.705 |
| reddit2 | 64 | 3 | 94 | 36.836 | 6.152 ± 0.243 | 42.987 |
