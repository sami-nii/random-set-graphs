# Focal-set budget K ablation

Setup time is the shared, one-time embedding/GMM/ellipsoid preparation. Overlap time is measured independently for each K; total is their sum.

| Dataset | K | Repetitions | Focal sets | Setup (s) | Overlap mean ± sample std (s) | Estimated total (s) |
|---|---:|---:|---:|---:|---:|---:|
| roman_empire | 4 | 3 | 17 | 23.275 | 3.558 ± 0.380 | 26.833 |
| roman_empire | 8 | 3 | 21 | 23.275 | 3.250 ± 0.080 | 26.525 |
| roman_empire | 16 | 3 | 29 | 23.275 | 3.709 ± 0.506 | 26.984 |
| roman_empire | 32 | 3 | 45 | 23.275 | 3.391 ± 0.346 | 26.666 |
| roman_empire | 64 | 3 | 77 | 23.275 | 3.275 ± 0.146 | 26.551 |
