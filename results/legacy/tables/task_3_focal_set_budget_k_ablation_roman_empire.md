# Focal-set budget K ablation

Setup time is the shared, one-time embedding/GMM/ellipsoid preparation. Overlap time is measured independently for each K; total is their sum.

| Dataset | K | Repetitions | Focal sets | Setup (s) | Overlap mean ± sample std (s) | Estimated total (s) |
|---|---:|---:|---:|---:|---:|---:|
| roman_empire | 4 | 3 | 17 | 20.212 | 3.026 ± 0.055 | 23.238 |
| roman_empire | 8 | 3 | 21 | 20.212 | 3.035 ± 0.007 | 23.247 |
| roman_empire | 16 | 3 | 29 | 20.212 | 3.040 ± 0.018 | 23.251 |
| roman_empire | 32 | 3 | 45 | 20.212 | 3.083 ± 0.058 | 23.295 |
| roman_empire | 64 | 3 | 77 | 20.212 | 3.104 ± 0.027 | 23.315 |
