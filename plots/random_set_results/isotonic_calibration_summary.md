# RS-GNN with post-hoc isotonic calibration

Each value is mean ± sample standard deviation across runs. The calibrator is fitted on labelled ID validation nodes only.

| Dataset | Runs | Test ID ECE: before → after | Δ ECE | Test entropy AUROC: before → after | Δ AUROC |
|---|---:|---|---:|---|---:|
| amazon_ratings | 3 | 0.1361 ± 0.1111 → 0.0072 ± 0.0040 | -0.1289 | 0.4836 ± 0.0029 → 0.4939 ± 0.0124 | +0.0103 |
| arxiv | 3 | 0.0218 ± 0.0239 → 0.0105 ± 0.0031 | -0.0113 | 0.5115 ± 0.0425 → 0.5317 ± 0.0014 | +0.0202 |
| chameleon | 3 | 0.0984 ± 0.1326 → 0.0528 ± 0.0407 | -0.0456 | 0.5992 ± 0.1756 → 0.5407 ± 0.1831 | -0.0586 |
| coauthor | 3 | 0.1191 ± 0.0359 → 0.0874 ± 0.0688 | -0.0317 | 0.4767 ± 0.0853 → 0.4761 ± 0.0963 | -0.0006 |
| cora | 3 | 0.0757 ± 0.0317 → 0.0527 ± 0.0359 | -0.0230 | 0.5246 ± 0.0464 → 0.5435 ± 0.1041 | +0.0189 |
| roman_empire | 3 | 0.0719 ± 0.0485 → 0.0186 ± 0.0017 | -0.0532 | 0.4640 ± 0.1599 → 0.3651 ± 0.0486 | -0.0989 |
| squirrel | 3 | 0.0186 ± 0.0135 → 0.0298 ± 0.0111 | +0.0112 | 0.5328 ± 0.0305 → 0.5569 ± 0.0261 | +0.0241 |
