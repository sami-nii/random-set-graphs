# Task 2: post-hoc baseline overhead

Setup is one-time post-hoc work; additional forward time is detector time minus vanilla-GNN forward time.
Values are mean ± sample standard deviation across the three fixed-seed vanilla checkpoints.

| Dataset | Method | Status | Checkpoints | Setup mean ± std (s) | Additional forward mean ± std (s) |
|---|---|---|---:|---:|---:|
| amazon_ratings | energy | complete | 3 | 0.000 ± 0.000 | 0.001 ± 0.001 |
| amazon_ratings | knn | complete | 3 | 0.021 ± 0.002 | 0.229 ± 0.032 |
| amazon_ratings | mahalanobis | complete | 3 | 0.029 ± 0.002 | 0.002 ± 0.001 |
| amazon_ratings | odin | complete | 3 | 0.000 ± 0.000 | 0.060 ± 0.004 |
| arxiv | energy | complete | 3 | 0.000 ± 0.000 | 0.011 ± 0.010 |
| arxiv | knn | complete | 3 | 0.188 ± 0.026 | 16.332 ± 0.225 |
| arxiv | mahalanobis | complete | 3 | 0.214 ± 0.005 | 0.010 ± 0.006 |
| arxiv | odin | complete | 3 | 0.000 ± 0.000 | 0.441 ± 0.022 |
| chameleon | energy | complete | 3 | 0.000 ± 0.000 | 0.001 ± 0.001 |
| chameleon | knn | complete | 3 | 0.009 ± 0.001 | 0.006 ± 0.001 |
| chameleon | mahalanobis | complete | 3 | 0.014 ± 0.004 | 0.001 ± 0.001 |
| chameleon | odin | complete | 3 | 0.000 ± 0.000 | 0.029 ± 0.001 |
| coauthor | energy | complete | 3 | 0.000 ± 0.000 | 0.006 ± 0.007 |
| coauthor | knn | complete | 3 | 0.121 ± 0.011 | 0.266 ± 0.017 |
| coauthor | mahalanobis | complete | 3 | 0.131 ± 0.003 | 0.007 ± 0.001 |
| coauthor | odin | complete | 3 | 0.000 ± 0.000 | 0.714 ± 0.355 |
| cora | energy | complete | 3 | 0.000 ± 0.000 | 0.000 ± 0.000 |
| cora | knn | complete | 3 | 0.005 ± 0.001 | 0.008 ± 0.002 |
| cora | mahalanobis | complete | 3 | 0.009 ± 0.002 | 0.001 ± 0.000 |
| cora | odin | complete | 3 | 0.000 ± 0.000 | 0.024 ± 0.008 |
| patents | all | excluded | 0 | — | — |
| reddit2 | all | excluded | 0 | — | — |
| roman_empire | energy | complete | 3 | 0.000 ± 0.000 | 0.001 ± 0.000 |
| roman_empire | knn | complete | 3 | 0.014 ± 0.000 | 0.225 ± 0.010 |
| roman_empire | mahalanobis | complete | 3 | 0.025 ± 0.001 | 0.012 ± 0.000 |
| roman_empire | odin | complete | 3 | 0.000 ± 0.000 | 0.050 ± 0.001 |
| squirrel | energy | complete | 3 | 0.000 ± 0.000 | -0.002 ± 0.005 |
| squirrel | knn | complete | 3 | 0.030 ± 0.001 | 0.017 ± 0.000 |
| squirrel | mahalanobis | complete | 3 | 0.042 ± 0.005 | 0.001 ± 0.001 |
| squirrel | odin | complete | 3 | 0.000 ± 0.000 | 0.091 ± 0.001 |
