# Task 1: full focal-set budget construction time

Times include sampling, auxiliary embeddings, GMM fitting, ellipsoid construction, and overlap selection.

| Dataset | Strategy | Status | K | Repetitions | Focal sets | Mean (s) | Sample std (s) |
|---|---|---|---:|---:|---:|---:|---:|
| amazon_ratings | full_power_set | not_applicable |  | 3 |  | — | — |
| arxiv | full_power_set | not_applicable |  | 6 |  | — | — |
| chameleon | full_power_set | not_applicable |  | 3 |  | — | — |
| coauthor | budgeted | complete | 32 | 3 | 43 | 18.823 | 2.080 |
| cora | full_power_set | not_applicable |  | 3 |  | — | — |
| patents | full_power_set | not_applicable |  | 2 |  | — | — |
| reddit2 | budgeted | complete | 32 | 3 | 62 | 39.233 | 1.706 |
| roman_empire | budgeted | complete | 32 | 6 | 45 | 21.676 | 3.447 |
| squirrel | full_power_set | not_applicable |  | 3 |  | — | — |
