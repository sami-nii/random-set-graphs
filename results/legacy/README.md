# Legacy results (reference only)

Tables and figures written by the earlier version of this repository (`main.py`, `trainers/`,
`models/`, `plots/` before the port onto the `cgnn` package; see the git history). They are kept for
provenance; the current code does not read them.

- `tables/task_3_focal_set_budget_k_ablation_*.md`: focal-set construction time vs K (paper Tab. 7).
- `tables/task_2_random_set_layer_overhead.md`, `tables/task_2_baseline_overhead_summary.md`:
  per-forward overhead (paper Tab. 9).
- `tables/*isotonic_calibration_summary.md`: isotonic-calibration runs. The road/nuScenes ones used the
  earlier single-graph object protocols, not the temporal scene graphs of the paper.
- `tables/aggregate_results.*`, `top_runs.txt`, `random_set_runs.csv`, `*.png`: aggregates of local W&B runs.

These runs used the earlier configuration (PyG `GAT`/`GCN`/`SAGE` backbones, budget K = 32, an untrained
warm-up GNN for the focal-set budget, unseeded random splits). Results from the current code will differ.
