# Paper ↔ Code Map

Paper: **Random-Set Graph Neural Networks** (NeurIPS 2026, main track, camera-ready).
Source: `RS-GNN-Camera-Ready-/neurips_2026.tex`. This is the camera-ready main file. `MAIN.tex` is an older Overleaf copy and differs in one intro paragraph.

Paper artefacts are referenced by their LaTeX `\label` (stable across recompiles).

Legend: ✅ the code matches the paper · ⚠️ partial match or discrepancy · ❌ no code in this repo.

## 1. Method (paper §3 `sec:method`, App. `sec:implementation_details`)

| Paper element | Where in code | Status |
|---|---|---|
| Belief head `Bel_v(A_k)=σ(g_θ(z_v))` | `models/random_set_gnn.py::RandomSetLayer` | ✅ |
| Möbius inversion matrix M (`eq:mobius`) | `models/random_set_gnn.py::generate_matrices` (`moebius_mat`) | ✅ |
| Pignistic matrix P, BetP | `generate_matrices` (`pignistic_mat`), `RandomSetGNN.get_pignistic_probs` | ✅ |
| Clip → residual to Θ → renormalise → uniform fallback (`fig:rsgnn_belief_head`) | `RandomSetGNN.get_pignistic_probs` | ✅ |
| Loss `BCE + α R_neg + β R_sum` (`eq:final_loss`), α=β=1e-3 | `models/random_set_gnn.py::RandomSetLoss` (`mr_loss`=R_neg, `ms_loss`=R_sum) | ✅ |
| Pignistic entropy H_v (`eq:pignistic_entropy`) as the OOD score | `RandomSetGNN.validation_step/test_step` → `val_auroc_entropy`, `test_auroc_entropy` | ✅ |
| **Credal-set width W_v (`eq:credal_width`)** as the epistemic score (second number per cell in `tab:ood`) | **not computed anywhere**. `test_step` logs only the entropy AUROC | ❌ |
| Full power set when \|Y\|<10 | `trainers/random_set_trainer.py` (`itertools.combinations`, branch `num_id_classes < 10`) | ✅ |
| Budgeted focal sets (`alg:focal_sets`): 200 nodes/class, warm-up GNN, t-SNE 3-D, one Gaussian per class, 95 % ellipsoid (χ²₃=7.815), Jaccard over voxels, pairs→triples, top-K | `trainers/random_set_trainer.py` ("Budget Override" block) + `trainers/budgeting.py` (`train_embeddings`, `fit_gmm`, `ellipse`, `overlaps`) | ⚠️ the "warm-up GNN" is an **untrained** fresh backbone (`aux_model = RandomSetGNN(...).gnn_model`, never trained before embedding). The paper says it is trained on labelled ID nodes |
| K (budget) | config `budget_k` (default **32**). `sec:focal_sets` says "up to 64" | ⚠️ |
| Max cardinality 3 | config `budget_max_cardinality` (default 3) | ✅ |
| Seeds: GMM `random_state=7`; t-SNE unseeded | `trainers/budgeting.py` | ✅ (the paper says it is non-deterministic) |
| Origin of the budgeting algorithm (RS-NN, CIFAR-10, TensorFlow) | `rsnnnotebookcoderough/RSNN_CIFAR10_RESNET50.ipynb` (reference only, not used) | — |

## 2. Experimental protocol (paper `sec:setup`, `sec:transductive_experimental_setup`)

| Paper element | Where in code | Status |
|---|---|---|
| Leave-out-class transductive OOD: train = ID only; val/test = ID+OOD; OOD rows of `y` all zeros | every `dataset_loader/loader_*.py` (`data.y` one-hot over ID classes, zeros for OOD) | ✅ |
| ID/OOD class splits (`tab:dataset_stats`) | hard-coded `OODclass`/`IDclass` in each loader | ✅ (matches `tab:dataset_stats`) |
| Splits: Chameleon/Squirrel use the PyG split 0; Cora uses `Planetoid split="full"`; Amazon/Roman use npz split 0; ArXiv/Patents/Coauthor use a random 60/20/20 split | loaders | ⚠️ ArXiv/Patents/Coauthor use `torch.randperm` **with no seed** (see the `TODO fix the seed` in `dataset_loader.py`) |
| Model selection uses **OOD validation AUROC** (Bayes sweep metric `val_auroc_entropy`) | `sweeps/sweeps.py::sweep_random_set` | ⚠️ a protocol detail the paper does not state explicitly |
| Backbone "GATv2, 2 layers, hidden 128, 4→1 heads, dropout 0.2" (`sec:backbone_implementation`) | RS-GNN uses PyG `GAT` (v1 unless `v2=True`) with a swept hidden size {64,128,256} and depth {2,3}. Vanilla/credal/baselines sweep `GCN`/`SAGE` | ⚠️ big mismatch with the text |
| NeighborLoader mini-batching | only for `batch_size>0` (arxiv, patents, reddit2). The other datasets are full-batch | ⚠️ |
| Hardware/env: NVIDIA PyG container `nvcr.io/nvidia/pyg:24.05-py3` | `docker/Dockerfile` | ✅ |
| Command `python main.py --dataset D --model M --count N` | `main.py` | ✅ |

## 3. Datasets (paper §4.1, App. A; `tab:road_scene_dataset_stats`, `tab:dataset_stats`, `tab:feature_sparsity`)

| Dataset | Loader | Data on disk |
|---|---|---|
| Cora | `loader_cora.py` (PyG Planetoid) | `dataset/Cora` |
| Coauthor CS | `loader_coauthor.py` | `dataset/coauthors` |
| Reddit2 | `loader_reddit2.py` | `dataset/reddit2` |
| ArXiv-year | `loader_ogb_arxiv_year.py` (`even_quantile_labels`, from Ma et al. 2024) | `dataset/ogbn_arxiv` |
| Patents-year | `loader_snap_patents_year.py` | `dataset/snap-patents.mat` |
| Chameleon, Squirrel | `loader_chameleon.py`, `loader_squirrel.py` (PyG WikipediaNetwork) | `dataset/chameleon`, `dataset/squirrel` |
| Roman Empire, Amazon-Ratings | `loader_roman_empire.py`, `loader_amazon_ratings.py` (npz, Platonov et al.) | `dataset/*/*.npz` |
| **ROAD (temporal scene graphs)** | ❌ The paper uses temporal scene graphs (scene node + agent nodes, window 4, stride 4, frame step 5, 37-dim features, 49k/26k/26k graphs, 10 agent classes, OOD {8,9}). `loader_road.py` builds something different: a single object-action graph with 4-dim bbox features, top-3 actions as ID, and 20k sampled nodes | ❌ |
| **nuScenes (temporal scene graphs)** | ❌ The paper uses 8 categories, OOD {6,7}, 10-dim features, 307/39/38 graphs. `loader_nuscenes.py` is an "exploratory" single-graph protocol with 3 ID classes and a Windows default path (`S:\nuScenes`) | ❌ |
| ROAD(ID)→nuScenes(OOD) cross-dataset | ❌ no code | ❌ |
| Dataset statistics (`tab:dataset_stats`), feature sparsity (`tab:feature_sparsity`) | ❌ no script | ❌ |
| `dataset/pokec.mat`, `dataset/tolokers` | not used by the paper | — |

## 4. Baselines (paper §4.1, `tab:ood`)

| Paper name | `--model` | Trainer → Model | Needs vanilla ckpts? |
|---|---|---|---|
| Energy | `energy` | `energy_tester.py` → `energy_detector.py` | yes (top-1) |
| KNN | `knn` | `knn_tester.py` → `knn_detector.py` (faiss) | yes |
| ODIN | `odin` | `odin_tester.py` → `ODIN_Detector.py` | yes |
| Mahalanobis | `mahalanobis` | `mahalanobis_tester.py` → `mahalanobis_Detector.py` | yes |
| GNNSafe | `gnnsafe` | `gnn_safe_tester.py` → `gnnSafe_detector.py` | yes |
| GEBM | `gebm` | `gebm_tester.py` → `GEBM_detector.py` | yes (top-5, indexed by `seed`). ⚠️ **`--model gebm` crashes**: the import is commented out in `trainers/__init__.py` because `gebm_tester.py` needs the external `graph_ebm` package (Fuchsgruber et al.'s repo, not vendored here; `.gitignore` mentions `graph_ebm/wandb`) |
| CaGCN | `cagcn` | `cagcn_trainer.py` → `CaGCN.py` | yes (top-5) |
| Classical ensemble / Credal ensemble | `ensemble` | `ensamble_tester.py` → `credal_Ensemble.py` (logs `auroc_*_classic` and `auroc_*_credal`) | yes (top-M) |
| Credal final | `credal` | `credal_trainer.py` → `credal_GNN_t.py` + `credal_layer.py`, `credal_loss.py`, `utils/math.py`, `utils/solvers.py` | no |
| CredalLJ | `credal_LJ` | `credal_LJ_trainer.py` → `credal_GNN_LJ.py` | no |
| Credal frozen | `frozen` | `credal_frozen_trainer.py` → `credal_frozen_LJ.py` | yes (top-5) |
| JLDE (presumably KNN on the joint latent) | `knn_LJ` | `knn_LJ_tester.py` → `knn_LJ_detector.py` | yes |
| Vanilla softmax | `vanilla` | `vanilla_trainer.py` → `VanillaGNN.py`. Saves `checkpoints/{run}_{dataset}_val_f1=X.ckpt` | — |

Order dependency: run `vanilla` first. Every post-hoc baseline calls `utils/model_manager.py::find_best_checkpoints`, which reads the **cwd-relative** `checkpoints/` directory. That directory does not exist right now. The `graph-uncertainty/` directory holds Lightning/W&B checkpoints for 298 runs, but they do not follow that naming scheme.

The baseline code comes from the authors' earlier Credal-GNN work (Tolloso et al. 2025).

## 5. Results: where each number or figure comes from

| Paper artefact | Produced by | Stored result | Status |
|---|---|---|---|
| **`tab:ood`** OOD AUROC, 14 methods × 9 datasets | W&B project `graph-uncertainty` (best run per sweep) | no export in repo | ❌ no aggregation script |
| `fig:results_heatmap`, `fig:topk_per_dataset`, `fig:rsgnn_ranks` (`Figures/outlined/fig_*.pdf`) | not in repo | — | ❌ |
| **`tab:id_core_metrics`, `tab:ood_uncertainty_auroc`** (ROAD/nuScenes); `fig:road_scene_id_app`, `fig:overconfidence_collapse` | not in repo (temporal pipeline missing) | — | ❌ |
| **`tab:ablation_main`** (0.731/0.703, …), `fig:ablation_summary` | `sweeps.py::sweep_random_set_ablation_*` → `plots/random_set_results.py` | `plots/random_set_results/{best_per_ablation,ablation_auroc_distribution}.png` (pngs also copied into `Figures/`; the ablation png differs from the paper copy) | ⚠️ |
| **`tab:focal_budget`** focal-set budget vs K | `experiments/benchmark_focal_set_budget.py` (via `run_task_3.ps1`) | `plots/random_set_results/task_3_focal_set_budget_k_ablation_{coauthor,reddit2,roman_empire}.md` | ✅ matches |
| Full-construction timing (Task 1) | `sweep_random_set_ablation_budget_timing` (`run_budget_timing.ps1`) → `experiments/summarize_focal_set_budget_runs.py` | `task_1_focal_set_budget_summary.md` | ✅ |
| **`tab:runtime`** runtime overhead: RS-GNN column | `experiments/benchmark_random_set_layer_overhead.py` | `task_2_random_set_layer_overhead.md` | ✅ matches |
| `tab:runtime`: Energy/ODIN/KNN/Mahalanobis columns | `sweep_vanilla_overhead_benchmark` + `experiments/benchmark_baseline_overhead.py` (`run_vanilla_overhead_benchmark.ps1`) → `experiments/summarize_baseline_overhead.py` | `task_2_baseline_overhead_summary.md`, `baseline_overhead_<ds>.md` | ✅ matches |
| **`tab:isotonic_calibration`** | `sweep_random_set_isotonic_calibration` (`run_isotonic_calibration.ps1`) → `utils/isotonic_calibration.py`, `random_set_trainer.py::_run_isotonic_calibration` → `experiments/summarize_isotonic_calibration.py` | `isotonic_calibration_summary.md`, `{road,nuscenes}_isotonic_calibration_summary.md` | ⚠️ the paper says "5 seeds" and its values differ from the 3-run tables stored here |
| `fig:motivation`, `fig:RSGNN`, `fig:road_example` (TikZ), `fig:dataset_examples` | `Figures/tikz/*.tex`, `Figures/scripts/make_fig_dataset_examples.py` (gitignored in the paper repo) | — | — |
| `plots/before_after_random_set_graph.py`, `plots/random_set_diagnostics.py` | exploratory figures, not in the paper | — | — |

`plots/random_set_results.py` and `experiments/summarize_*` read the **local** `wandb/run-*/files/{config.yaml,wandb-summary.json,output.log}` directories. Those are not in the repo right now because `wandb` is gitignored.

## 6. Other reproducibility hazards

- The `.ps1` scripts hard-code `C:\Users\tommy\anaconda3\envs\graph_uncertainty\python.exe` (Windows, co-author's machine).
- `requirements.txt` has no version pins.
- `vanilla_trainer` seeds with `L.seed_everything`. `random_set_trainer` sets **no seed**.
- `VanillaGNN` uses `act=F.sigmoid` inside the backbone. RS-GNN uses `F.relu`.
- The paper's code link is still the anonymous one (`anonymous.4open.science/r/RS-GNN`). README citation is "TBD".
- `RS-GNN-Camera-Ready-/` is a nested git repo (Overleaf-synced) and shows as untracked in the code repo.
