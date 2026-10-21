# Random-Set Graph Neural Networks

Code for **Random-Set Graph Neural Networks** (RS-GNN), NeurIPS 2026.

Show a graph neural network a node from a class it has never seen, and it will confidently name one of
the classes it knows. RS-GNN replaces the softmax head of a message-passing GNN with a **belief head**
that assigns mass to *sets* of classes (focal sets). Each prediction is a belief function, i.e. a credal
set of class distributions: its pignistic centre gives the label, its width measures what the model
does not know. A small, graph-aware family of focal sets, chosen from classes that overlap in the
learned embedding space, keeps the head tractable.

The repository is built on [`cgnn`](https://github.com/matteotolloso/graph-uncertainty), the framework
of *Credal Graph Neural Networks*: the same leave-out-class OOD protocol, runner, baselines and tests,
extended with RS-GNN, Cora/Chameleon and the temporal road-scene benchmarks.

**Documentation**
- [docs/REPRODUCING.md](docs/REPRODUCING.md): step-by-step reproduction of every table, including a reduced plan.
- [docs/USING_RSGNN.md](docs/USING_RSGNN.md): RS-GNN on your own graph (Python API, custom datasets).
- [docs/PAPER_CODE_MAP.md](docs/PAPER_CODE_MAP.md): paper equations, algorithms and tables → code.

## Install
```bash
git clone --recurse-submodules https://github.com/sami-nii/random-set-graphs.git && cd random-set-graphs
conda env create -f environment.yml && conda activate rsgnn    # or an env with torch + PyG
pip install -e ".[all]"                                        # the package, CLI, optional deps, tests
git submodule update --init graph-ebm && pip install --no-build-isolation -e graph-ebm   # GEBM baseline
wandb login                                                    # sweeps and result tables use W&B
cgnn doctor                                                    # checks deps, data, checkpoints, GPUs
```
Docker (NVIDIA PyG image `nvcr.io/nvidia/pyg:24.05-py3`): `bash docker/docker-build.sh && bash docker/docker-run.sh`.

Datasets live under `dataset/` (`CGNN_DATA_DIR`). `bash scripts/download_data.sh` fetches every
dataset that can be downloaded and prints instructions for the manual ones (Patents, ROAD, nuScenes).
Checkpoints go to `checkpoints/` (`CGNN_CKPT_DIR`), run summaries to `outputs/` (`CGNN_OUTPUT_DIR`).

## Quick start
```bash
cgnn list                                                         # methods and datasets
cgnn run -m random_set -d synthetic --cpu --wandb disabled        # RS-GNN end-to-end in seconds
cgnn run -m random_set -d cora --seeds 0 1 2                      # single runs on a real dataset (1 GPU)
cgnn describe random_set                                          # defaults and sweep space
python -m pytest -q -m "not slow"                                 # test suite on CPU
```

## RS-GNN on your own graph
```python
from cgnn import rsgnn

data = rsgnn.prepare_graph(x, edge_index, y, train_mask, val_mask, test_mask,
                           id_classes=[2, 3, 4], ood_classes=[0, 1])   # leave-out-class protocol
model = rsgnn.fit(data)                     # focal sets + training with the paper's defaults
out = rsgnn.predict(model, data)            # probs, prediction, entropy (total), credal_width (epistemic)
```
Runnable example: `python examples/rsgnn_on_your_graph.py`. Details: [docs/USING_RSGNN.md](docs/USING_RSGNN.md).

## Reproducing the paper
Full guide: [docs/REPRODUCING.md](docs/REPRODUCING.md). `bash scripts/reproduce_paper.sh` prints the full
pipeline (`EXECUTE=1` runs it; `STEPS=...` selects steps).

| Paper | Step | Commands |
|---|---|---|
| Tab. 1 (OOD AUROC, 9 benchmarks), Figs. 5-7 | `ood` | `cgnn sweep -m vanilla`, then post-hoc / credal baselines and `random_set`; `cgnn results` |
| Tabs. 2-3 (ROAD, nuScenes, ROAD→nuScenes) | `road` | `cgnn run -m {vanilla,random_set} -d {road,nuscenes,road_to_nuscenes}`; `experiments/summarize_runs.py road` |
| Tab. 4 / Fig. 10 (ablations) | `ablation` | `random_set_{singletons,bce_only,bce_mr}` on Cora and Roman Empire |
| Tab. 12 (isotonic calibration) | `isotonic` | `cgnn run -m random_set_isotonic --seeds 0 1 2 3 4 --from-sweep <id>`; `experiments/summarize_runs.py isotonic` |
| Tab. 7 (focal-set budget vs K) | `cost` | `experiments/benchmark_focal_set_budget.py` |
| Tab. 9 (per-forward overhead) | `cost` | `experiments/benchmark_overhead.py` (after 3 vanilla backbones) |
| Tab. 10 (dataset statistics), Tab. 11 (feature sparsity) | — | `cgnn data <dataset>`; `experiments/feature_sparsity.py` |

`results/legacy/` holds the tables and figures produced by the earlier version of this code (before the
port to `cgnn`), for reference.

## RS-GNN in the code
| Paper | Code |
|---|---|
| Belief head `Bel(A) = σ(g(z_v))`, Möbius inversion (Eq. 1) | `cgnn/models/random_set.py` (`RandomSetGNN.forward`, `valid_masses`) |
| Clip → residual to Θ → renormalise → BetP (Fig. 11) | `RandomSetGNN.valid_masses`, `RandomSetGNN.predict` |
| Pignistic entropy (Eq. 2), credal width (Eq. 3) | `RandomSetGNN.predict` → scores `entropy`, `credal` |
| Loss `BCE + α R_neg + β R_sum` (Eq. 4) and its ablations | `RandomSetLoss`, `LOSS_ABLATIONS` |
| Focal sets: power set (\|Y\| < 10) or budgeted (Alg. 1) | `cgnn/models/focal_sets.py`, `cgnn/methods/random_set.py` |
| Warm-up GNN for the budget | `train_warmup_backbone` (same backbone, labelled ID nodes only) |
| Post-hoc isotonic calibration | `cgnn/uncertainty/isotonic.py` (`random_set_isotonic`) |
| GATv2 backbone (App. D.2) | `cgnn/models/backbones.py` (`GATv2`), `configs/methods/*.yaml` |
| Metrics: AUROC/AUPRC/FPR95, ECE (15 bins)/NLL/Brier (App. D.11) | `cgnn/metrics.py` |

Key options of `configs/methods/random_set.yaml`: `focal_set_strategy` (`auto` | `budgeted` | `singletons`),
`budget_k` (64), `budget_max_cardinality` (3), `budget_samples_per_class` (200), `budget_warmup_epochs`,
`alpha`/`beta` (1e-3), `loss_ablation`, `isotonic_calibration`. RS-GNN logs `val_auroc_{entropy,credal}`,
`test_{auroc,auprc,fpr95}_{entropy,credal}`, `test_id_{acc,ece,nll,brier}` and `focal_set_budget_seconds`.

## Methods
| Paper | Key | Kind |
|---|---|---|
| **RS-GNN** | `random_set` | trainable |
| RS-GNN ablations / + isotonic | `random_set_singletons`, `random_set_bce_only`, `random_set_bce_mr`, `random_set_isotonic` | trainable |
| Credal final / CredalLJ / Credal frozen | `credal`, `credal_LJ`, `frozen` | trainable / credal head on a frozen backbone |
| Credal (dual head) | `credal_LJ_dual_head_detached` | trainable |
| Classical ensemble / Credal ensemble | `ensemble` (`*_classic` / `*_credal` metrics) | post-hoc |
| Energy, ODIN, Mahalanobis, KNN | `energy`, `odin`, `mahalanobis`, `knn` | post-hoc |
| GNNSafe, GEBM, JLDE | `gnnsafe`, `gebm`, `knn_LJ` | post-hoc |
| CaGCN | `cagcn` | post-hoc, calibrated on ID validation nodes |
| Vanilla softmax GNN (backbone of the post-hoc methods) | `vanilla` | trainable |

Every trainable model, and therefore every post-hoc backbone, uses the paper's backbone: a two-layer
GATv2 with hidden size 128, 4 attention heads in the first layer and 1 in the second, ReLU and dropout
0.2. The vanilla model puts a linear softmax classifier on the node embedding.

## Datasets
| Dataset | Key | ID / OOD classes | Split |
|---|---|---|---|
| Chameleon, Squirrel | `chameleon`, `squirrel` | {2,3,4} / {0,1} | public (split 0) |
| ArXiv-year, Patents-year | `arxiv`, `patents` | {2,3,4} / {0,1} | random 60/20/20 |
| Amazon-Ratings | `amazon_ratings` | {2,3,4} / {0,1} | public (split 0) |
| Roman Empire | `roman_empire` | {5..17} / {0..4} | public (split 0) |
| Coauthor-CS | `coauthor` | {4..14} / {0..3} | random 60/20/20 |
| Reddit2 | `reddit2` | {11..40} / {0..10} | public (GraphSAINT) |
| Cora | `cora` | {4,5,6} / {0,1,2,3} | public (`split="full"`) |
| ROAD (temporal scene graphs) | `road` | agents {0..7} / {8,9} | `train_1` videos / `val_1` videos halved into val/test |
| nuScenes (temporal scene graphs) | `nuscenes` | categories {0..5} / {6,7} | v1.0-mini scenes: 8 train, 1 val, 1 test |
| ROAD (ID) vs nuScenes (OOD) | `road_to_nuscenes` | all ROAD / nuScenes test agents | as ROAD; nuScenes test graphs added to test |

Road-scene data (not downloaded automatically): put `road_trainval_v1.0.json` of
[ROAD](https://github.com/gurkirt/road-dataset) in `dataset/road/` and the
[nuScenes](https://www.nuscenes.org/) `v1.0-mini` metadata tables in `dataset/nuscenes/v1.0-mini/`.
Window graphs are cached next to the raw data. Construction, features and splits:
`cgnn/data/sources/road_scene.py`.

## Protocol
Leave-out-class OOD detection, transductive on the standard benchmarks: the whole graph is visible during
message passing, models are trained on the ID training nodes, OOD nodes have all-zero label rows, and
AUROC treats OOD as the positive class. Road-scene benchmarks are inductive (batches of 16 window graphs).
Hyper-parameters are selected on a validation split with ID and OOD nodes. Configuration is resolved as
`configs/defaults.yaml` → `configs/methods/<method>.yaml` → dataset metadata/protocol → sweep values / `--set`.
Differences from the CGNN protocol: the GATv2 backbone above, and `split_seed: 0`, so the random-split
datasets use one split shared by every method.

## Layout
```
cgnn/models/       RS-GNN (random_set.py, focal_sets.py), vanilla GNN, credal models, CaGCN, detectors/
cgnn/methods/      method registry (random_set.py, trainable.py, posthoc.py)
cgnn/data/         dataset registry, leave-out-class split, loaders; sources/ has one file per dataset
cgnn/uncertainty/  credal intervals and entropies, ensembles, isotonic calibration
cgnn/runner.py     generic runner; cgnn/cli.py is the `cgnn` command; results.py builds W&B tables
configs/           defaults.yaml and one YAML per method (defaults + W&B sweep space)
experiments/       focal-set budget, overhead and feature-sparsity scripts, summaries of local runs
examples/          rsgnn_on_your_graph.py: the Python API on a custom graph
cgnn/rsgnn.py      Python API: prepare_graph, focal_sets, fit, predict
scripts/           reproduce_paper.sh, download_data.sh
tests/             pytest suite (every method runs end-to-end on a synthetic graph)
docs/              REPRODUCING.md, USING_RSGNN.md, PAPER_CODE_MAP.md
docker/            NVIDIA PyG container
graph-ebm/         GEBM baseline (git submodule)
results/legacy/    outputs of the pre-port code (reference only)
reference/         RS-NN CIFAR-10 notebook the focal-set budget is derived from
```

## Hardware
Runs use one GPU by default and pin an idle one automatically (`--gpus N`, at most 2; `--cpu` for CPU).
The paper's experiments ran on A100-40GB GPUs. Post-hoc methods on Reddit2 need `--cpu` or a large GPU.

## Citation
```bibtex
@inproceedings{woodley2026randomset,
  title     = {Random-Set Graph Neural Networks},
  author    = {Woodley, Tommy and Tolloso, Matteo and Bacciu, Davide and Kudukkil Manchingal, Shireen and Cuzzolin, Fabio},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2026}
}
```

## Acknowledgements
Built on the `cgnn` code of *Credal Graph Neural Networks*. The focal-set budget follows Random-Set
Neural Networks (Manchingal et al.). Dataset classes for Coauthor/Reddit2 and the year-based labels follow
Ma et al. (NeurIPS 2024). GEBM uses the official `graph-ebm` implementation.
