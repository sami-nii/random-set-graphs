# Reproducing the paper

Step-by-step guide to reproduce *Random-Set Graph Neural Networks* (NeurIPS 2026) from a fresh clone.
`docs/PAPER_CODE_MAP.md` maps every equation, algorithm and table to the code, and lists the points
where the paper leaves choices open.

## 0. What you need

- Linux with an NVIDIA GPU. The paper used A100-40GB GPUs. Everything also runs on CPU (`--cpu`),
  just slower.
- About 4 GB of disk for the datasets (3.1 GB for the nine standard benchmarks, Reddit2 alone 2 GB).
- A [Weights & Biases](https://wandb.ai) account for **sweeps** and the **result tables**. Single runs
  work offline (`--wandb disabled`).
- Compute: the full set of experiments took about 1,000 A100 GPU-hours (paper App. D.2). Section 5
  gives a reduced plan.

## 1. Install

```bash
git clone --recurse-submodules https://github.com/sami-nii/random-set-graphs.git
cd random-set-graphs
conda env create -f environment.yml && conda activate rsgnn
pip install -e ".[all]"
pip install --no-build-isolation -e graph-ebm      # GEBM baseline only
```

`environment.yml` pins the versions used for the experiments: Python 3.11, torch 2.6.0 + CUDA 12.4,
PyG 2.7.0, Lightning 2.6.0. If you already cloned without submodules, run
`git submodule update --init graph-ebm`.

Alternatively, use the NVIDIA PyG container (`nvcr.io/nvidia/pyg:24.05-py3`):
`bash docker/docker-build.sh && bash docker/docker-run.sh`.

Check the installation:

```bash
python -m pytest -q -m "not slow"     # about 10 s on CPU; every method runs end-to-end on a toy graph
cgnn doctor                           # dependencies, datasets, checkpoints, GPUs
cgnn run -m random_set -d synthetic --cpu --wandb disabled     # RS-GNN in a few seconds
```

## 2. Data

```bash
bash scripts/download_data.sh         # every dataset that can be downloaded automatically
```

| Dataset | Source | How |
|---|---|---|
| Cora, Chameleon, Squirrel | PyG | automatic |
| Coauthor-CS | gnn-benchmark | automatic |
| ArXiv-year | OGB (`ogbn-arxiv`) | automatic |
| Reddit2 | GraphSAINT (Google Drive) | automatic |
| Amazon-Ratings, Roman Empire | [heterophilous-graphs](https://github.com/yandex-research/heterophilous-graphs) | automatic (`download_data.sh`) |
| Patents-year | `snap-patents.mat` from [LINKX](https://github.com/CUAI/Non-Homophily-Large-Scale) | manual → `dataset/snap-patents.mat` |
| ROAD | `road_trainval_v1.0.json` from [ROAD](https://github.com/gurkirt/road-dataset) | manual → `dataset/road/` |
| nuScenes | `v1.0-mini` metadata from [nuScenes](https://www.nuscenes.org/download) (free registration) | manual → `dataset/nuscenes/v1.0-mini/` |

Only the annotation files of ROAD and nuScenes are used, not images or LiDAR. The temporal scene graphs
are built and cached on first use.

`cgnn data <dataset>` prints the ID/OOD split sizes. They should match paper Tab. 10; for example,
Cora gives `train(ID)=421 | val ID/OOD=167/333 | test ID/OOD=316/684`.

## 3. Weights & Biases

```bash
wandb login
export PROJECT=rs-gnn     # any project name; pass the same one to `cgnn results`
```

Every sweep run and its config are logged to the project. `cgnn results` reads them back to build
the tables. Runs also write a JSON summary to `outputs/runs/`, which the `experiments/summarize_runs.py`
tables read without W&B.

## 4. Experiments

`scripts/reproduce_paper.sh` contains the whole pipeline in dependency order. By default it only prints
the commands; `EXECUTE=1` runs them, and `STEPS`, `DATASETS` and the `*_RUNS` variables select subsets.
Every command uses one GPU; never run more than two in parallel on a shared machine.

```bash
bash scripts/reproduce_paper.sh                                    # print everything
STEPS="ood" DATASETS="cora" EXECUTE=1 bash scripts/reproduce_paper.sh
```

| Paper | Step | What runs | Output |
|---|---|---|---|
| **Tab. 1**: OOD AUROC on 9 benchmarks (Figs. 5–7) | `ood` | per dataset: `vanilla` sweep (backbones) → post-hoc baselines → CaGCN, CGNN post train → credal methods and `random_set` | `cgnn results --format latex` → `outputs/tables/ood.tex` |
| **Tabs. 2–3**: ROAD, nuScenes, ROAD→nuScenes | `road` | `vanilla` and `random_set` with the fixed protocol of Tab. 8 | `experiments/summarize_runs.py road` |
| **Tab. 4**: ablations (Fig. 10) | `ablation` | `random_set_{singletons,bce_only,bce_mr}` on Cora and Roman Empire | `cgnn results -m ...` |
| **Tab. 12**: isotonic calibration | `isotonic` | best `random_set` config of each sweep (`SWEEP_<dataset>=<id>`), 5 seeds | `experiments/summarize_runs.py isotonic` |
| **Tab. 7**: focal-set budget vs K | `cost` | `experiments/benchmark_focal_set_budget.py` | `outputs/tables/focal_budget_<ds>.md` |
| **Tab. 9**: per-forward overhead | `cost` | 3 vanilla backbones, then `experiments/benchmark_overhead.py` | `outputs/tables/overhead_<ds>.md` |
| **Tab. 11**: feature sparsity | — | `python experiments/feature_sparsity.py` | `outputs/tables/feature_sparsity/` |

**How Tab. 1 numbers are selected.** For each method and dataset, `cgnn results` takes the sweep runs
with the best validation score. It picks the uncertainty score with the best validation AUROC
(entropy or credal width for RS-GNN; EU/AU for credal methods) and reports the mean ± std of the test
AUROC over the top-5 runs (`-k`). Test labels are never used for selection.

**Order matters.** Post-hoc baselines, the ensembles, CaGCN and CGNN post train load the best `vanilla`
checkpoints of the same dataset (`checkpoints/vanilla/`), so run the `vanilla` sweep first.

## 5. A reduced reproduction

For a course project or a reproducibility challenge, this subset covers the paper's main claims on
small graphs at a small fraction of the full cost (three small datasets, fewer sweep runs):

```bash
export DATASETS="cora chameleon squirrel"
# RS-GNN vs. the strongest baselines of Tab. 1 (fewer sweep runs than the paper)
STEPS="ood" VANILLA_RUNS=20 TRAINABLE_RUNS=10 EXECUTE=1 bash scripts/reproduce_paper.sh
# ablations (Tab. 4) on Cora
STEPS="ablation" TRAINABLE_RUNS=10 EXECUTE=1 bash scripts/reproduce_paper.sh
cgnn results -p rs-gnn -d $DATASETS --details
```

To run RS-GNN alone with fixed hyper-parameters over several seeds, without W&B:

```bash
cgnn run -m random_set -d chameleon --seeds 0 1 2 3 4 --wandb disabled --set lr=0.005
python experiments/summarize_runs.py budget          # focal-set family sizes and construction time
```

## 6. Reproducibility notes

- **Seeds.** Each run seeds Python, NumPy and torch from `seed` (default 42), and the focal-set
  sampling and t-SNE from the same seed. The random 60/20/20 splits (ArXiv, Patents, Coauthor) use
  `split_seed: 0`, so every method sees the same split.
- **Determinism.** Trainable runs use Lightning's deterministic mode (`deterministic: true` in
  `configs/defaults.yaml`). Results can still differ slightly across GPU models and library versions;
  `environment.yml` pins the versions used for the paper.
- **Backbone.** Every trainable model, and so every post-hoc backbone, uses the paper's GATv2 encoder
  (App. D.2). The baseline rows of Tab. 1 on the seven datasets shared with the CGNN paper were taken
  from that paper, which used GCN/SAGE backbones (see `docs/PAPER_CODE_MAP.md`). Re-running the
  baselines here gives GATv2 numbers, which are not expected to match those rows exactly.
- **Memory.** Post-hoc methods on Reddit2 run full-graph inference (about 52 GB): use `--cpu`. The
  focal-set budget voxelises class ellipsoids on a 3-D grid. That takes seconds on a GPU and minutes on
  CPU.

## 7. Troubleshooting

| Symptom | Fix |
|---|---|
| `No vanilla checkpoints for '<ds>'` | train backbones first: `cgnn sweep -m vanilla -d <ds>` (or `cgnn run -m vanilla -d <ds> --seeds 0 1 2`) |
| `ModuleNotFoundError: graph_uq` | GEBM only: `git submodule update --init graph-ebm && pip install --no-build-isolation -e graph-ebm` |
| `ModuleNotFoundError: faiss` | `pip install faiss-cpu` (KNN, JLDE) |
| CUDA out of memory | add `--cpu`, or use a dataset with `batch_size > 0` |
| `cgnn doctor` reports `MISS` | the dataset files are not under `$CGNN_DATA_DIR` (section 2) |
| `devices > 2` refused | intended: the runner never uses more than 2 GPUs |
