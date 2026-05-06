# Random-Set Graph Neural Networks

Code for the paper **Random-Set Graph Neural Networks**.

Graph Neural Networks used in safety-critical settings should be able to signal when a node's neighborhood is structurally novel, conflicting, or underrepresented in the training graph. This repository implements **Random-Set Graph Neural Networks (RS-GNN)**, a framework that replaces the usual softmax classification head of a message-passing backbone with a belief function over focal sets. Point predictions are obtained through the pignistic projection, while uncertainty is derived from the induced random-set/credal representation.

The implementation supports node-classification and leave-out-class out-of-distribution experiments on homophilic, heterophilic, large-scale, and temporal road-scene graph benchmarks.

## Abstract

Graph Neural Networks deployed in safety-critical settings must recognise when a node's neighborhood is structurally novel, conflicting, or underrepresented in the training graph; standard softmax classifiers collapse this epistemic ambiguity into a point estimate over known classes, producing confident errors on unseen structure. We propose **Random-Set Graph Neural Networks (RS-GNN)**, a framework that replaces the softmax head of any message-passing backbone with a belief function over a budgeted family of focal sets, extending Random-Set Neural Networks (RS-NN) to relational data. Each node is mapped to a mass function whose pignistic projection yields the point prediction and whose induced credal-set width quantifies epistemic uncertainty. Focal sets are selected from class-overlap statistics in the learned node-embedding space, making the budget graph-aware rather than uniform over the power set. Across nine node-classification benchmarks spanning homophilic and heterophilic regimes (Cora, Coauthor, Reddit2, ArXiv, Patents, Chameleon, Squirrel, Roman Empire, Amazon-Ratings), RS-GNN matches or outperforms strong probabilistic, ensemble, and credal baselines (including Classical Ensemble, GEBM, and CaGCN) on leave-out-class OOD detection, reaching AUROC of 88.84 on Cora and 84.18 on Chameleon. On temporal road-scene graphs constructed from ROAD and nuScenes, RS-GNN improves nuScenes node-classification accuracy from 0.418 to 0.593 and avoids the overconfidence collapse that softmax heads exhibit under cross-dataset shift. These results indicate that credal-set width recovers an epistemic signal that scalar softmax-derived scores cannot, particularly when graph evidence is incomplete or out of distribution.

## Repository Layout

```text
.
├── main.py                         # WandB sweep entry point
├── models/                         # GNN backbones, RS-GNN, and OOD baselines
├── trainers/                       # Training and evaluation routines
│   ├── random_set_trainer.py       # RS-GNN training loop
│   └── budgeting.py                # Embedding-overlap focal-set budgeting
├── dataset_loader/                 # Dataset-specific loaders and split logic
├── sweeps/sweeps.py                # Hyperparameter and dataset sweep configs
├── utils/                          # Scores, solvers, WandB helpers, model manager
├── plots/                          # Plotting scripts and generated figures
├── docker/                         # Docker build/run files
├── requirements.txt                # Python dependencies
└── run_all.sh                      # Example batch commands for experiments
```

## Implemented Methods

The main RS-GNN implementation is in `models/random_set_gnn.py` and `trainers/random_set_trainer.py`. The repository also includes comparison methods used in the experiments:

- `vanilla`: standard softmax GNN
- `random_set`: Random-Set GNN
- `credal`, `credal_LJ`, `frozen`: credal GNN variants
- `ensemble`: classical ensemble baseline
- `odin`, `mahalanobis`, `knn`, `knn_LJ`, `energy`, `gnnsafe`, `gebm`, `cagcn`: OOD and graph uncertainty baselines

Supported GNN backbones for RS-GNN include `GCN`, `SAGE`, `GAT`, `GIN`, and `EdgeCNN`, via PyTorch Geometric.

## Datasets

The experiment entry point currently supports:

- `cora`
- `coauthor`
- `reddit2`
- `arxiv`
- `patents`
- `chameleon`
- `squirrel`
- `roman_empire`
- `amazon_ratings`
- `road`

Dataset files are expected under:

```text
./dataset/
```

Some loaders may download or prepare standard PyTorch Geometric/OGB datasets automatically, depending on the dataset and local cache state.

## Installation

Create and activate a Python environment with PyTorch and PyTorch Geometric installed for your CUDA version. Then install the repository dependencies:

```bash
pip install -r requirements.txt
```

The dependency list includes `torch_geometric`, `torch_sparse`, `torch_scatter`, `lightning`, `wandb`, `faiss-gpu`, and plotting/evaluation utilities. For GPU runs, make sure the installed PyTorch Geometric wheels match your PyTorch and CUDA versions.

## Running Experiments

Experiments are launched through `main.py`. Each run creates or joins a WandB sweep by combining:

1. a model sweep configuration, such as `sweep_random_set`
2. a dataset metadata configuration, such as `metadata_cora`

Run a small RS-GNN sweep on Cora:

```bash
python main.py --dataset cora --model random_set --count 5
```

Run RS-GNN on Chameleon:

```bash
python main.py --dataset chameleon --model random_set --count 5
```

Run a baseline:

```bash
python main.py --dataset cora --model vanilla --count 5
python main.py --dataset cora --model gebm --count 5
python main.py --dataset cora --model cagcn --count 5
```

Use an existing WandB sweep:

```bash
python main.py --dataset cora --model random_set --sweep <SWEEP_ID> --count 10
```

Use a specific sweep object from `sweeps/sweeps.py`, for example an ablation:

```bash
python main.py \
  --dataset cora \
  --model random_set \
  --sweep_name sweep_random_set_ablation_bce_only \
  --count 10
```

Available model names are:

```text
vanilla, credal, ensemble, credal_LJ, odin, mahalanobis, knn,
energy, gnnsafe, knn_LJ, gebm, frozen, cagcn, random_set
```

## RS-GNN Configuration

The RS-GNN sweep configuration is defined in `sweeps/sweeps.py` as `sweep_random_set`. Important options include:

- `gnn_type`: message-passing backbone, for example `GAT`, `GCN`, or `SAGE`
- `hidden_channels`: hidden representation size
- `num_layers`: number of message-passing layers
- `singletons_only`: whether to restrict focal sets to singleton classes
- `loss_ablation`: one of `full`, `bce_only`, `bce_mr`, `bce_ms`, `mr_ms`
- `use_bce_loss`, `use_mr_loss`, `use_ms_loss`: enable or disable RS-GNN loss components

For datasets with many ID classes, the trainer constructs a budgeted focal-set family using learned embedding overlap statistics rather than enumerating the full power set.

## Outputs

Training and evaluation metrics are logged to WandB. RS-GNN logs include:

- `train_loss`, `train_bce`, `train_mr`, `train_ms`, `train_acc`
- `val_loss`, `val_f1`, `val_auroc_entropy`
- `test_acc_id`, `test_auroc_entropy`

Generated result summaries and figures are stored under:

```text
plots/random_set_results/
```

## Docker

A Docker setup based on NVIDIA's PyG image is provided in `docker/`.

```bash
cd docker
./docker-build.sh
./docker-run.sh
```

## Citation

If you use this code, please cite the paper:

```bibtex
@article{random_set_gnn,
  title = {Random-Set Graph Neural Networks},
  author = {TBD},
  journal = {TBD},
  year = {TBD}
}
```

