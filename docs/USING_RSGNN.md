# Using RS-GNN on your own data

There are two ways to use RS-GNN outside the paper's benchmarks:

1. **Python API** (`cgnn.rsgnn`): any graph given as tensors. No configs, registries or W&B needed.
2. **Registered dataset**: add your graph to the dataset registry to get the `cgnn` CLI, sweeps, every
   baseline and the result tables.

A runnable version of section 1 is in `examples/rsgnn_on_your_graph.py` (a few seconds on CPU).

## 1. Python API

```python
from cgnn import rsgnn

# x [N, F] float, edge_index [2, E] long, y [N] integer labels, boolean masks [N]
data = rsgnn.prepare_graph(
    x, edge_index, y, train_mask, val_mask, test_mask,
    id_classes=[2, 3, 4],      # classes the model is trained on
    ood_classes=[0, 1],        # held-out classes: never used for training, present at val/test time
)

families = rsgnn.focal_sets(data)                 # focal-set family (see "Focal sets" below)
model = rsgnn.fit(data, families=families)        # trains with the paper's defaults
out = rsgnn.predict(model, data)                  # outputs for every node
out["probs"]          # [N, C] pignistic probabilities BetP (rows sum to 1)
out["prediction"]     # [N]    argmax of BetP, as an index into sorted(id_classes)
out["entropy"]        # [N]    pignistic entropy: total uncertainty
out["credal_width"]   # [N]    width of the credal set of the predicted class: epistemic uncertainty
```

`prepare_graph` applies the paper's leave-out-class protocol:
- `y` becomes one-hot over the sorted ID classes, with all-zero rows for OOD nodes;
- training uses ID nodes only;
- validation and test keep ID and OOD nodes.

If you have no OOD classes, pass `ood_classes=()`. The scores still rank nodes by uncertainty.
`fit` early-stops on the ID validation loss. The validation split needs OOD nodes only for the
validation AUROC.

**Configuration.** `fit` and `focal_sets` accept any key of `configs/methods/random_set.yaml` as a
keyword argument, for example:

```python
model = rsgnn.fit(data, lr=1e-2, max_epochs=300, gnn_type="SAGE", accelerator="cpu")
```

| Key | Default | Meaning |
|---|---|---|
| `gnn_type` | `GATv2` | backbone: `GATv2`, `GAT`, `GCN`, `SAGE`, `GIN` |
| `hidden_channels`, `num_layers` | 128, 2 | backbone size |
| `gat_heads`, `act`, `dropout` | 4, `relu`, 0.2 | GATv2 heads (first layer; the last layer has 1), activation, dropout |
| `lr`, `weight_decay`, `max_epochs`, `patience` | 1e-3, 1e-4, 200, 50 | optimisation and early stopping (`val_loss`) |
| `alpha`, `beta` | 1e-3, 1e-3 | weights of the negative-mass and sum-to-one regularisers (Eq. 4) |
| `focal_set_strategy` | `auto` | `auto`, `budgeted`, `singletons` |
| `budget_k` | 64 | non-singleton focal sets of the budgeted family |
| `seed` | 42 | global seed |

**Large graphs.** The API trains full-batch. For graphs that do not fit in GPU memory, register the
dataset (section 2) with `batch_size > 0` to train with NeighborLoader.

**Saving and loading.** `RandomSetGNN` is a LightningModule, and its focal sets are stored in the
hyper-parameters:

```python
import lightning as L
from cgnn.models.random_set import RandomSetGNN

trainer = L.Trainer(logger=False); trainer.strategy.connect(model)
trainer.save_checkpoint("rsgnn.ckpt")
model = RandomSetGNN.load_from_checkpoint("rsgnn.ckpt")
```

## Focal sets

The belief head predicts one belief value per focal set (a set of classes), so the focal family fixes
the output size.

- `auto` (default): the full power set when there are fewer than 10 ID classes (`2^C - 1` sets).
  Otherwise it uses the **budgeted** family.
- `budgeted`: all singletons plus the `budget_k` pairs and triples of classes that overlap most in the
  embedding space of a warm-up GNN. The steps are: up to 200 labelled ID nodes per class, 3-D t-SNE,
  one Gaussian per class, then the Jaccard overlap of their 95% ellipsoids (paper Alg. 1). OOD labels
  are never used. The output size is `C + K`.
- `singletons`: one set per class. The credal width is then always 0 (paper Prop. 1 (iv)), so use
  this only as an ablation.

`rsgnn.focal_sets(data)` returns the family as a list of class-index lists, so you can inspect it.
`rsgnn.fit(data, families=...)` reuses it.

## Interpreting the scores

- **Pignistic entropy** (`entropy`) is the entropy of BetP. It is high when the class probabilities
  are spread out, for whatever reason.
- **Credal width** (`credal_width`) is `Pl({ŷ}) - Bel({ŷ})`: the mass the model gave to sets of classes
  that contain the predicted class but that it could not commit to that class alone. It is the
  epistemic score of the paper.

In the paper, the two scores agree closely on most benchmarks (Tab. 1). On ROAD the entropy is the
informative one (Tab. 3). Use a validation split with OOD nodes to choose between them.

## 2. Registering a dataset (CLI, sweeps, baselines)

Create `cgnn/data/sources/my_graph.py`:

```python
from pathlib import Path
import torch
from cgnn.data.spec import DatasetSpec, RawGraph, register_dataset


def _load(data_dir: Path, spec: DatasetSpec) -> RawGraph:
    blob = torch.load(data_dir / "my_graph.pt")          # your own format
    return RawGraph(x=blob["x"], edge_index=blob["edge_index"], y=blob["y"])  # original labels


register_dataset(
    DatasetSpec(
        name="my_graph",
        paper_name="My graph",
        load_raw=_load,
        id_classes=(2, 3, 4),
        ood_classes=(0, 1),
        num_features=128,
        homophily="heterophilic",
        split="random",             # 60/20/20 with split_seed; or "public" and return the masks in RawGraph
        files=("my_graph.pt",),     # checked by `cgnn doctor`
        batch_size=-1,              # > 0: NeighborLoader mini-batches of this size
    )
)
```

Then add `my_graph` to the import list in `cgnn/data/sources/__init__.py`. Now:

```bash
cgnn data my_graph                                         # split sizes
cgnn run -m random_set -d my_graph --wandb disabled        # one run, summary in outputs/runs/
cgnn run -m vanilla -d my_graph --seeds 0 1 2              # backbones for the post-hoc baselines
cgnn run -m energy -d my_graph                             # any baseline (see `cgnn list methods`)
cgnn sweep -m random_set -d my_graph -c 30                 # Bayesian sweep (needs W&B)
```

Graph collections, where every sample is a graph as in the road-scene benchmarks, use
`kind="inductive"` with a `load_graphs` function. See `cgnn/data/sources/road_scene.py`.
