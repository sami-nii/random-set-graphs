"""Dataset specification and registry.

A dataset is fully described by a :class:`DatasetSpec` living next to its raw
loader in ``cgnn/data/sources/<name>.py``. The spec is the single source of
truth for the leave-out-class OOD protocol (RS-GNN paper Tab. 10) and for the
metadata merged into sweep configs (``in_channels``, ``out_channels``, ...).

``kind="transductive"`` (default): one graph from ``load_raw``, with the
leave-out-class split applied by :func:`cgnn.data.load_dataset`.
``kind="inductive"`` (road-scene benchmarks): ``load_graphs`` returns
``{"train"|"val"|"test": [Data, ...]}`` with labels already encoded
(one-hot over ID classes, all-zero for OOD) and per-graph split masks.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import torch

from cgnn.registry import Registry


@dataclass
class RawGraph:
    """What a source loader returns: the graph with its *original* labels.

    ``train_mask``/``val_mask``/``test_mask`` hold the dataset's public split
    (already indexed to a single split) or ``None`` for random splits.
    """

    x: torch.Tensor
    edge_index: torch.Tensor
    y: torch.Tensor
    train_mask: torch.Tensor | None = None
    val_mask: torch.Tensor | None = None
    test_mask: torch.Tensor | None = None

    @property
    def num_nodes(self) -> int:
        return int(self.x.size(0))


@dataclass(frozen=True)
class DatasetSpec:
    name: str  # CLI / W&B key, e.g. "amazon_ratings"
    paper_name: str  # name used in the paper tables
    load_raw: Callable[[Path, DatasetSpec], RawGraph] | None
    id_classes: tuple[int, ...]
    ood_classes: tuple[int, ...]
    num_features: int
    homophily: Literal["heterophilic", "homophilic", "synthetic", "temporal"]
    split: Literal["public", "random"]
    kind: Literal["transductive", "inductive"] = "transductive"
    # inductive datasets: (data_dir, spec, cfg) -> {"train"|"val"|"test": [Data, ...]}
    load_graphs: Callable[[Path, DatasetSpec, Mapping], dict[str, list]] | None = None
    # classes of the full class space (ID + OOD) for RS-GNN focal singletons; None = ID classes only
    num_total_classes: int | None = None
    # Protocol settings fixed by the dataset (override method defaults, fixed in sweeps),
    # e.g. the road-scene hyperparameters of the RS-GNN paper (Tab. hyperparams).
    protocol: dict = field(default_factory=dict)
    split_ratios: tuple[float, float] = (0.6, 0.2)  # train/val fraction for random splits
    public_split_index: int = 0
    # Defaults merged into every sweep of this dataset .
    batch_size: int = -1  # <= 0: full-batch training
    num_neighbors: int = 10  # NeighborLoader fan-out per layer when batch_size > 0
    # Reddit2: also evaluate with NeighborLoader when batch_size > 0.
    eval_neighbor_sampling: bool = False
    in_paper: bool = True
    notes: str = ""
    # Paths (relative to CGNN_DATA_DIR) that must exist; checked by `cgnn doctor`. "a|b": either one.
    files: tuple[str, ...] = ()
    extra: dict = field(default_factory=dict)

    @property
    def num_id_classes(self) -> int:
        return len(self.id_classes)

    def sweep_metadata(self) -> dict[str, dict]:
        """W&B sweep parameters fixed by the dataset ."""
        meta = {
            "in_channels": {"values": [self.num_features]},
            "out_channels": {"values": [self.num_id_classes]},
            "batch_size": {"values": [self.batch_size]},
            "num_neighbors": {"values": [self.num_neighbors]},
        }
        if self.num_total_classes is not None:
            meta["num_total_classes"] = {"values": [self.num_total_classes]}
        meta.update({k: {"values": [v]} for k, v in self.protocol.items()})
        return meta

    def missing_files(self, data_dir: Path) -> list[str]:
        return [f for f in self.files if not any((data_dir / alt).exists() for alt in f.split("|"))]

    def default_config(self) -> dict:
        return {k: v["values"][0] for k, v in self.sweep_metadata().items()}


DATASETS: Registry[DatasetSpec] = Registry("dataset")


def register_dataset(spec: DatasetSpec) -> DatasetSpec:
    return DATASETS.register(spec.name, spec)
