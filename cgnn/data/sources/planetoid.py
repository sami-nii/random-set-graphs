"""Cora (Planetoid, Yang et al. 2016) with the ``split="full"`` public masks.

RS-GNN paper only: ID classes {4, 5, 6}, OOD classes {0, 1, 2, 3}.
"""

from __future__ import annotations

from pathlib import Path

from cgnn.data.spec import DatasetSpec, RawGraph, register_dataset


def _load_planetoid(data_dir: Path, spec: DatasetSpec) -> RawGraph:
    from torch_geometric.datasets import Planetoid

    data = Planetoid(root=str(data_dir), name=spec.extra["pyg_name"], split="full")[0]
    return RawGraph(
        x=data.x,
        edge_index=data.edge_index,
        y=data.y,
        train_mask=data.train_mask,
        val_mask=data.val_mask,
        test_mask=data.test_mask,
    )


register_dataset(
    DatasetSpec(
        name="cora",
        files=("Cora/raw",),
        paper_name="Cora",
        load_raw=_load_planetoid,
        id_classes=(4, 5, 6),
        ood_classes=(0, 1, 2, 3),
        num_features=1433,
        homophily="homophilic",
        split="public",
        notes="RS-GNN paper only (not in the CGNN paper).",
        extra={"pyg_name": "Cora"},
    )
)
