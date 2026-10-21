"""RS-GNN on your own graph, without the dataset/method registries, YAML configs or W&B.

    from cgnn import rsgnn

    data = rsgnn.prepare_graph(x, edge_index, y, train_mask, val_mask, test_mask,
                               id_classes=[2, 3, 4], ood_classes=[0, 1])
    model = rsgnn.fit(data, max_epochs=100)                  # focal sets + training
    out = rsgnn.predict(model, data)                          # per-node outputs
    out["probs"], out["entropy"], out["credal_width"], out["prediction"]

``fit`` uses the RS-GNN defaults of ``configs/methods/random_set.yaml`` (GATv2 backbone, focal-set
budget, loss weights); any key can be overridden as a keyword argument, e.g. ``lr=1e-2`` or
``focal_set_strategy="budgeted"``. ``predict`` returns tensors for *all* nodes of the graph.
"""

from __future__ import annotations

from collections.abc import Sequence
from types import SimpleNamespace
from typing import Any

import torch
from torch_geometric.data import Data

from cgnn.config import resolve_config
from cgnn.data.ood import apply_leave_out_classes

__all__ = ["config", "fit", "focal_sets", "predict", "prepare_graph"]


def prepare_graph(
    x: torch.Tensor,
    edge_index: torch.Tensor,
    y: torch.Tensor,
    train_mask: torch.Tensor,
    val_mask: torch.Tensor,
    test_mask: torch.Tensor,
    id_classes: Sequence[int],
    ood_classes: Sequence[int] = (),
) -> Data:
    """Graph with the leave-out-class protocol of the paper.

    ``y`` holds the original integer labels. The result has ``y`` one-hot over the (sorted, re-indexed)
    ID classes with all-zero rows for OOD nodes, ``train_mask`` restricted to ID nodes, and the
    validation/test masks unchanged (ID + OOD). Nodes whose label is in neither list are dropped from
    every mask.
    """
    return apply_leave_out_classes(
        x.float(), edge_index.long(), y.long(), train_mask, val_mask, test_mask, id_classes, ood_classes
    )


def config(data: Data, **overrides: Any) -> dict[str, Any]:
    """Resolved RS-GNN run config for ``data`` (defaults of ``configs/methods/random_set.yaml``)."""
    dataset_defaults = {
        "in_channels": int(data.num_features),
        "out_channels": int(data.y.size(1)),
        "batch_size": -1,
    }
    return resolve_config("random_set", dataset_defaults, overrides)


def _bundle(data: Data):
    from torch_geometric.loader import DataLoader

    loader = DataLoader([data], batch_size=1, shuffle=False)
    return SimpleNamespace(data=data, train_loader=loader, val_loader=loader, test_loader=loader)


def focal_sets(data: Data, **overrides: Any) -> list[list[int]]:
    """Focal-set family for ``data``: power set below 10 ID classes, otherwise the graph-aware budget."""
    import lightning as L

    from cgnn.methods.random_set import build_focal_sets
    from cgnn.paths import Paths

    cfg = config(data, **overrides)
    L.seed_everything(int(cfg["seed"]), workers=True)
    ctx = SimpleNamespace(cfg=cfg, bundle=_bundle(data), paths=Paths.from_env())
    families, _ = build_focal_sets(ctx)
    return families


def fit(data: Data, *, families: list[list[int]] | None = None, **overrides: Any):
    """Build and train RS-GNN on ``data`` (full batch); returns the trained ``RandomSetGNN``.

    ``families``: a precomputed focal-set family (see :func:`focal_sets`); by default it is built from
    ``data``. Early stopping follows the RS-GNN protocol (``val_loss``, patience 50). Pass
    ``accelerator="cpu"`` to force CPU, ``max_epochs=...`` to bound training.
    """
    import lightning as L
    from lightning.pytorch.callbacks import EarlyStopping

    from cgnn.methods.random_set import random_set_builder
    from cgnn.paths import Paths

    cfg = config(data, **overrides)
    L.seed_everything(int(cfg["seed"]), workers=True)
    bundle = _bundle(data)
    ctx = SimpleNamespace(cfg=cfg, bundle=bundle, paths=Paths.from_env())
    if families is not None:
        from cgnn.methods.trainable import _backbone_kwargs
        from cgnn.models.random_set import RandomSetGNN

        model = RandomSetGNN(
            **_backbone_kwargs(cfg),
            weight_decay=cfg["weight_decay"],
            focal_sets=families,
            alpha=cfg["alpha"],
            beta=cfg["beta"],
            loss_ablation=cfg["loss_ablation"],
        )
    else:
        model = random_set_builder()(ctx)
    callbacks = [EarlyStopping(monitor=cfg["monitor"], patience=cfg["patience"], mode=cfg["mode"])]
    L.Trainer(
        accelerator=cfg.get("accelerator", "auto"),
        devices=1,
        max_epochs=cfg["max_epochs"],
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=bool(cfg.get("progress_bar", True)),
        enable_model_summary=False,
        callbacks=callbacks,
        num_sanity_val_steps=0,
    ).fit(model, bundle.train_loader, bundle.val_loader)
    return model.eval()


@torch.no_grad()
def predict(model, data: Data) -> dict[str, torch.Tensor]:
    """Per-node pignistic probabilities, prediction, pignistic entropy (total) and credal width (epistemic)."""
    model = model.eval()
    data = data.to(model.device)
    probs, entropy, width = model.predict(model(data))
    return {
        "probs": probs.cpu(),
        "prediction": probs.argmax(dim=1).cpu(),
        "entropy": entropy.cpu(),
        "credal_width": width.cpu(),
    }
