"""RS-GNN (the paper's method): focal-set construction and method registration.

Focal-set strategy (config ``focal_set_strategy``):

- ``auto`` (standard benchmarks): full power set of the ID classes when
  ``|Y| < power_set_max_classes`` (10), otherwise the budgeted family;
- ``budgeted`` (road-scene benchmarks): always the budgeted family;
- ``singletons``: singletons only (ablation).

``num_total_classes`` > ``|Y|`` adds singletons for the remaining classes of the
full class space (road-scene benchmarks, Tab. focal_sets); OOD labels are never
used to build the family.
"""

from __future__ import annotations

import time
from typing import Any

from cgnn.methods.spec import MethodSpec, RunContext, register_method
from cgnn.methods.trainable import _backbone_kwargs

RANDOM_SET_SCORES = ("entropy", "credal")


def train_warmup_backbone(cfg: dict[str, Any], bundle, output_dir):
    """Warm-up GNN with the backbone architecture, trained on labelled ID nodes (Alg. focal_sets)."""
    import lightning as L

    from cgnn.models.vanilla import VanillaGNN

    warmup = VanillaGNN(
        **_backbone_kwargs(cfg), weight_decay=cfg.get("weight_decay", 0.0), head="linear", ood_in_val=False
    )
    accelerator = cfg.get("accelerator", "auto")
    L.Trainer(
        accelerator=accelerator,
        devices=1,
        max_epochs=int(cfg.get("budget_warmup_epochs", 20)),
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        limit_val_batches=0,
        num_sanity_val_steps=0,
        deterministic=bool(cfg.get("deterministic", True)),
        default_root_dir=str(output_dir / "lightning"),
    ).fit(warmup, bundle.train_loader)
    return warmup.eval()


def budget_embeddings(cfg: dict[str, Any], bundle, output_dir):
    """Embeddings of up to ``budget_samples_per_class`` labelled ID training nodes per class (and labels),
    from their induced subgraph and a trained warm-up GNN: the K-independent part of Alg. focal_sets."""
    import torch
    from torch_geometric.data import Data
    from torch_geometric.utils import subgraph

    from cgnn.models.focal_sets import sample_nodes_per_class

    data = bundle.data
    num_id = int(cfg["out_channels"])
    labels = torch.argmax(data.y, dim=1)
    train_id = data.train_mask & (data.y.sum(dim=1) == 1)
    nodes = sample_nodes_per_class(
        labels, train_id, num_id, int(cfg.get("budget_samples_per_class", 200)), int(cfg["seed"])
    )
    edge_index, _ = subgraph(nodes, data.edge_index, relabel_nodes=True, num_nodes=data.num_nodes)
    print(f"Budgeted focal sets: {nodes.numel()} sampled nodes, {edge_index.size(1)} edges")

    warmup = train_warmup_backbone(cfg, bundle, output_dir)
    with torch.no_grad():
        embeddings = warmup.embedding(Data(x=data.x[nodes], edge_index=edge_index).to(warmup.device))
    return embeddings.cpu().numpy(), labels[nodes].numpy()


def budget_device(cfg: dict[str, Any]) -> str:
    import torch

    return "cuda" if torch.cuda.is_available() and cfg.get("accelerator", "auto") != "cpu" else "cpu"


def budgeted_focal_sets_for(ctx: RunContext) -> list[list[int]]:
    from cgnn.models.focal_sets import budgeted_focal_sets

    cfg = ctx.cfg
    embeddings, labels = budget_embeddings(cfg, ctx.bundle, ctx.paths.output_dir)
    return budgeted_focal_sets(
        embeddings,
        labels,
        int(cfg["out_channels"]),
        k=int(cfg.get("budget_k", 64)),
        max_cardinality=int(cfg.get("budget_max_cardinality", 3)),
        seed=int(cfg["seed"]),
        device=budget_device(cfg),
    )


def build_focal_sets(ctx: RunContext) -> tuple[list[list[int]], dict[str, Any]]:
    from cgnn.models.focal_sets import power_set, singletons

    cfg = ctx.cfg
    num_id = int(cfg["out_channels"])
    num_total = int(cfg.get("num_total_classes") or num_id)
    strategy = cfg.get("focal_set_strategy", "auto")
    if strategy not in ("auto", "budgeted", "singletons"):
        raise ValueError(f"focal_set_strategy must be auto, budgeted or singletons, got {strategy!r}")

    info: dict[str, Any] = {}
    budgeted = strategy == "budgeted" or (
        strategy == "auto" and num_id >= int(cfg.get("power_set_max_classes", 10))
    )
    if strategy == "singletons":
        focal_sets = singletons(num_id)
    elif budgeted:
        start = time.perf_counter()
        focal_sets = budgeted_focal_sets_for(ctx)
        info["focal_set_budget_seconds"] = time.perf_counter() - start
    else:
        focal_sets = power_set(num_id)
    focal_sets = focal_sets + singletons(num_total, start=num_id)
    info["num_focal_sets"] = len(focal_sets)
    print(f"Focal sets ({'budgeted' if budgeted else strategy}): {len(focal_sets)}")
    return focal_sets, info


def random_set_builder(**fixed: Any):
    """Builder of RS-GNN; ``fixed`` config values define an ablation and cannot be overridden."""

    def build(ctx: RunContext):
        from cgnn.models.random_set import RandomSetGNN

        ctx.cfg.update(fixed)
        cfg = ctx.cfg
        focal_sets, info = build_focal_sets(ctx)
        model = RandomSetGNN(
            **_backbone_kwargs(cfg),
            weight_decay=cfg.get("weight_decay", 0.0),
            focal_sets=focal_sets,
            num_classes=int(cfg.get("num_total_classes") or cfg["out_channels"]),
            alpha=cfg.get("alpha", 1e-3),
            beta=cfg.get("beta", 1e-3),
            loss_ablation=cfg.get("loss_ablation", "full"),
            isotonic_calibration=bool(cfg.get("isotonic_calibration", False)),
        )
        model.build_metrics = info
        return model

    return build


_RANDOM_SET_METHODS = [
    # name, paper name, fixed config, description
    (
        "random_set",
        "RS-GNN",
        {},
        "Belief head over a (budgeted) focal-set family; pignistic entropy and credal width.",
    ),
    (
        "random_set_singletons",
        "RS-GNN singletons only",
        {"focal_set_strategy": "singletons"},
        "Ablation: focal family restricted to singletons.",
    ),
    (
        "random_set_bce_only",
        "RS-GNN BCE only",
        {"loss_ablation": "bce_only"},
        "Ablation: BCE in belief space without mass-validity regularisers.",
    ),
    (
        "random_set_bce_mr",
        "RS-GNN BCE + margin",
        {"loss_ablation": "bce_mr"},
        "Ablation: BCE + negative-mass regulariser R_neg only.",
    ),
    (
        "random_set_isotonic",
        "RS-GNN + isotonic calibration",
        {"isotonic_calibration": True},
        "RS-GNN with post-hoc one-vs-rest isotonic calibration (ID validation nodes).",
    ),
]

for _name, _paper, _fixed, _desc in _RANDOM_SET_METHODS:
    register_method(
        MethodSpec(
            name=_name,
            kind="trainable",
            build=random_set_builder(**_fixed),
            paper_name=_paper,
            description=_desc,
            build_needs_data=True,
            requires=("sklearn",),
            score_keys=RANDOM_SET_SCORES,
        )
    )
