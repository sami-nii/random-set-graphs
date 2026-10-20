"""GNN backbones (PyG ``BasicGNN`` models) and layer-wise embeddings.

``act`` defaults to ``F.sigmoid`` (the CGNN protocol). The RS-GNN paper uses
``GATv2`` with ``act="relu"`` and ``dropout=0.2`` for every model
(configs/methods/*.yaml).
"""

from __future__ import annotations

from typing import Final

import torch
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv
from torch_geometric.nn.models import GAT, GCN, GIN, EdgeCNN, GraphSAGE
from torch_geometric.nn.models.basic_gnn import BasicGNN


class GATv2(BasicGNN):
    """GATv2 encoder of the RS-GNN paper (App. "Backbone and implementation details").

    Hidden layers concatenate ``heads`` attention heads of size
    ``hidden_channels // heads`` (so every layer emits ``hidden_channels``
    features); the last layer uses a single head. Unlike PyG's ``GAT(v2=True)``,
    whose last layer averages ``heads`` heads.
    """

    supports_edge_weight: Final[bool] = False
    supports_edge_attr: Final[bool] = True
    supports_padding_mask: Final[bool] = False

    def init_conv(self, in_channels, out_channels, **kwargs):
        heads = kwargs.pop("heads", 4)
        if getattr(self, "_is_conv_to_out", False) or heads == 1:
            return GATv2Conv(in_channels, out_channels, heads=1, concat=False, **kwargs)
        if out_channels % heads != 0:
            raise ValueError(f"hidden_channels={out_channels} must be divisible by heads={heads}")
        return GATv2Conv(in_channels, out_channels // heads, heads=heads, concat=True, **kwargs)


BACKBONES = {"GCN": GCN, "SAGE": GraphSAGE, "GAT": GAT, "GATv2": GATv2, "GIN": GIN, "EdgeCNN": EdgeCNN}


def build_backbone(
    gnn_type: str,
    in_channels: int,
    hidden_channels: int,
    num_layers: int,
    out_channels: int,
    act=F.sigmoid,
    **kwargs,
):
    if gnn_type not in BACKBONES:
        raise KeyError(f"Unknown gnn_type '{gnn_type}'. Known: {sorted(BACKBONES)}")
    if "gat_heads" in kwargs:  # attention heads of GAT/GATv2 (``heads`` is taken by CredalGNN)
        kwargs["heads"] = kwargs.pop("gat_heads")
    return BACKBONES[gnn_type](
        in_channels=in_channels,
        hidden_channels=hidden_channels,
        num_layers=num_layers,
        out_channels=out_channels,
        act=act,
        **kwargs,
    )


def check_layerwise_support(gnn, num_layers: int | None = None) -> None:
    if not hasattr(gnn, "convs") or not hasattr(gnn, "act"):
        raise NotImplementedError(
            f"{gnn.__class__.__name__} must expose 'convs' and 'act' for layer-wise embeddings."
        )
    if num_layers is not None and len(gnn.convs) < num_layers:
        raise ValueError(f"Expected at least {num_layers} conv layers, got {len(gnn.convs)}.")


def layerwise_embeddings(
    gnn,
    x: torch.Tensor,
    edge_index: torch.Tensor,
    *,
    num_layers: int | None = None,
    act_on_last: bool,
    include_input: bool,
) -> list[torch.Tensor]:
    """Per-layer node embeddings ``[z0?, z1, ..., zL]`` of a PyG ``BasicGNN``.

    Replicates ``BasicGNN.forward`` for the default configuration (no norm,
    no JK); dropout between layers follows the backbone's ``dropout``. ``act_on_last=False`` matches the standard forward
    (the last layer returns pre-activation outputs, i.e. logits);
    ``act_on_last=True`` is what the trainable joint-latent credal models use.
    ``include_input`` prepends the raw features ``z0 = x`` (paper Eq. 14).
    """
    num_layers = gnn.num_layers if num_layers is None else num_layers
    check_layerwise_support(gnn, num_layers)
    embeddings = [x] if include_input else []
    h = x
    for i in range(num_layers):
        h = gnn.convs[i](h, edge_index)
        if act_on_last or i < num_layers - 1:
            h = gnn.act(h)
        embeddings.append(h)
        if i < num_layers - 1:
            h = gnn.dropout(h)
    return embeddings


def joint_representation(gnn, x, edge_index, **kwargs) -> torch.Tensor:
    """Concatenation of :func:`layerwise_embeddings` (the "latent joint")."""
    return torch.cat(layerwise_embeddings(gnn, x, edge_index, **kwargs), dim=1)
