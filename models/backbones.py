"""Shared message-passing backbone used by every model in the paper.

Paper (App. "Backbone and implementation details"): a two-layer GATv2 encoder
with hidden size 128; the first layer uses 4 attention heads, the second a
single head returning a 128-dimensional node embedding; ReLU after each layer
and dropout 0.2 after the first layer. Prediction heads (softmax, belief,
credal) sit on top of the embedding ``z_v``.

Multi-head layers concatenate ``heads`` outputs of size ``hidden // heads`` so
that every layer emits ``hidden_channels`` features.
"""

import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GATv2Conv, GCNConv, GINConv, SAGEConv


def _make_conv(gnn_type, in_dim, hidden_channels, heads, last):
    if gnn_type in ("GATv2", "GAT"):
        conv_cls = GATv2Conv if gnn_type == "GATv2" else GATConv
        if last or heads == 1:
            return conv_cls(in_dim, hidden_channels, heads=1)
        if hidden_channels % heads != 0:
            raise ValueError(f"hidden_channels={hidden_channels} must be divisible by heads={heads}.")
        return conv_cls(in_dim, hidden_channels // heads, heads=heads, concat=True)
    if gnn_type == "GCN":
        return GCNConv(in_dim, hidden_channels)
    if gnn_type == "SAGE":
        return SAGEConv(in_dim, hidden_channels)
    if gnn_type == "GIN":
        mlp = nn.Sequential(nn.Linear(in_dim, hidden_channels), nn.ReLU(), nn.Linear(hidden_channels, hidden_channels))
        return GINConv(mlp)
    raise ValueError(f"Unsupported gnn_type: {gnn_type}")


class MessagePassingBackbone(nn.Module):
    """``num_layers`` message-passing layers, each followed by ``act``; dropout between layers."""

    def __init__(
        self,
        gnn_type: str,
        in_channels: int,
        hidden_channels: int = 128,
        num_layers: int = 2,
        heads: int = 4,
        dropout: float = 0.2,
        act=F.relu,
    ):
        super().__init__()
        self.gnn_type = gnn_type
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = hidden_channels
        self.num_layers = num_layers
        self.dropout = dropout
        self.act = act
        self.convs = nn.ModuleList(
            _make_conv(
                gnn_type,
                in_channels if i == 0 else hidden_channels,
                hidden_channels,
                heads,
                last=(i == num_layers - 1),
            )
            for i in range(num_layers)
        )

    def layer_outputs(self, x, edge_index):
        """Post-activation output of every layer, ``[H^(1), ..., H^(L)]``."""
        outputs = []
        for i, conv in enumerate(self.convs):
            x = self.act(conv(x, edge_index))
            outputs.append(x)
            if i < self.num_layers - 1:
                x = F.dropout(x, p=self.dropout, training=self.training)
        return outputs

    def forward(self, x, edge_index):
        return self.layer_outputs(x, edge_index)[-1]


def build_backbone(gnn_type, in_channels, hidden_channels=128, num_layers=2, heads=4, dropout=0.2, act=F.relu):
    return MessagePassingBackbone(
        gnn_type=gnn_type,
        in_channels=in_channels,
        hidden_channels=hidden_channels,
        num_layers=num_layers,
        heads=heads,
        dropout=dropout,
        act=act,
    )
