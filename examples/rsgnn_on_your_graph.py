"""Use RS-GNN on your own graph: leave-out-class OOD detection on any PyG-style graph.

Runs on CPU in a few seconds on a synthetic graph; replace `load_my_graph` with your data.

    python examples/rsgnn_on_your_graph.py
"""

from __future__ import annotations

import torch
from sklearn.metrics import roc_auc_score

from cgnn import rsgnn
from cgnn.data.sources.synthetic import make_sbm_graph
from cgnn.data.splits import random_split_masks


def load_my_graph():
    """Return node features [N, F], edge_index [2, E], integer labels [N] and train/val/test masks [N]."""
    g = make_sbm_graph(num_nodes=600, num_classes=5, num_features=16, seed=0)
    train, val, test = random_split_masks(g.num_nodes, 0.6, 0.2, split_seed=0)
    return g.x, g.edge_index, g.y, train, val, test


def main() -> None:
    x, edge_index, y, train, val, test = load_my_graph()

    # 1) Leave-out-class protocol: classes 0 and 1 are never seen in training (OOD).
    data = rsgnn.prepare_graph(x, edge_index, y, train, val, test, id_classes=[2, 3, 4], ood_classes=[0, 1])

    # 2) Focal sets: full power set for < 10 ID classes, the graph-aware budget otherwise.
    families = rsgnn.focal_sets(data)
    print(f"{len(families)} focal sets: {families}")

    # 3) Train (paper defaults: GATv2 backbone, belief-space loss). Any config key can be overridden.
    model = rsgnn.fit(data, families=families, max_epochs=50, accelerator="cpu", progress_bar=False)

    # 4) Per-node outputs: pignistic probabilities, prediction, total and epistemic uncertainty.
    out = rsgnn.predict(model, data)
    is_ood = data.y.sum(dim=1) == 0
    is_id = ~is_ood
    mask = data.test_mask
    accuracy = (out["prediction"][mask & is_id] == data.y[mask & is_id].argmax(dim=1)).float().mean()
    print(f"test ID accuracy: {accuracy:.3f}")
    for score in ("entropy", "credal_width"):
        auroc = roc_auc_score(is_ood[mask].numpy(), out[score][mask].numpy())
        print(f"test OOD AUROC ({score}): {auroc:.3f}")

    # Nodes the model is least sure about (largest credal width): candidates for review.
    top = torch.topk(out["credal_width"], k=5).indices
    print("most uncertain nodes:", top.tolist())


if __name__ == "__main__":
    main()
