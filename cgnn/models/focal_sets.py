"""Focal-set families of RS-GNN (paper Sec. 3, App. "Focal-set construction", Alg. focal_sets).

- :func:`power_set`: all non-empty subsets of the ID classes (used when ``|Y| < 10``).
- :func:`budgeted_focal_sets`: singletons plus the ``K`` pairs/triples whose
  95% class ellipsoids overlap most in a 3-D t-SNE projection of a warm-up
  GNN's embeddings. Only labelled ID training nodes are used.
- :func:`focal_set_matrices`: Moebius, pignistic, membership and singleton
  matrices of a family.

Focal sets are lists of sorted class indices (JSON/YAML friendly, stored in
checkpoints).
"""

from __future__ import annotations

import inspect
import itertools
from collections.abc import Sequence

import numpy as np
import torch

FocalSets = list[list[int]]

# chi^2 quantile with 3 degrees of freedom at 95% (ellipsoid of a 3-D Gaussian)
CHI2_3_95 = 7.815


def power_set(num_classes: int) -> FocalSets:
    """All non-empty subsets of ``{0, ..., C-1}``, ordered by cardinality."""
    return [list(s) for r in range(1, num_classes + 1) for s in itertools.combinations(range(num_classes), r)]


def singletons(num_classes: int, start: int = 0) -> FocalSets:
    return [[c] for c in range(start, num_classes)]


def focal_set_matrices(focal_sets: Sequence[Sequence[int]], num_classes: int):
    """
    Returns ``(moebius, pignistic, membership, singleton)``:

    - ``moebius[i, j] = (-1)^{|A_i| - |A_j|}`` if ``A_j ⊆ A_i`` (masses ``m = Bel @ moebius.T``);
    - ``pignistic[i, c] = 1/|A_i|`` if ``c ∈ A_i`` (``BetP = m @ pignistic``);
    - ``membership[i, c] = 1`` if ``c ∈ A_i`` (``Pl({c}) = m @ membership``);
    - ``singleton[i, c] = 1`` if ``A_i = {c}`` (``Bel({c}) = m @ singleton``).
    """
    sets = [frozenset(s) for s in focal_sets]
    n = len(sets)
    moebius = torch.zeros((n, n))
    for i, a in enumerate(sets):
        for j, b in enumerate(sets):
            if b <= a:
                moebius[i, j] = (-1) ** (len(a) - len(b))
    pignistic = torch.zeros((n, num_classes))
    membership = torch.zeros((n, num_classes))
    singleton = torch.zeros((n, num_classes))
    for i, a in enumerate(sets):
        for c in a:
            pignistic[i, c] = 1.0 / len(a)
            membership[i, c] = 1.0
            if len(a) == 1:
                singleton[i, c] = 1.0
    return moebius, pignistic, membership, singleton


# ---------------------------------------------------------------------------
# Budgeted, graph-aware construction
# ---------------------------------------------------------------------------
def sample_nodes_per_class(
    labels: torch.Tensor, mask: torch.Tensor, num_classes: int, per_class: int, seed: int
) -> torch.Tensor:
    """Up to ``per_class`` node indices of each class among ``mask`` (seeded)."""
    generator = torch.Generator().manual_seed(int(seed))
    chosen, missing = [], []
    for c in range(num_classes):
        nodes = torch.nonzero(mask & (labels == c), as_tuple=False).view(-1)
        if nodes.numel() == 0:
            missing.append(c)
            continue
        if nodes.numel() > per_class:
            nodes = nodes[torch.randperm(nodes.numel(), generator=generator)[:per_class]]
        chosen.append(nodes)
    if missing:
        raise ValueError(f"Budgeted focal-set construction found no labelled ID training nodes for {missing}")
    return torch.cat(chosen)


def tsne_3d(embeddings: np.ndarray, seed: int | None) -> np.ndarray:
    from sklearn.manifold import TSNE

    kwargs = {"n_components": 3, "init": "random", "perplexity": 30, "random_state": seed}
    if "n_jobs" in inspect.signature(TSNE.__init__).parameters:
        kwargs["n_jobs"] = -1
    return TSNE(**kwargs).fit_transform(embeddings)


def class_regions(points: np.ndarray, labels: np.ndarray, num_classes: int, device="cpu"):
    """95% ellipsoid of a Gaussian fitted to each class, discretised on a shared 3-D grid."""
    from scipy import linalg
    from sklearn.mixture import GaussianMixture

    eigen_vecs, stds = [], []
    for c in range(num_classes):
        class_points = points[labels == c]
        gm = GaussianMixture(n_components=1, random_state=7).fit(class_points)
        v, w = linalg.eigh(gm.covariances_[0])
        stds.append(2.0 * np.sqrt(CHI2_3_95) * np.sqrt(v))
        eigen_vecs.append(w)
    eigen_vecs = torch.tensor(np.asarray(eigen_vecs), dtype=torch.float32, device=device)
    stds = torch.tensor(np.asarray(stds), dtype=torch.float32, device=device)

    # Every ellipsoid is centred on the grid centre (as in RS-NN): overlaps compare
    # class shapes and orientations in the embedding space.
    max_len = int(stds.max().item()) + 2
    shape = (max_len,) * 3
    grid = torch.stack(torch.meshgrid(*[torch.arange(max_len, device=device)] * 3, indexing="ij"), dim=-1)
    vecs = grid.reshape(-1, 3).float() - torch.tensor(shape, device=device, dtype=torch.float32) / 2.0
    norms = torch.norm(vecs, dim=-1, keepdim=True) + 1e-31

    regions = []
    for c in range(num_classes):
        proj = torch.abs(vecs @ eigen_vecs[c] / (norms * torch.norm(eigen_vecs[c], dim=-1)))
        radius = torch.norm(((proj * (stds[c] / 2)).unsqueeze(-1) * eigen_vecs[c]).sum(dim=1), dim=-1) + 1e-25
        regions.append((norms.squeeze(-1) <= radius).reshape(shape))
    return regions


def overlap_scores(regions, num_classes: int, max_cardinality: int = 3) -> dict[tuple[int, ...], float]:
    """Jaccard overlap of pairs, then of triples extending pairs with non-zero overlap, etc."""
    scores: dict[tuple[int, ...], float] = {}
    frontier = [(c,) for c in range(num_classes)]
    for cardinality in range(2, max(2, min(max_cardinality, num_classes)) + 1):
        next_frontier = []
        for base in frontier:
            for c in range(num_classes):
                candidate = tuple(sorted(set(base) | {c}))
                if len(candidate) != cardinality or candidate in scores:
                    continue
                stacked = torch.stack([regions[k] for k in candidate]).sum(dim=0)
                union = int((stacked > 0).sum())
                scores[candidate] = 0.0 if union == 0 else int((stacked == cardinality).sum()) / union
                if scores[candidate] > 0:
                    next_frontier.append(candidate)
        frontier = next_frontier
    return scores


def top_k_focal_sets(scores: dict[tuple[int, ...], float], num_classes: int, k: int) -> FocalSets:
    """Singletons plus the ``k`` highest-scoring candidates with non-zero overlap (stable ties)."""
    ranked = sorted(scores.items(), key=lambda item: -item[1])
    selected = [list(s) for s, v in ranked[:k] if v > 0]
    return singletons(num_classes) + selected


def budgeted_focal_sets(
    embeddings: np.ndarray,
    labels: np.ndarray,
    num_classes: int,
    *,
    k: int = 64,
    max_cardinality: int = 3,
    seed: int | None = 0,
    device="cpu",
) -> FocalSets:
    """Alg. focal_sets on precomputed warm-up embeddings of the sampled training nodes."""
    points = tsne_3d(embeddings, seed)
    regions = class_regions(points, labels, num_classes, device=device)
    return top_k_focal_sets(overlap_scores(regions, num_classes, max_cardinality), num_classes, k)
