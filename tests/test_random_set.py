"""RS-GNN: belief head, Proposition 1, focal sets, backbone (paper Sec. 3 and App. D)."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from torch_geometric.data import Data

from cgnn.models.focal_sets import (
    focal_set_matrices,
    overlap_scores,
    power_set,
    top_k_focal_sets,
)
from cgnn.models.random_set import RandomSetGNN, RandomSetLoss


def _model(focal_sets, num_classes, **kwargs):
    return RandomSetGNN("GATv2", 4, 8, 2, num_classes, focal_sets, num_classes=num_classes, **kwargs).eval()


def _beliefs_from_masses(focal_sets, masses):
    """Bel(A) = sum_{B in F, B subset of A} m(B)."""
    sets = [frozenset(s) for s in focal_sets]
    return torch.tensor([[sum(masses.get(b, 0.0) for b in sets if b <= a) for a in sets]])


def test_worked_example_of_the_paper():
    """Sec. 3 'Example': BetP = (0.3, 0.5, 0.2), prediction c2, W = Pl({c2}) - Bel({c2}) = 0.5."""
    F = power_set(3)
    m = {
        frozenset({0}): 0.1,
        frozenset({1}): 0.3,
        frozenset({2}): 0.1,
        frozenset({0, 1}): 0.2,
        frozenset({0, 1, 2}): 0.3,
    }
    betp, entropy, width = _model(F, 3).predict(_beliefs_from_masses(F, m))
    assert torch.allclose(betp, torch.tensor([[0.3, 0.5, 0.2]]), atol=1e-6)
    assert betp.argmax().item() == 1
    assert width.item() == pytest.approx(0.5, abs=1e-6)
    assert entropy.item() == pytest.approx(
        -(0.3 * np.log(0.3) + 0.5 * np.log(0.5) + 0.2 * np.log(0.2)), abs=1e-5
    )


def test_moebius_inverts_beliefs_on_the_power_set():
    F = power_set(4)
    m = torch.distributions.Dirichlet(torch.ones(len(F))).sample((5,))
    moebius, _, _, _ = focal_set_matrices(F, 4)
    beliefs = torch.cat(
        [_beliefs_from_masses(F, dict(zip(map(frozenset, F), row.tolist(), strict=True))) for row in m]
    )
    assert torch.allclose(beliefs @ moebius.t(), m, atol=1e-5)


@pytest.mark.parametrize("focal_sets", [power_set(3), [[0], [1], [2], [0, 1], [1, 2]]])
def test_proposition_validity(focal_sets):
    """(ii) BetP in the credal set, (iii) W = sum of non-singleton masses containing y_hat, in [0, 1]."""
    torch.manual_seed(0)
    model = _model(focal_sets, 3)
    beliefs = torch.rand(64, len(focal_sets))
    masses, degenerate = model.valid_masses(beliefs)
    betp, _, width = model.predict(beliefs)
    valid = ~degenerate
    assert torch.allclose(masses[valid].sum(1), torch.ones(int(valid.sum())), atol=1e-5)
    assert torch.all(masses >= 0)
    sets = [set(s) for s in focal_sets]
    for subset in power_set(3):  # BetP(B) >= Bel(B) for every B
        bel = sum(masses[:, i] for i, s in enumerate(sets) if s <= set(subset))
        assert torch.all(betp[valid][:, subset].sum(1) >= bel[valid] - 1e-6)
    pred = betp.argmax(1)
    expected = torch.stack(
        [sum(masses[n, i] for i, s in enumerate(sets) if len(s) > 1 and int(pred[n]) in s) for n in range(64)]
    )
    assert torch.allclose(width, expected, atol=1e-6) and torch.all((width >= 0) & (width <= 1))


def test_singletons_give_zero_width():
    """(iv) a mass function on singletons has zero credal width: the softmax-like special case."""
    _, _, width = _model([[0], [1], [2]], 3).predict(torch.rand(16, 3))
    assert torch.all(width == 0)


def test_belief_targets_are_set_membership():
    F = [[0], [1], [2], [0, 1], [1, 2]]
    moebius, _, membership, _ = focal_set_matrices(F, 3)
    loss = RandomSetLoss(membership, moebius)
    assert loss.membership[:, torch.tensor([1])].t().tolist() == [[0, 1, 0, 1, 1]]


def test_universal_set_absorbs_residual_mass():
    F = power_set(2)  # {0}, {1}, {0, 1}
    masses, _ = _model(F, 2).valid_masses(
        torch.tensor([[0.2, 0.3, 0.6]])
    )  # m = (0.2, 0.3, 0.1) + residual 0.4
    assert torch.allclose(masses, torch.tensor([[0.2, 0.3, 0.5]]), atol=1e-6)


def test_power_set_and_budget_sizes():
    assert len(power_set(3)) == 7
    scores = {(0, 1): 0.5, (1, 2): 0.2, (0, 2): 0.0, (0, 1, 2): 0.1}
    fs = top_k_focal_sets(scores, 3, k=2)
    assert fs == [[0], [1], [2], [0, 1], [1, 2]]
    assert len(top_k_focal_sets(scores, 3, k=64)) == 3 + 3  # only non-zero overlaps are kept


def test_overlap_scores_are_jaccard_of_regions():
    a = torch.zeros(4, 4, 4, dtype=torch.bool)
    b = a.clone()
    c = a.clone()
    a[:2], b[1:3], c[3:] = True, True, True
    scores = overlap_scores([a, b, c], 3, max_cardinality=3)
    assert scores[(0, 1)] == pytest.approx(1 / 3)
    assert scores[(0, 2)] == 0.0
    assert (0, 1, 2) in scores  # triples extend pairs with non-zero overlap


def test_gatv2_backbone_matches_the_paper():
    from cgnn.models.vanilla import VanillaGNN

    model = VanillaGNN("GATv2", 16, 128, 2, 3, head="linear", act="relu", dropout=0.2, gat_heads=4)
    first, second = model.gnn_model.convs
    assert (first.heads, first.concat, first.out_channels) == (4, True, 32)  # 4 heads x 32 = 128
    assert (second.heads, second.out_channels) == (1, 128)
    data = Data(x=torch.randn(10, 16), edge_index=torch.randint(0, 10, (2, 30)))
    model.eval()
    assert model(data).shape == (10, 3)
    assert model.embedding(data).shape == (10, 128)
    assert model.joint_latent(data).shape == (10, model.joint_latent_dim) == (10, 16 + 2 * 128 + 3)


def test_public_api_end_to_end():
    """cgnn.rsgnn: prepare -> focal sets -> fit -> predict on a small graph (README / docs/USING_RSGNN.md)."""
    from cgnn import rsgnn
    from cgnn.data.sources.synthetic import make_sbm_graph
    from cgnn.data.splits import random_split_masks

    g = make_sbm_graph(num_nodes=120, seed=0)
    train, val, test = random_split_masks(g.num_nodes, 0.6, 0.2, split_seed=0)
    data = rsgnn.prepare_graph(
        g.x, g.edge_index, g.y, train, val, test, id_classes=[2, 3, 4], ood_classes=[0, 1]
    )
    families = rsgnn.focal_sets(data)
    assert families == power_set(3)
    model = rsgnn.fit(data, families=families, max_epochs=2, accelerator="cpu", progress_bar=False)
    out = rsgnn.predict(model, data)
    assert out["probs"].shape == (120, 3)
    assert torch.allclose(out["probs"].sum(1), torch.ones(120), atol=1e-5)
    assert out["entropy"].shape == out["credal_width"].shape == out["prediction"].shape == (120,)
