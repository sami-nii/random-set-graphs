"""Result aggregation (pure functions; no W&B access)."""

from __future__ import annotations

import pytest

from cgnn.results import RunRow, format_table, paper_table, split_sweep_name, summarize


def _row(
    i, method="credal_LJ_dual_head_detached", ds="squirrel", val=None, test=None, cfg=None, state="finished"
):
    return RunRow(f"r{i}", ds, method, state, cfg or {"lr": 0.1, "seed": i}, {**(val or {}), **(test or {})})


def test_split_sweep_name_handles_underscores():
    known = ["roman_empire", "amazon_ratings", "squirrel"]
    assert split_sweep_name("roman_empire_credal_LJ_dual_head", known) == (
        "roman_empire",
        "credal_LJ_dual_head",
    )
    assert split_sweep_name("amazon_ratings_knn", known) == ("amazon_ratings", "knn")
    assert split_sweep_name("unknown_x", known) is None


def test_topk_selects_by_validation_and_reports_test():
    rows = [
        _row(i, val={"val_auroc_EU": v}, test={"test_auroc_EU": t})
        for i, (v, t) in enumerate([(0.9, 0.7), (0.8, 0.9), (0.7, 0.1), (0.95, 0.5)])
    ]
    s = summarize(rows, "credal_LJ_dual_head_detached", "squirrel", ("EU",), aggregate="topk", k=2)
    assert s.run_ids == ["r3", "r0"] and s.n == 2
    assert s.test_mean == pytest.approx(0.6) and s.val_mean == pytest.approx(0.925)


def test_component_is_selected_on_validation():
    rows = [
        _row(
            i,
            val={"val_auroc_EU": 0.6, "val_auroc_AU": 0.8},
            test={"test_auroc_EU": 0.9, "test_auroc_AU": 0.5},
        )
        for i in range(3)
    ]
    s = summarize(rows, "credal_LJ_dual_head_detached", "squirrel", ("EU", "AU"))
    assert s.component == "AU" and s.test_mean == pytest.approx(0.5)  # chosen on val, even if test is worse


def test_group_aggregation_uses_best_config_group():
    rows = [
        _row(i, cfg={"lr": 0.1, "seed": i}, val={"val_auroc": 0.7}, test={"test_auroc": 0.6})
        for i in range(3)
    ]
    rows += [
        _row(10 + i, cfg={"lr": 0.2, "seed": i}, val={"val_auroc": 0.8}, test={"test_auroc": 0.4})
        for i in range(3)
    ]
    s = summarize(rows, "credal_LJ_dual_head_detached", "squirrel", ("",), aggregate="group")
    assert s.n == 3 and s.test_mean == pytest.approx(0.4)


def test_unfinished_and_incomplete_runs_are_ignored():
    rows = [
        _row(0, val={"val_auroc": 0.9}, test={"test_auroc": 0.9}, state="crashed"),
        _row(1, val={"val_auroc": 0.9}),
    ]
    assert summarize(rows, "credal_LJ_dual_head_detached", "squirrel") is None


def test_table_formats():
    rows = [
        _row(i, method="energy", val={"val_auroc": 0.5 + i / 10}, test={"test_auroc": 0.6}) for i in range(3)
    ]
    table = paper_table(rows, ["squirrel", "arxiv"], [("Energy", "energy", ("",))])
    md = format_table(table, "md")
    assert "| Energy | **60.00 ± 0.00** | - |" in md
    assert r"\textbf{60.00 $\pm$ 0.00}" in format_table(table, "latex")
    assert format_table(table, "csv").splitlines()[1].startswith("Energy,0.600000,0.000000,,3")


def test_rsgnn_table_cells_mean_and_rank():
    """RS-GNN paper Tab. 1 layout: "first/second" cells (each component selected on its own validation
    AUROC), Mean and Rank on the first value."""
    from cgnn.results import format_rsgnn_table, rsgnn_table, table_mean_rank

    rows = [
        _row(
            0,
            "random_set",
            "cora",
            {"val_auroc_entropy": 0.9, "val_auroc_credal": 0.1},
            {"test_auroc_entropy": 0.88, "test_auroc_credal": 0.50},
        ),
        _row(
            1,
            "random_set",
            "cora",
            {"val_auroc_entropy": 0.1, "val_auroc_credal": 0.9},
            {"test_auroc_entropy": 0.40, "test_auroc_credal": 0.85},
        ),
        _row(2, "energy", "cora", {"val_auroc": 0.7}, {"test_auroc": 0.60}),
        _row(3, "energy", "chameleon", {"val_auroc": 0.7}, {"test_auroc": 0.70}),
    ]
    spec = [("Energy", "energy", ("",)), ("RS-GNN (ours)", "random_set", ("entropy", "credal"))]
    table = rsgnn_table(rows, ["chameleon", "cora"], spec, k=1)
    entropy, credal = table["RS-GNN (ours)"]["cora"]
    assert entropy.run_ids == ["r0"] and credal.run_ids == ["r1"]  # independent selection per component
    mean_rank = table_mean_rank(table)
    assert mean_rank["Energy"] == (pytest.approx(0.65), pytest.approx(1.5))  # rank 1 on chameleon, 2 on cora
    assert mean_rank["RS-GNN (ours)"] == (pytest.approx(0.88), pytest.approx(1.0))
    text = format_rsgnn_table(table)
    assert "| RS-GNN (ours) | - | **88.00/85.00** | 88.00 | 1.00 |" in text
