"""Temporal road-scene graphs (paper App. "Temporal road-scene graph construction") on tiny fixtures
that follow the ROAD (road_trainval_v1.0.json) and nuScenes (v1.0-mini tables) schemas."""

from __future__ import annotations

import json
import random

import pytest
import torch

from cgnn.data import load_dataset
from cgnn.data.sources.road_scene import NUSCENES_MINI_TRAIN, NUSCENES_MINI_VAL
from cgnn.paths import Paths

AGENTS = ["Ped", "Car", "Cyc", "Mobike", "MedVeh", "LarVeh", "Bus", "EmVeh", "TL", "OthTL"]


def _write_road(root):
    rng = random.Random(0)
    road = {
        "all_agent_labels": ["Unused", *AGENTS],
        "agent_labels": AGENTS,
        "all_action_labels": [f"a{i}" for i in range(19)] + ["Unused"],
        "action_labels": [f"a{i}" for i in range(19)],
        "all_loc_labels": [f"l{i}" for i in range(12)],
        "loc_labels": [f"l{i}" for i in range(12)],
        "db": {},
    }
    for v, split in enumerate(["train_1"] * 3 + ["val_1"] * 2):
        frames = {}
        for f in range(1, 41):
            annos = {}
            for a in range(4):
                x1, y1 = rng.random() * 0.8, rng.random() * 0.8
                annos[f"b{f}_{a}"] = {
                    "box": [x1, y1, x1 + 0.1, y1 + 0.1],
                    "agent_ids": [1 + (a + f) % 10],
                    "action_ids": [rng.randrange(20)],
                    "loc_ids": [rng.randrange(12)],
                    "tube_uid": f"t{v}_{a}",
                }
            frames[str(f)] = {"annotated": 1, "annos": annos}
        road["db"][f"video{v}"] = {"split_ids": [split], "frames": frames}
    (root / "road").mkdir(parents=True)
    (root / "road" / "road_trainval_v1.0.json").write_text(json.dumps(road))


def _write_nuscenes(root):
    rng = random.Random(0)
    cats = [
        "vehicle.car",
        "vehicle.truck",
        "vehicle.bus.rigid",
        "vehicle.trailer",
        "vehicle.construction",
        "human.pedestrian.adult",
        "vehicle.motorcycle",
        "vehicle.bicycle",
        "movable_object.barrier",
    ]
    tables = {
        "category": [{"token": f"c{i}", "name": n} for i, n in enumerate(cats)],
        "instance": [],
        "sample": [],
        "scene": [],
        "sample_annotation": [],
    }
    for s, name in enumerate(NUSCENES_MINI_TRAIN + NUSCENES_MINI_VAL):
        insts = [f"i{s}_{k}" for k in range(9)]
        tables["instance"] += [{"token": t, "category_token": f"c{k}"} for k, t in enumerate(insts)]
        toks = [f"s{s}_{j}" for j in range(10)]
        for j, t in enumerate(toks):
            tables["sample"].append({"token": t, "next": toks[j + 1] if j + 1 < len(toks) else ""})
            for k, it in enumerate(insts):
                tables["sample_annotation"].append(
                    {
                        "token": f"a{s}_{j}_{k}",
                        "sample_token": t,
                        "instance_token": it,
                        "translation": [rng.random() * 50, rng.random() * 50, 1.0],
                        "size": [2.0, 4.0, 1.5],
                        "num_lidar_pts": rng.randint(0, 100),
                        "num_radar_pts": rng.randint(0, 5),
                    }
                )
        tables["scene"].append({"token": f"sc{s}", "name": name, "first_sample_token": toks[0]})
    d = root / "nuscenes" / "v1.0-mini"
    d.mkdir(parents=True)
    for n, t in tables.items():
        (d / f"{n}.json").write_text(json.dumps(t))


@pytest.fixture
def road_paths(tmp_path) -> Paths:
    _write_road(tmp_path)
    _write_nuscenes(tmp_path)
    return Paths(data_dir=tmp_path, ckpt_dir=tmp_path / "ckpt", output_dir=tmp_path / "out")


def _graph_checks(graph, num_features):
    assert graph.x.size(1) == num_features
    scene = graph.x[:, -2] == 1
    labelled = graph.train_mask | graph.val_mask | graph.test_mask
    assert not (scene & labelled).any()  # scene nodes are never labelled
    src, dst = graph.edge_index
    assert (src == dst).sum() == graph.num_nodes  # self-loops
    # every agent is connected to the scene node of its frame
    agents = torch.nonzero(~scene).view(-1)
    assert all(scene[src[dst == a]].any() for a in agents.tolist())


def test_road_windows(road_paths):
    bundle = load_dataset("road", {"batch_size": 4}, paths=road_paths)
    g = bundle.train_loader.dataset[0]
    _graph_checks(g, 37)
    assert int(g.x[:, -2].sum()) == 4  # window size 4 -> 4 scene nodes
    assert bundle.data.y.size(1) == 8  # ID agent classes {0..7}
    assert len(bundle.val_loader.dataset) + len(bundle.test_loader.dataset) > 0
    assert (bundle.data.train_mask & (bundle.data.y.sum(1) == 0)).sum() == 0  # no OOD in training


def test_nuscenes_windows_and_mini_split(road_paths):
    bundle = load_dataset("nuscenes", {}, paths=road_paths)
    g = bundle.train_loader.dataset[0]
    _graph_checks(g, 10)
    assert int(g.x[:, -2].sum()) == 3  # window size 3
    # 10 samples per scene, window 3, stride 1 -> 8 windows per scene; 8 train / 1 val / 1 test scenes
    assert [
        len(loader.dataset) for loader in (bundle.train_loader, bundle.val_loader, bundle.test_loader)
    ] == [64, 8, 8]
    s = bundle.summary()
    assert s["test_ood"] > 0 and s["test_id"] > 0  # motorcycle/bicycle are OOD


def test_road_to_nuscenes_test_split(road_paths):
    bundle = load_dataset("road_to_nuscenes", {}, paths=road_paths)
    s = bundle.summary()
    assert s["val_ood"] == 0  # all ROAD classes are ID
    assert s["test_ood"] > 0  # every labelled nuScenes agent is OOD
    assert {g.x.size(1) for g in bundle.test_loader.dataset} == {37}  # nuScenes zero-padded to ROAD width


@pytest.mark.slow  # ellipsoid voxel grid of the focal-set budget: ~2 min on CPU
def test_rs_gnn_road_protocol(road_paths):
    from cgnn.runner import run_experiment
    from tests.conftest import FAST

    results = run_experiment(
        "random_set",
        "road",
        {**FAST, "max_epochs": 1, "budget_warmup_epochs": 1, "budget_samples_per_class": 20},
        logger=False,
        run_id="r0",
        paths=road_paths,
        write_summary=False,
    )
    assert results["num_focal_sets"] <= 8 + 64 + 2  # budgeted ID pairs/triples + OOD singletons
    assert "test_auroc_entropy" in results and "test_auroc_credal" in results
