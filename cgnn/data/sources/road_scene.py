"""Temporal road-scene graph benchmarks: ROAD, nuScenes, ROAD (ID) -> nuScenes (OOD).

RS-GNN paper Sec. 4.1 and App. "Temporal road-scene graph construction". Each
temporal window of a video/scene becomes one graph:

- one unlabeled scene node per frame and one node per annotated agent;
- intra-frame edges connect the scene node bidirectionally to every agent;
- temporal edges connect the scene nodes of adjacent frames and the agents with
  the same track id (ROAD ``tube_uid``, nuScenes ``instance_token``) in adjacent
  frames; self-loops are added.

Node features are ``[agent features, is_scene, is_agent]``:

- ROAD (37): box centre/size (4), action multi-hot (19), location multi-hot (12), type (2);
  agent-ID features are excluded (label leakage);
- nuScenes (10): position relative to the frame centroid (3), size (3),
  log(1 + lidar/radar points) (2), type (2), standardised on the training scenes.

Splits: ROAD windows of the ``road_train_split`` videos are training graphs and
those of the ``road_eval_split`` videos are halved (seeded) into validation and
test graphs; nuScenes uses the official v1.0-mini scenes (8 train, scene-0103
validation, scene-0916 test). Processed graphs are cached next to the raw data.

Labels: one-hot over the ID classes, all-zero rows for OOD agents; scene nodes
and agents without a category are excluded from every mask.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch_geometric.data import Data

from cgnn.data.spec import DatasetSpec, register_dataset

NUSCENES_CATEGORIES = [
    "car",
    "truck",
    "bus",
    "trailer",
    "construction_vehicle",
    "pedestrian",
    "motorcycle",
    "bicycle",
]
NUSCENES_CATEGORY_PREFIXES = {
    "vehicle.car": "car",
    "vehicle.truck": "truck",
    "vehicle.bus": "bus",
    "vehicle.trailer": "trailer",
    "vehicle.construction": "construction_vehicle",
    "human.pedestrian": "pedestrian",
    "vehicle.motorcycle": "motorcycle",
    "vehicle.bicycle": "bicycle",
}
# Official v1.0-mini split of the nuScenes devkit (nuscenes/utils/splits.py).
NUSCENES_MINI_TRAIN = [
    "scene-0061",
    "scene-0553",
    "scene-0655",
    "scene-0757",
    "scene-0796",
    "scene-1077",
    "scene-1094",
    "scene-1100",
]
NUSCENES_MINI_VAL = ["scene-0103", "scene-0916"]


# ---------------------------------------------------------------------------
# Window graph construction (shared by ROAD and nuScenes)
# ---------------------------------------------------------------------------
def _windows(num_frames, window_size, window_stride, frame_step):
    """Frame indices of each window: [t, t + step, ..., t + (W - 1) * step], t += stride."""
    span = (window_size - 1) * frame_step
    return [
        [start + k * frame_step for k in range(window_size)]
        for start in range(0, num_frames - span, window_stride)
    ]


def _window_graph(frames, agent_feature_dim, scene_feature=None):
    """
    frames: list (one per frame) of lists of agents (track_id, features, label);
    label is an int class index or -1 for unlabeled agents.
    Node features are [agent features, is_scene, is_agent].
    """
    features, labels, edges = [], [], []
    previous_scene, previous_tracks = None, {}
    for agents in frames:
        scene_index = len(features)
        scene_x = np.zeros(agent_feature_dim, dtype=np.float32) if scene_feature is None else scene_feature
        features.append(np.concatenate([scene_x, [1.0, 0.0]]))
        labels.append(-1)
        if previous_scene is not None:
            edges += [(previous_scene, scene_index), (scene_index, previous_scene)]
        tracks = {}
        for track_id, agent_x, label in agents:
            agent_index = len(features)
            features.append(np.concatenate([agent_x, [0.0, 1.0]]))
            labels.append(label)
            edges += [(scene_index, agent_index), (agent_index, scene_index)]
            if track_id in previous_tracks:
                edges += [(previous_tracks[track_id], agent_index), (agent_index, previous_tracks[track_id])]
            tracks[track_id] = agent_index
        previous_scene, previous_tracks = scene_index, tracks

    num_nodes = len(features)
    edges += [(i, i) for i in range(num_nodes)]
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    return (
        torch.tensor(np.asarray(features), dtype=torch.float),
        edge_index,
        torch.tensor(labels, dtype=torch.long),
    )


def _to_data(x, edge_index, labels, split, id_classes):
    """Encode labels as one-hot over ID classes and attach split masks."""
    id_map = {c: i for i, c in enumerate(id_classes)}
    labelled = labels >= 0
    is_id = torch.tensor([int(label) in id_map for label in labels.tolist()], dtype=torch.bool)
    y = torch.zeros((labels.numel(), len(id_classes)), dtype=torch.float)
    for node in torch.nonzero(is_id).view(-1).tolist():
        y[node, id_map[int(labels[node])]] = 1.0
    data = Data(x=x, edge_index=edge_index, y=y)
    data.train_mask = (labelled & is_id) if split == "train" else torch.zeros_like(labelled)
    data.val_mask = labelled if split == "val" else torch.zeros_like(labelled)
    data.test_mask = labelled if split == "test" else torch.zeros_like(labelled)
    data.raw_label = labels
    return data


def _cached(cache_dir, name, params, build):
    os.makedirs(cache_dir, exist_ok=True)
    key = hashlib.md5(json.dumps(params, sort_keys=True).encode()).hexdigest()[:10]
    path = os.path.join(cache_dir, f"{name}_{key}.pt")
    if os.path.isfile(path):
        print(f"Loading cached graphs from {path}")
        return torch.load(path, weights_only=False)
    graphs = build()
    torch.save(graphs, path)
    print(f"Cached graphs to {path}")
    return graphs


# ---------------------------------------------------------------------------
# ROAD
# ---------------------------------------------------------------------------
def _road_annotation_path(dataset_root, config):
    candidates = [
        config.get("road_root"),
        os.path.join(dataset_root, "road", "road_trainval_v1.0.json"),
        os.path.join(dataset_root, "road-dataset-master", "road", "road_trainval_v1.0.json"),
    ]
    for candidate in candidates:
        if candidate and os.path.isfile(candidate):
            return candidate
    raise FileNotFoundError(
        "Could not find ROAD annotations. Set road_root to road_trainval_v1.0.json, "
        "or place it under dataset/road/."
    )


def _label_mapper(annotations, label_type):
    """Map indices of ``all_<type>_labels`` to indices of the used ``<type>_labels`` (ROAD devkit)."""
    all_labels = annotations[f"all_{label_type}_labels"]
    used = {name: i for i, name in enumerate(annotations[f"{label_type}_labels"])}
    return {i: used[name] for i, name in enumerate(all_labels) if name in used}, len(used)


def _road_agent(annotation, agent_map, action_map, loc_map, num_actions, num_locs):
    x1, y1, x2, y2 = annotation["box"]
    actions = np.zeros(num_actions, dtype=np.float32)
    actions[[action_map[a] for a in annotation.get("action_ids", []) if a in action_map]] = 1.0
    locs = np.zeros(num_locs, dtype=np.float32)
    locs[[loc_map[loc] for loc in annotation.get("loc_ids", []) if loc in loc_map]] = 1.0
    geometry = np.asarray([(x1 + x2) / 2.0, (y1 + y2) / 2.0, x2 - x1, y2 - y1], dtype=np.float32)
    agents = [agent_map[a] for a in annotation.get("agent_ids", []) if a in agent_map]
    label = agents[0] if agents else -1
    # Agent-ID features are excluded to prevent label leakage.
    return annotation["tube_uid"], np.concatenate([geometry, actions, locs]), label


def build_road_graphs(annotation_path, config):
    window_size = int(config.get("road_window_size", 4))
    window_stride = int(config.get("road_window_stride", 4))
    frame_step = int(config.get("road_frame_step", 5))
    train_split = config.get("road_train_split", "train_1")
    eval_split = config.get("road_eval_split", "val_1")
    split_seed = int(config.get("road_split_seed", 0))

    with open(annotation_path, encoding="utf-8") as handle:
        annotations = json.load(handle)
    agent_map, _ = _label_mapper(annotations, "agent")
    action_map, num_actions = _label_mapper(annotations, "action")
    loc_map, num_locs = _label_mapper(annotations, "loc")
    feature_dim = 4 + num_actions + num_locs

    graphs = {"train": [], "eval": []}
    for video in annotations["db"].values():
        split_ids = video.get("split_ids", [])
        target = "train" if train_split in split_ids else "eval" if eval_split in split_ids else None
        if target is None:
            continue
        frame_ids = sorted(int(f) for f, frame in video["frames"].items() if frame.get("annotated"))
        frames = [
            [
                _road_agent(a, agent_map, action_map, loc_map, num_actions, num_locs)
                for a in video["frames"][str(f)].get("annos", {}).values()
            ]
            for f in frame_ids
        ]
        for window in _windows(len(frames), window_size, window_stride, frame_step):
            graphs[target].append(_window_graph([frames[i] for i in window], feature_dim))

    # Held-out evaluation windows are split in half into validation and test graphs.
    order = np.random.default_rng(split_seed).permutation(len(graphs["eval"]))
    half = (len(order) + 1) // 2
    return {
        "train": graphs["train"],
        "val": [graphs["eval"][i] for i in order[:half]],
        "test": [graphs["eval"][i] for i in order[half:]],
    }


def _road_graphs(dataset_root, config):
    annotation_path = _road_annotation_path(dataset_root, config)
    params = {
        key: config.get(key, default)
        for key, default in [
            ("road_window_size", 4),
            ("road_window_stride", 4),
            ("road_frame_step", 5),
            ("road_train_split", "train_1"),
            ("road_eval_split", "val_1"),
            ("road_split_seed", 0),
        ]
    }
    return _cached(
        os.path.join(os.path.dirname(annotation_path), "processed"),
        "road_scene_graphs",
        params,
        lambda: build_road_graphs(annotation_path, config),
    )


# ---------------------------------------------------------------------------
# nuScenes
# ---------------------------------------------------------------------------
def _nuscenes_metadata_root(dataset_root, config):
    root = (
        config.get("nuscenes_root")
        or os.environ.get("NUSCENES_ROOT")
        or os.path.join(dataset_root, "nuscenes")
    )
    version = config.get("nuscenes_version", "v1.0-mini")
    metadata_root = os.path.join(root, version)
    if not os.path.isfile(os.path.join(metadata_root, "sample_annotation.json")):
        raise FileNotFoundError(
            f"Could not find nuScenes {version} metadata under {metadata_root}. "
            "Set nuscenes_root to the directory containing the version directory."
        )
    return metadata_root


def _nuscenes_category(name):
    for prefix, category in NUSCENES_CATEGORY_PREFIXES.items():
        if name.startswith(prefix):
            return NUSCENES_CATEGORIES.index(category)
    return None


def build_nuscenes_graphs(metadata_root, config):
    window_size = int(config.get("nuscenes_window_size", 3))
    window_stride = int(config.get("nuscenes_window_stride", 1))
    train_scenes = set(config.get("nuscenes_train_scenes", NUSCENES_MINI_TRAIN))
    val_scenes = set(config.get("nuscenes_val_scenes", NUSCENES_MINI_VAL[:1]))
    test_scenes = set(config.get("nuscenes_test_scenes", NUSCENES_MINI_VAL[1:]))

    def table(name):
        with open(os.path.join(metadata_root, f"{name}.json"), encoding="utf-8") as handle:
            return json.load(handle)

    category_name = {row["token"]: row["name"] for row in table("category")}
    instance_category = {
        row["token"]: _nuscenes_category(category_name[row["category_token"]]) for row in table("instance")
    }
    samples = {row["token"]: row for row in table("sample")}
    annotations_by_sample = defaultdict(list)
    for row in table("sample_annotation"):
        category = instance_category[row["instance_token"]]
        if category is not None:
            annotations_by_sample[row["sample_token"]].append((row, category))

    raw_frames = {}
    for scene in table("scene"):
        token, frames = scene["first_sample_token"], []
        while token:
            agents = []
            rows = annotations_by_sample.get(token, [])
            centre = np.mean([r["translation"] for r, _ in rows], axis=0) if rows else np.zeros(3)
            for row, category in rows:
                features = np.concatenate(
                    [
                        np.asarray(row["translation"]) - centre,  # position relative to the frame centroid
                        row["size"],
                        [np.log1p(row["num_lidar_pts"]), np.log1p(row["num_radar_pts"])],
                    ]
                ).astype(np.float32)
                agents.append((row["instance_token"], features, category))
            frames.append(agents)
            token = samples[token]["next"]
        raw_frames[scene["name"]] = frames

    # Standardise the physical features with statistics of the training scenes.
    train_features = np.asarray([a[1] for s in train_scenes for f in raw_frames.get(s, []) for a in f])
    mean, std = train_features.mean(axis=0), np.clip(train_features.std(axis=0), 1e-6, None)

    graphs = {"train": [], "val": [], "test": []}
    for scene_name, frames in raw_frames.items():
        split = (
            "train"
            if scene_name in train_scenes
            else "val"
            if scene_name in val_scenes
            else "test"
            if scene_name in test_scenes
            else None
        )
        if split is None:
            continue
        frames = [[(t, (x - mean) / std, c) for t, x, c in f] for f in frames]
        for window in _windows(len(frames), window_size, window_stride, 1):
            graphs[split].append(_window_graph([frames[i] for i in window], len(mean)))
    return graphs


def _nuscenes_graphs(dataset_root, config):
    metadata_root = _nuscenes_metadata_root(dataset_root, config)
    params = {
        "version": config.get("nuscenes_version", "v1.0-mini"),
        "window_size": config.get("nuscenes_window_size", 3),
        "window_stride": config.get("nuscenes_window_stride", 1),
        "train": sorted(config.get("nuscenes_train_scenes", NUSCENES_MINI_TRAIN)),
        "val": sorted(config.get("nuscenes_val_scenes", NUSCENES_MINI_VAL[:1])),
        "test": sorted(config.get("nuscenes_test_scenes", NUSCENES_MINI_VAL[1:])),
    }
    return _cached(
        os.path.join(metadata_root, "processed"),
        "nuscenes_scene_graphs",
        params,
        lambda: build_nuscenes_graphs(metadata_root, config),
    )


# ---------------------------------------------------------------------------
# Cross-dataset: ROAD (ID) vs nuScenes (OOD)
# ---------------------------------------------------------------------------
def _pad_features(graph, width):
    x, edge_index, labels = graph
    padded = torch.zeros((x.size(0), width), dtype=x.dtype)
    # Keep the node-type indicators in the last two columns, as in ROAD graphs.
    padded[:, : x.size(1) - 2] = x[:, :-2]
    padded[:, -2:] = x[:, -2:]
    return padded, edge_index, labels


# ---------------------------------------------------------------------------
# Dataset specs
# ---------------------------------------------------------------------------
def _encode(raw, id_classes):
    return {split: [_to_data(*g, split, list(id_classes)) for g in raw[split]] for split in raw}


def _load_road(data_dir: Path, spec: DatasetSpec, cfg) -> dict[str, list[Data]]:
    return _encode(_road_graphs(str(data_dir), cfg), spec.id_classes)


def _load_nuscenes(data_dir: Path, spec: DatasetSpec, cfg) -> dict[str, list[Data]]:
    return _encode(_nuscenes_graphs(str(data_dir), cfg), spec.id_classes)


def _load_road_to_nuscenes(data_dir: Path, spec: DatasetSpec, cfg) -> dict[str, list[Data]]:
    """Train/validate on ROAD (all ROAD classes ID); the test split adds nuScenes test graphs whose
    labelled agents are all OOD. nuScenes features are zero-padded to the ROAD width (type columns aligned)."""
    road = _road_graphs(str(data_dir), cfg)
    nuscenes = _nuscenes_graphs(str(data_dir), cfg)
    width = road["train"][0][0].size(1)
    graphs = _encode(road, spec.id_classes)
    for g in nuscenes["test"]:
        x, edge_index, labels = _pad_features(g, width)
        data = _to_data(
            x, edge_index, torch.where(labels >= 0, torch.full_like(labels, -2), labels), "test", []
        )
        data.y = torch.zeros((x.size(0), len(spec.id_classes)))
        data.test_mask = labels >= 0  # every labelled nuScenes agent is OOD
        graphs["test"].append(data)
    return graphs


_ROAD_SCENE = dict(kind="inductive", load_raw=None, homophily="temporal", split="public", batch_size=16)
# RS-GNN paper, Tabs. focal_sets and hyperparams (Adam, batch 16 graphs, alpha = beta = 1e-3).
_ROAD_PROTOCOL = {
    "focal_set_strategy": "budgeted",
    "budget_k": 64,
    "alpha": 1e-3,
    "beta": 1e-3,
    "weight_decay": 0.0,
    "lr": 1e-3,
    "max_epochs": 30,
    "early_stopping": False,  # train for the full number of epochs
}
_NUSCENES_PROTOCOL = {**_ROAD_PROTOCOL, "lr": 5e-4, "max_epochs": 80}

register_dataset(
    DatasetSpec(
        name="road",
        paper_name="ROAD",
        load_graphs=_load_road,
        id_classes=tuple(range(8)),
        ood_classes=(8, 9),
        num_total_classes=10,
        num_features=37,
        files=("road/road_trainval_v1.0.json",),
        notes="ROAD held-out OOD: 10 agent classes, OOD {8, 9}.",
        protocol=_ROAD_PROTOCOL,
        **_ROAD_SCENE,
    )
)
register_dataset(
    DatasetSpec(
        name="nuscenes",
        paper_name="nuScenes",
        load_graphs=_load_nuscenes,
        id_classes=tuple(range(6)),
        ood_classes=(6, 7),
        num_total_classes=8,
        num_features=10,
        files=("nuscenes/v1.0-mini/sample_annotation.json",),
        notes="nuScenes v1.0-mini held-out OOD: 8 categories, OOD {6, 7} (motorcycle, bicycle).",
        protocol=_NUSCENES_PROTOCOL,
        **_ROAD_SCENE,
    )
)
register_dataset(
    DatasetSpec(
        name="road_to_nuscenes",
        paper_name="ROAD(ID) vs nuScenes(OOD)",
        load_graphs=_load_road_to_nuscenes,
        id_classes=tuple(range(10)),
        ood_classes=(),
        num_total_classes=10,
        num_features=37,
        files=("road/road_trainval_v1.0.json", "nuscenes/v1.0-mini/sample_annotation.json"),
        notes="Cross-dataset shift: ROAD classes are ID, nuScenes test agents are OOD.",
        protocol=_ROAD_PROTOCOL,
        **_ROAD_SCENE,
    )
)
