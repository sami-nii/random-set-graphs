"""Temporal road-scene graph benchmarks (paper §4.1 and App. "Temporal road-scene graph construction").

Each temporal window of a video/scene becomes one graph:
  * one unlabeled scene node per frame and one node per annotated agent;
  * intra-frame edges connect the scene node bidirectionally to every agent;
  * temporal edges connect scene nodes of adjacent frames and agents with the
    same track id (ROAD ``tube_uid``, nuScenes ``instance_token``) in adjacent
    frames; self-loops are added.
The task is inductive node classification of agent categories with held-out
OOD categories (ROAD: {8, 9}; nuScenes: {6, 7}).

Labels follow the repository convention: ``y`` is one-hot over the ID classes
and all-zero for OOD agents. Scene nodes and agents without a category are
unlabeled and excluded from every mask. Each graph carries ``train_mask``,
``val_mask`` and ``test_mask``; train graphs only mark labelled ID agents.
"""

import hashlib
import json
import os
from collections import defaultdict

import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

NUSCENES_CATEGORIES = ["car", "truck", "bus", "trailer", "construction_vehicle", "pedestrian", "motorcycle", "bicycle"]
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
    "scene-0061", "scene-0553", "scene-0655", "scene-0757",
    "scene-0796", "scene-1077", "scene-1094", "scene-1100",
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
    is_id = torch.tensor([int(l) in id_map for l in labels.tolist()], dtype=torch.bool)
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


def _report(name, graphs, id_classes):
    for split in ("train", "val", "test"):
        split_graphs = [g for g in graphs[split]]
        mask_name = f"{split}_mask"
        labelled = sum(int(getattr(g, mask_name).sum()) for g in split_graphs)
        print(f"{name} {split}: {len(split_graphs)} graphs, {labelled} labelled nodes")
    print(f"{name} ID classes: {id_classes}")


def _loaders(graphs, config):
    batch_size = int(config.get("batch_size", 16))
    batch_size = 16 if batch_size <= 0 else batch_size
    return (
        DataLoader(graphs["train"], batch_size=batch_size, shuffle=True),
        DataLoader(graphs["val"], batch_size=batch_size, shuffle=False),
        DataLoader(graphs["test"], batch_size=batch_size, shuffle=False),
    )


# ---------------------------------------------------------------------------
# ROAD
# ---------------------------------------------------------------------------
def _road_annotation_path(dataset_root, config):
    candidates = [
        config.get("road_root"),
        os.path.join(dataset_root, "road", "road_trainval_v1.0.json"),
        os.path.join(dataset_root, "road-dataset-master", "road", "road_trainval_v1.0.json"),
        os.path.join(dataset_root, "road_trainval_v1.0.json"),
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
    locs[[loc_map[l] for l in annotation.get("loc_ids", []) if l in loc_map]] = 1.0
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
            ("road_window_size", 4), ("road_window_stride", 4), ("road_frame_step", 5),
            ("road_train_split", "train_1"), ("road_eval_split", "val_1"), ("road_split_seed", 0),
        ]
    }
    return _cached(
        os.path.join(os.path.dirname(annotation_path), "processed"),
        "road_scene_graphs",
        params,
        lambda: build_road_graphs(annotation_path, config),
    )


def loader_road_scene(dataset_root, config):
    """ROAD held-out OOD: 10 agent classes, ID {0..7}, OOD {8, 9}."""
    raw = _road_graphs(dataset_root, config)
    ood_classes = [int(c) for c in config.get("road_ood_classes", [8, 9])]
    num_classes = int(config.get("road_num_classes", 10))
    id_classes = [c for c in range(num_classes) if c not in ood_classes]
    graphs = {split: [_to_data(*g, split, id_classes) for g in raw[split]] for split in raw}
    _report("ROAD", graphs, id_classes)
    return _loaders(graphs, config)


# ---------------------------------------------------------------------------
# nuScenes
# ---------------------------------------------------------------------------
def _nuscenes_metadata_root(config):
    root = config.get("nuscenes_root") or os.environ.get("NUSCENES_ROOT") or os.path.join("dataset", "nuscenes")
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
    instance_category = {row["token"]: _nuscenes_category(category_name[row["category_token"]]) for row in table("instance")}
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
                features = np.concatenate([
                    np.asarray(row["translation"]) - centre,  # position relative to the frame centroid
                    row["size"],
                    [np.log1p(row["num_lidar_pts"]), np.log1p(row["num_radar_pts"])],
                ]).astype(np.float32)
                agents.append((row["instance_token"], features, category))
            frames.append(agents)
            token = samples[token]["next"]
        raw_frames[scene["name"]] = frames

    # Standardise the physical features with statistics of the training scenes.
    train_features = np.asarray([a[1] for s in train_scenes for f in raw_frames.get(s, []) for a in f])
    mean, std = train_features.mean(axis=0), np.clip(train_features.std(axis=0), 1e-6, None)

    graphs = {"train": [], "val": [], "test": []}
    for scene_name, frames in raw_frames.items():
        split = "train" if scene_name in train_scenes else "val" if scene_name in val_scenes else "test" if scene_name in test_scenes else None
        if split is None:
            continue
        frames = [[(t, (x - mean) / std, c) for t, x, c in f] for f in frames]
        for window in _windows(len(frames), window_size, window_stride, 1):
            graphs[split].append(_window_graph([frames[i] for i in window], len(mean)))
    return graphs


def _nuscenes_graphs(config):
    metadata_root = _nuscenes_metadata_root(config)
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


def loader_nuscenes_scene(_, config):
    """nuScenes held-out OOD: 8 object categories, ID {0..5}, OOD {6, 7} (motorcycle, bicycle)."""
    raw = _nuscenes_graphs(config)
    ood_classes = [int(c) for c in config.get("nuscenes_ood_classes", [6, 7])]
    id_classes = [c for c in range(len(NUSCENES_CATEGORIES)) if c not in ood_classes]
    graphs = {split: [_to_data(*g, split, id_classes) for g in raw[split]] for split in raw}
    _report("nuScenes", graphs, id_classes)
    return _loaders(graphs, config)


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


def loader_road_to_nuscenes(dataset_root, config):
    """
    Train and validate on ROAD with all ROAD classes as ID; the test set adds
    nuScenes test graphs whose labelled agents are all OOD. nuScenes features
    are zero-padded to the ROAD feature width (node-type columns aligned).
    """
    road = _road_graphs(dataset_root, config)
    nuscenes = _nuscenes_graphs(config)
    width = road["train"][0][0].size(1)
    id_classes = list(range(int(config.get("road_num_classes", 10))))

    graphs = {split: [_to_data(*g, split, id_classes) for g in road[split]] for split in road}
    for g in nuscenes["test"]:
        x, edge_index, labels = _pad_features(g, width)
        # nuScenes agents are labelled but belong to no ROAD class (all-zero y).
        data = _to_data(x, edge_index, torch.where(labels >= 0, torch.full_like(labels, -2), labels), "test", id_classes)
        data.test_mask = labels >= 0
        graphs["test"].append(data)
    _report("ROAD->nuScenes", graphs, id_classes)
    return _loaders(graphs, config)
