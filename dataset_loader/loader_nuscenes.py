"""Bounded nuScenes object-category graph loader without the devkit dependency."""

import json
import os
from collections import Counter, defaultdict

import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.utils import coalesce


def _resolve_paths(config):
    root = config.get("nuscenes_root") or os.environ.get("NUSCENES_ROOT")
    if not root:
        root = r"S:\nuScenes\v1.0-trainval"
    version = config.get("nuscenes_version", "v1.0-trainval")
    metadata_root = os.path.join(root, version)
    if not os.path.isfile(os.path.join(metadata_root, "sample_annotation.json")):
        raise FileNotFoundError(
            f"Could not find nuScenes {version} metadata under {metadata_root}. "
            "Set nuscenes_root to the directory containing samples/, sweeps/, and the version directory."
        )
    return root, metadata_root


def _load_table(metadata_root, name):
    with open(os.path.join(metadata_root, f"{name}.json"), encoding="utf-8") as handle:
        return json.load(handle)


def _scene_split(scene_tokens, seed, train_ratio, val_ratio):
    generator = np.random.default_rng(seed)
    tokens = np.asarray(sorted(scene_tokens))
    generator.shuffle(tokens)
    train_end = int(train_ratio * len(tokens))
    val_end = train_end + int(val_ratio * len(tokens))
    return set(tokens[:train_end]), set(tokens[train_end:val_end]), set(tokens[val_end:])


def _select_complete_scenes(scene_groups, annotation_counts, max_nodes, seed):
    """Select whole scenes approximately up to the requested node budget."""
    if max_nodes <= 0:
        return set().union(*scene_groups.values())

    generator = np.random.default_rng(seed)
    total_ratio = len(scene_groups["train"]) + len(scene_groups["val"]) + len(scene_groups["test"])
    selected = set()
    for split_name, scenes in scene_groups.items():
        target = max(1, int(max_nodes * len(scenes) / total_ratio))
        candidates = np.asarray(sorted(scenes))
        generator.shuffle(candidates)
        split_total = 0
        for scene in candidates:
            count = annotation_counts[scene]
            # Always keep the first scene for each split. This makes the cap
            # approximate for very small requested budgets, but avoids empty
            # validation/test splits and preserves complete scene structure.
            if split_total == 0 or split_total + count <= target:
                selected.add(scene)
                split_total += count
    return selected


def _build_edges(records, spatial_neighbors):
    token_to_index = {record[0]: index for index, record in enumerate(records)}
    by_sample = defaultdict(list)
    edges = []
    for index, record in enumerate(records):
        by_sample[record[1]].append(index)
        previous = record[4]
        if previous and previous in token_to_index:
            previous_index = token_to_index[previous]
            edges.extend(((previous_index, index), (index, previous_index)))

    features = np.asarray([record[5] for record in records], dtype=np.float32)
    for indices in by_sample.values():
        if len(indices) < 2:
            continue
        positions = features[indices, :3]
        distances = np.sum((positions[:, None, :] - positions[None, :, :]) ** 2, axis=-1)
        for local_index, node_index in enumerate(indices):
            nearest = np.argsort(distances[local_index])[1 : spatial_neighbors + 1]
            edges.extend((node_index, indices[neighbour]) for neighbour in nearest)

    if not edges:
        return torch.empty((2, 0), dtype=torch.long)
    return coalesce(torch.tensor(edges, dtype=torch.long).t().contiguous())


def loader_nuscenes(_, config):
    """Create a deterministic transductive nuScenes ID/OOD object graph.

    This is an exploratory object-category protocol, not the official nuScenes
    detection benchmark. Whole scenes are assigned to train/validation/test,
    and only ID-category nodes are eligible for the training loss. The node
    budget is approximate because whole scenes are retained to preserve edges.
    """
    root, metadata_root = _resolve_paths(config)
    max_nodes = int(config.get("nuscenes_max_nodes", 20_000))
    spatial_neighbors = int(config.get("nuscenes_spatial_neighbors", 5))
    split_seed = int(config.get("nuscenes_split_seed", 0))
    train_ratio = float(config.get("nuscenes_train_ratio", 0.7))
    val_ratio = float(config.get("nuscenes_val_ratio", 0.15))
    if not 0 < train_ratio < 1 or not 0 < val_ratio < 1 or train_ratio + val_ratio >= 1:
        raise ValueError("nuScenes train and validation ratios must be positive and sum to less than one.")
    if spatial_neighbors < 1:
        raise ValueError("nuscenes_spatial_neighbors must be at least one.")

    categories = _load_table(metadata_root, "category")
    instances = _load_table(metadata_root, "instance")
    samples = _load_table(metadata_root, "sample")
    scenes = _load_table(metadata_root, "scene")
    annotations = _load_table(metadata_root, "sample_annotation")

    category_name = {row["token"]: row["name"] for row in categories}
    category_by_instance = {row["token"]: row["category_token"] for row in instances}
    scene_by_sample = {row["token"]: row["scene_token"] for row in samples}
    available_scenes = {row["token"] for row in scenes}

    raw_categories = np.asarray(
        [category_by_instance[row["instance_token"]] for row in annotations], dtype=object
    )
    configured_id_classes = config.get("nuscenes_id_classes")
    if configured_id_classes is None:
        id_categories = [
            category for category, _ in Counter(raw_categories).most_common(
                int(config.get("nuscenes_num_id_classes", 3))
            )
        ]
    else:
        requested_names = set(configured_id_classes)
        id_categories = [token for token, name in category_name.items() if name in requested_names]
        missing = requested_names - {category_name[token] for token in id_categories}
        if missing:
            raise ValueError(f"Unknown nuScenes category names: {sorted(missing)}")
    if len(id_categories) < 2:
        raise ValueError("nuScenes requires at least two ID categories.")

    annotation_scenes = np.asarray(
        [scene_by_sample[row["sample_token"]] for row in annotations], dtype=object
    )
    scene_train, scene_val, scene_test = _scene_split(
        available_scenes, split_seed, train_ratio, val_ratio
    )
    selected_scenes = _select_complete_scenes(
        {"train": scene_train, "val": scene_val, "test": scene_test},
        Counter(annotation_scenes),
        max_nodes,
        split_seed,
    )
    selected_indices = np.flatnonzero(np.isin(annotation_scenes, list(selected_scenes)))

    # token, sample token, scene token, category token, previous token, features
    records = []
    for index in selected_indices:
        annotation = annotations[index]
        translation = annotation["translation"]
        size = annotation["size"]
        records.append((
            annotation["token"],
            annotation["sample_token"],
            scene_by_sample[annotation["sample_token"]],
            category_by_instance[annotation["instance_token"]],
            annotation["prev"],
            [
                translation[0], translation[1], translation[2],
                size[0], size[1], size[2],
                np.log1p(annotation["num_lidar_pts"]),
                np.log1p(annotation["num_radar_pts"]),
            ],
        ))
    del annotations

    categories_selected = np.asarray([record[3] for record in records], dtype=object)
    is_id = np.isin(categories_selected, id_categories)
    if not np.any(is_id):
        raise ValueError("No nuScenes nodes remain after applying nuscenes_id_classes.")
    label_to_id = {category: index for index, category in enumerate(id_categories)}
    labels = torch.zeros((len(records), len(id_categories)), dtype=torch.float)
    for category, label in label_to_id.items():
        labels[np.flatnonzero(categories_selected == category), label] = 1.0

    node_scenes = np.asarray([record[2] for record in records], dtype=object)
    train_mask = torch.tensor(np.isin(node_scenes, list(scene_train)) & is_id, dtype=torch.bool)
    val_mask = torch.tensor(np.isin(node_scenes, list(scene_val)), dtype=torch.bool)
    test_mask = torch.tensor(np.isin(node_scenes, list(scene_test)), dtype=torch.bool)
    if not train_mask.any() or not val_mask.any() or not test_mask.any():
        raise RuntimeError("nuScenes sampling produced an empty split; increase nuscenes_max_nodes.")

    features = np.asarray([record[5] for record in records], dtype=np.float32)
    # Normalise physical features using ID training nodes only to avoid test-set leakage.
    mean = features[train_mask.numpy()].mean(axis=0, keepdims=True)
    std = features[train_mask.numpy()].std(axis=0, keepdims=True)
    features = (features - mean) / np.clip(std, 1e-6, None)
    data = Data(
        x=torch.tensor(features, dtype=torch.float),
        edge_index=_build_edges(records, spatial_neighbors),
        y=labels,
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask,
    )

    id_names = [category_name[category] for category in id_categories]
    node_is_id = torch.from_numpy(is_id)
    print("--- nuScenes object-category graph ---")
    print(f"Root: {root}; metadata: {metadata_root}")
    print(f"Nodes: {data.num_nodes}; edges: {data.edge_index.size(1)}")
    print(f"ID categories ({len(id_names)}): {id_names}")
    print(
        "Splits: "
        f"train={int(train_mask.sum())} ID, "
        f"val={int(val_mask.sum())} ({int((val_mask & node_is_id).sum())} ID), "
        f"test={int(test_mask.sum())} ({int((test_mask & node_is_id).sum())} ID)"
    )
    return DataLoader([data], batch_size=1), DataLoader([data], batch_size=1), DataLoader([data], batch_size=1)
