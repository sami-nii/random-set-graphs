"""ROAD object-action graph loader.

Nodes are annotated road agents in individual video frames.  Node features are
the normalised bounding-box centre and size; edges link consecutive observations
of the same track and spatial nearest neighbours in the same video frame.
"""

import json
import os
from collections import Counter, defaultdict

import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.utils import coalesce


def _annotation_path(dataset_root, config):
    configured_root = config.get("road_root")
    candidates = [
        configured_root,
        os.path.join(dataset_root, "road", "road_trainval_v1.0.json"),
        os.path.join(dataset_root, "road-dataset-master", "road", "road_trainval_v1.0.json"),
        os.path.join(dataset_root, "road_trainval_v1.0.json"),
    ]
    for candidate in candidates:
        if candidate and os.path.isfile(candidate):
            return candidate
    raise FileNotFoundError(
        "Could not find ROAD annotations. Set road_root to road_trainval_v1.0.json, "
        "or place it under dataset/road/ or dataset/road-dataset-master/road/."
    )


def _split_per_class(primary_labels, train_ratio, val_ratio, seed):
    generator = np.random.default_rng(seed)
    train = np.zeros(len(primary_labels), dtype=bool)
    val = np.zeros(len(primary_labels), dtype=bool)
    test = np.zeros(len(primary_labels), dtype=bool)

    for label in np.unique(primary_labels):
        indices = np.flatnonzero(primary_labels == label)
        generator.shuffle(indices)
        train_end = int(train_ratio * len(indices))
        val_end = train_end + int(val_ratio * len(indices))
        train[indices[:train_end]] = True
        val[indices[train_end:val_end]] = True
        test[indices[val_end:]] = True
    return train, val, test


def _build_edges(records, spatial_neighbors):
    by_track = defaultdict(list)
    by_frame = defaultdict(list)
    for index, record in enumerate(records):
        by_track[(record[0], record[2])].append((record[1], index))
        by_frame[(record[0], record[1])].append(index)

    edges = []
    for observations in by_track.values():
        observations.sort()
        for (_, left), (_, right) in zip(observations, observations[1:]):
            edges.extend(((left, right), (right, left)))

    features = np.asarray([record[3] for record in records], dtype=np.float32)
    for indices in by_frame.values():
        if len(indices) < 2:
            continue
        centres = features[indices, :2]
        distances = np.sum((centres[:, None, :] - centres[None, :, :]) ** 2, axis=-1)
        for local_index, node_index in enumerate(indices):
            nearest = np.argsort(distances[local_index])[1 : spatial_neighbors + 1]
            edges.extend((node_index, indices[neighbour]) for neighbour in nearest)

    if not edges:
        return torch.empty((2, 0), dtype=torch.long)
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    return coalesce(edge_index)


def loader_road(dataset_root, config):
    """Return a deterministic transductive ROAD graph for ID/OOD evaluation.

    The default 20,000-node cap keeps full-batch experiments practical on a
    CPU-only workstation.  Set ``road_max_nodes`` to 0 to construct the full
    graph, or set ``road_id_classes`` to an explicit list of original ROAD
    action-label indices for a different leave-out-class split.
    """
    annotation_path = _annotation_path(dataset_root, config)
    max_nodes = int(config.get("road_max_nodes", 20_000))
    spatial_neighbors = int(config.get("road_spatial_neighbors", 5))
    split_seed = int(config.get("road_split_seed", 0))
    train_ratio = float(config.get("road_train_ratio", 0.6))
    val_ratio = float(config.get("road_val_ratio", 0.2))
    if not 0 < train_ratio < 1 or not 0 < val_ratio < 1 or train_ratio + val_ratio >= 1:
        raise ValueError("ROAD train and validation ratios must be positive and sum to less than one.")
    if spatial_neighbors < 1:
        raise ValueError("road_spatial_neighbors must be at least one.")

    with open(annotation_path, encoding="utf-8") as handle:
        annotations = json.load(handle)

    # (video identifier, frame number, tube identifier, [centre_x, centre_y,
    # width, height], primary action label)
    records = []
    for video_name, video in annotations["db"].items():
        for frame_name, frame in video["frames"].items():
            if not frame.get("annotated"):
                continue
            for annotation in frame.get("annos", {}).values():
                action_ids = annotation.get("action_ids", [])
                if not action_ids:
                    continue
                x1, y1, x2, y2 = annotation["box"]
                records.append((
                    video_name,
                    int(frame_name),
                    annotation["tube_uid"],
                    [(x1 + x2) / 2.0, (y1 + y2) / 2.0, x2 - x1, y2 - y1],
                    int(action_ids[0]),
                ))

    if not records:
        raise ValueError("ROAD annotations contain no object instances with action labels.")

    labels_before_sampling = np.asarray([record[4] for record in records], dtype=np.int64)
    configured_id_classes = config.get("road_id_classes")
    if configured_id_classes is None:
        id_classes = [
            label for label, _ in Counter(labels_before_sampling).most_common(
                int(config.get("road_num_id_classes", 3))
            )
        ]
    else:
        id_classes = [int(label) for label in configured_id_classes]
    if len(id_classes) < 2:
        raise ValueError("ROAD requires at least two ID action classes.")

    if max_nodes > 0 and len(records) > max_nodes:
        generator = np.random.default_rng(split_seed)
        selected = np.sort(generator.choice(len(records), size=max_nodes, replace=False))
        records = [records[index] for index in selected]

    primary_labels = np.asarray([record[4] for record in records], dtype=np.int64)
    label_to_id = {label: index for index, label in enumerate(id_classes)}
    is_id = np.isin(primary_labels, id_classes)
    if not np.any(is_id):
        raise ValueError("No ROAD nodes remain after applying road_id_classes.")

    features = torch.tensor(np.asarray([record[3] for record in records]), dtype=torch.float)
    labels = torch.zeros((len(records), len(id_classes)), dtype=torch.float)
    for original_label, id_label in label_to_id.items():
        node_indices = np.flatnonzero(primary_labels == original_label)
        labels[node_indices, id_label] = 1.0

    train, val, test = _split_per_class(primary_labels, train_ratio, val_ratio, split_seed)
    train_mask = torch.tensor(train & is_id, dtype=torch.bool)
    val_mask = torch.tensor(val, dtype=torch.bool)
    test_mask = torch.tensor(test, dtype=torch.bool)
    edge_index = _build_edges(records, spatial_neighbors)
    data = Data(
        x=features,
        edge_index=edge_index,
        y=labels,
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask,
    )

    action_names = annotations.get("all_action_labels", annotations.get("action_labels", []))
    id_names = [action_names[label] if label < len(action_names) else str(label) for label in id_classes]
    print("--- ROAD object-action graph ---")
    print(f"Annotations: {annotation_path}")
    print(f"Nodes: {data.num_nodes}; edges: {data.edge_index.size(1)}")
    print(f"ID action classes ({len(id_classes)}): {id_names}")
    print(
        "Splits: "
        f"train={int(train_mask.sum())} ID, "
        f"val={int(val_mask.sum())} ({int((val_mask & torch.from_numpy(is_id)).sum())} ID), "
        f"test={int(test_mask.sum())} ({int((test_mask & torch.from_numpy(is_id)).sum())} ID)"
    )
    return DataLoader([data], batch_size=1), DataLoader([data], batch_size=1), DataLoader([data], batch_size=1)
