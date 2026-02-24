import os
import torch
import numpy as np
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from sklearn.neighbors import NearestNeighbors
from .utils import one_hot_encode

def loader_road_direct(DATASET_STORAGE_PATH, split_ratio=(0.6,0.2,0.2), k_spatial=5):
    """
    Convert ROAD dataset directly into a single PyG Data object.
    Nodes = agent instances
    Edges = temporal + spatial
    """
    if not os.path.exists(DATASET_STORAGE_PATH):
        raise FileNotFoundError(f"ROAD dataset folder not found: {DATASET_STORAGE_PATH}")

    videos = [v for v in os.listdir(DATASET_STORAGE_PATH) 
              if os.path.isdir(os.path.join(DATASET_STORAGE_PATH, v))]
    if len(videos) == 0:
        raise FileNotFoundError(f"No video subfolders found in ROAD dataset path: {DATASET_STORAGE_PATH}")
    videos.sort()

    nodes = []
    labels = []
    edges = []
    node_index = 0
    track_last_node = {}
    frame_node_indices = []

    for video in videos:
        video_path = os.path.join(DATASET_STORAGE_PATH, video)
        annotation_path = os.path.join(video_path, "annotations")

        if not os.path.exists(annotation_path):
            raise FileNotFoundError(f"Annotations folder missing for video {video}: {annotation_path}")

        frame_files = [f for f in os.listdir(annotation_path) if f.endswith(".txt")]
        if len(frame_files) == 0:
            raise FileNotFoundError(f"No annotation files found in {annotation_path}")
        frame_files.sort()

        for frame_file in frame_files:
            frame_nodes = []
            frame_file_path = os.path.join(annotation_path, frame_file)

            with open(frame_file_path, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) < 6:
                        continue
                    track_id, x, y, w, h, action_id = map(int, parts[:6])
                    nodes.append([x, y, w, h])
                    labels.append(action_id)
                    frame_nodes.append(node_index)

                    if track_id in track_last_node:
                        edges.append([track_last_node[track_id], node_index])
                        edges.append([node_index, track_last_node[track_id]])
                    track_last_node[track_id] = node_index
                    node_index += 1

            frame_node_indices.append(frame_nodes)

            # Spatial edges using k-NN
            if len(frame_nodes) > 1:
                coords = np.array([nodes[i][:2] for i in frame_nodes])
                k = min(k_spatial, len(frame_nodes)-1)
                nbrs = NearestNeighbors(n_neighbors=k+1, algorithm="ball_tree").fit(coords)
                _, neighbors = nbrs.kneighbors(coords)
                for i, nbr_idxs in enumerate(neighbors):
                    for j in nbr_idxs[1:]:
                        edges.append([frame_nodes[i], frame_nodes[j]])
                        edges.append([frame_nodes[j], frame_nodes[i]])

    x = torch.tensor(nodes, dtype=torch.float)
    y_raw = torch.tensor(labels, dtype=torch.long)
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    num_nodes = x.shape[0]

    # Train/val/test split by track_id
    track_ids = list(track_last_node.keys())
    np.random.shuffle(track_ids)
    num_tracks = len(track_ids)
    train_end = int(split_ratio[0]*num_tracks)
    val_end = train_end + int(split_ratio[1]*num_tracks)

    # Map track to nodes
    track_to_nodes = {}
    for frame_nodes in frame_node_indices:
        for idx in frame_nodes:
            for t_id, last_idx in track_last_node.items():
                if last_idx >= idx:
                    track_to_nodes.setdefault(t_id, []).append(idx)
                    break

    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)
    for i, t_id in enumerate(track_ids):
        mask = torch.tensor(track_to_nodes[t_id], dtype=torch.long)
        if i < train_end:
            train_mask[mask] = True
        elif i < val_end:
            val_mask[mask] = True
        else:
            test_mask[mask] = True

    # ID/OOD split
    IDclass = [0,1,2]
    OODclass = [3,4]

    id_mask = torch.isin(y_raw, torch.tensor(IDclass))
    train_mask = train_mask & id_mask

    num_id_classes = len(IDclass)
    y = torch.zeros((num_nodes, num_id_classes), dtype=torch.float)
    remapped = y_raw[id_mask] - min(IDclass)
    y[id_mask] = one_hot_encode(remapped, num_id_classes)

    data = Data(
        x=x,
        edge_index=edge_index,
        y=y,
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask
    )

    print("ROAD Graph Created")
    print("Nodes:", num_nodes)
    print("Edges:", edge_index.shape[1])
    print("Feature dim:", x.shape[1])

    loader_train = DataLoader([data], batch_size=1)
    loader_val = DataLoader([data], batch_size=1)
    loader_test = DataLoader([data], batch_size=1)

    return loader_train, loader_val, loader_test