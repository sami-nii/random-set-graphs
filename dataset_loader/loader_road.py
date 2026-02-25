import os
import json
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from sklearn.neighbors import NearestNeighbors

def loader_road(DATASET_STORAGE_PATH, split_ratio=(0.6, 0.2, 0.2), k_spatial=5):
    """
    Loads ROAD dataset into a PyTorch Geometric graph.
    Uses only preprocessed annotations, no video parsing.
    Returns train/val/test DataLoaders with batch_size=1.
    """

    if len(split_ratio) != 3 or not abs(sum(split_ratio) - 1.0) < 1e-8:
        raise ValueError("split_ratio must be a 3-tuple that sums to 1.0.")
    if k_spatial < 1:
        raise ValueError("k_spatial must be >= 1.")

    json_file = os.path.join(DATASET_STORAGE_PATH, "road_trainval_v1.0.json")
    with open(json_file, "r") as f:
        final_annots = json.load(f)

    nodes = []
    labels = []
    edges = []
    tube_last_node = {}
    frame_nodes = {}
    node_idx = 0

    db = final_annots["db"]
    action_labels = final_annots["action_labels"]

    # Step 1: collect nodes and temporal edges
    for video_name in db.keys():
        frames = db[video_name]["frames"]
        for frame_id in frames.keys():
            if frames[frame_id]["annotated"] == 0:
                continue

            annos = frames[frame_id]["annos"]
            # Frame ids are reused across videos (e.g., "1", "2", ...), so scope by video.
            frame_key = (video_name, frame_id)
            frame_nodes[frame_key] = []

            for anno_id in annos.keys():
                anno = annos[anno_id]

                box = anno["box"]  # normalized [x1,y1,x2,y2]
                tube_uid = anno["tube_uid"]

                # Use first action label
                if len(anno["action_ids"]) == 0:
                    continue
                action_id = anno["action_ids"][0]

                nodes.append(box)
                labels.append(action_id)
                frame_nodes[frame_key].append(node_idx)

                # Temporal edges along tubes
                if tube_uid in tube_last_node:
                    prev = tube_last_node[tube_uid]
                    edges.append([prev, node_idx])
                    edges.append([node_idx, prev])
                tube_last_node[tube_uid] = node_idx

                node_idx += 1

    if len(nodes) == 0:
        raise ValueError("No annotated nodes found in ROAD dataset.")

    # Step 2: convert nodes and labels to tensors
    x = torch.tensor(nodes, dtype=torch.float)
    y_raw = torch.tensor(labels, dtype=torch.long)
    num_classes = max(y_raw.max().item() + 1, len(action_labels))
    y = torch.zeros((y_raw.shape[0], num_classes), dtype=torch.float)
    y[torch.arange(y_raw.shape[0]), y_raw] = 1.0

    # Step 3: add spatial edges (kNN per frame)
    for frame_id in frame_nodes:
        idxs = frame_nodes[frame_id]
        if len(idxs) < 2:
            continue

        coords = x[idxs][:, :2]  # use x1,y1 for spatial
        nbrs = NearestNeighbors(n_neighbors=min(k_spatial, len(idxs))).fit(coords)
        _, indices = nbrs.kneighbors(coords)
        for i, neighbors in enumerate(indices):
            for j in neighbors:
                if i != j:
                    edges.append([idxs[i], idxs[j]])

    if len(edges) == 0:
        edge_index = torch.empty((2, 0), dtype=torch.long)
    else:
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

    # Step 4: simple random split
    num_nodes = x.shape[0]
    perm = torch.randperm(num_nodes)
    train_end = int(split_ratio[0] * num_nodes)
    val_end = train_end + int(split_ratio[1] * num_nodes)

    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)
    train_mask[perm[:train_end]] = True
    val_mask[perm[train_end:val_end]] = True
    test_mask[perm[val_end:]] = True

    data = Data(
        x=x,
        edge_index=edge_index,
        y=y,
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask
    )

    print("ROAD Graph Created")
    print("Nodes:", x.shape[0])
    print("Edges:", edge_index.shape[1])
    print("Num classes:", num_classes)

    return (
        DataLoader([data], batch_size=1),
        DataLoader([data], batch_size=1),
        DataLoader([data], batch_size=1)
    )
