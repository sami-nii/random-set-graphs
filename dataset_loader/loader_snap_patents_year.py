import torch_geometric
import torch
import os
import errno
import numpy as np
import time
from .utils import one_hot_encode, even_quantile_labels
import scipy
from torch_geometric.loader import NeighborLoader


def _dense_features_from_sparse(node_feat_sparse, shape):
    node_feat_coo = node_feat_sparse.tocoo()
    dense_features = np.zeros(shape, dtype=np.int8)
    dense_features[node_feat_coo.row, node_feat_coo.col] = node_feat_coo.data.astype(np.int8, copy=False)
    return dense_features


def _resolve_feature_cache_path(dataset_storage_path, config):
    cache_dir = (
        config.get("feature_cache_dir")
        or os.environ.get("RSG_FEATURE_CACHE_DIR")
        or dataset_storage_path
    )
    return os.path.join(cache_dir, "snap-patents-features.int8.memmap")


def _remove_partial_cache_file(memmap_path):
    if os.path.exists(memmap_path):
        try:
            os.remove(memmap_path)
        except OSError:
            pass


def _build_or_load_feature_tensor(node_feat_sparse, memmap_path):
    """
    Materialize the sparse SNAP feature matrix into an on-disk int8 memmap.

    This avoids SciPy's eager dense allocation, which can fail on Windows even
    though the matrix itself is small enough to stream from disk for mini-batch
    training.
    """
    shape = tuple(int(dim) for dim in node_feat_sparse.shape)
    expected_nbytes = int(np.prod(shape, dtype=np.int64))

    if os.path.exists(memmap_path):
        actual_nbytes = os.path.getsize(memmap_path)
        if actual_nbytes != expected_nbytes:
            print(
                f"Removing stale SNAP patents feature cache at {memmap_path} "
                f"(expected {expected_nbytes} bytes, found {actual_nbytes})."
            )
            _remove_partial_cache_file(memmap_path)

    try:
        if not os.path.exists(memmap_path):
            print(f"Building cached SNAP patents feature memmap at {memmap_path} ...")
            cache_dir = os.path.dirname(memmap_path)
            if cache_dir:
                os.makedirs(cache_dir, exist_ok=True)
            dense_features = np.memmap(memmap_path, dtype=np.int8, mode="w+", shape=shape)
            dense_features[:] = 0

            node_feat_coo = node_feat_sparse.tocoo()
            dense_features[node_feat_coo.row, node_feat_coo.col] = node_feat_coo.data.astype(np.int8, copy=False)
            dense_features.flush()
            del dense_features

        dense_features = np.memmap(memmap_path, dtype=np.int8, mode="r", shape=shape)
        return torch.from_numpy(dense_features)
    except OSError as exc:
        if exc.errno == errno.ENOSPC:
            print(
                "Disk is full while preparing the SNAP patents feature cache. "
                "Falling back to an in-memory dense tensor instead."
            )
            _remove_partial_cache_file(memmap_path)
            dense_features = _dense_features_from_sparse(node_feat_sparse, shape)
            return torch.from_numpy(dense_features)
        raise


def loader_snap_patents_year(DATASET_STORAGE_PATH, config):
    """
    Loads the snap-patents dataset and prepares it for transductive OOD detection,
    now including OOD samples in the validation set for hyperparameter tuning.

    Args:
        DATASET_STORAGE_PATH (str): Path to the directory containing 'snap-patents.mat'.
        config (dict): A configuration dictionary for batch_size, num_neighbors, etc.

    Returns:
        A tuple of (train_loader, val_loader, test_loader).
    """
    # --- 1. Load the Full Graph Data ---
    fulldata = scipy.io.loadmat(f'{DATASET_STORAGE_PATH}/snap-patents.mat')

    edge_index = torch.tensor(fulldata['edge_index'], dtype=torch.long)
    node_feat_sparse = fulldata['node_feat']
    node_feat_cache_path = _resolve_feature_cache_path(DATASET_STORAGE_PATH, config)
    node_feat = _build_or_load_feature_tensor(node_feat_sparse, node_feat_cache_path)
    years = fulldata['years'].flatten()

    original_labels = torch.tensor(even_quantile_labels(years, nclasses=5, verbose=False), dtype=torch.long)
    data = torch_geometric.data.Data(x=node_feat, edge_index=edge_index, y=original_labels)
    num_nodes = data.num_nodes

    # --- 2. Prepare Masks ---
    OODclass = [0, 1]
    IDclass = [2, 3, 4]
    num_id_classes = len(IDclass)
    
    train_ratio = 0.6
    val_ratio = 0.2

    indices = torch.randperm(num_nodes)
    train_size = int(train_ratio * num_nodes)
    val_size = int(val_ratio * num_nodes)
    
    # Create initial boolean masks for the random splits
    train_mask_split = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask_split = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask_split = torch.zeros(num_nodes, dtype=torch.bool)
    
    train_mask_split[indices[:train_size]] = True
    val_mask_split[indices[train_size:train_size + val_size]] = True
    test_mask_split[indices[train_size + val_size:]] = True

    # --- 3. Create the Final Masks for the Data Object ---
    ood_node_mask = torch.isin(data.y, torch.tensor(OODclass))
    id_node_mask = ~ood_node_mask

    # The final train mask must ONLY include ID-class nodes
    data.train_mask = train_mask_split & id_node_mask
    
    # The validation mask should now include ALL nodes (ID and OOD) from its split
    data.val_mask = val_mask_split
    
    # The test mask also includes ALL nodes from its split
    data.test_mask = test_mask_split

    # --- Reporting for verification ---
    print("--- SNAP-Patents (Year) Dataset with OOD Validation ---")
    id_val_nodes = data.val_mask & id_node_mask
    ood_val_nodes = data.val_mask & ood_node_mask
    id_test_nodes = data.test_mask & id_node_mask
    ood_test_nodes = data.test_mask & ood_node_mask
    
    print(f"Nodes for training (ID only): {data.train_mask.sum().item()}")
    print(f"Nodes for validation (ID+OOD): {data.val_mask.sum().item()} -> {id_val_nodes.sum()} ID, {ood_val_nodes.sum()} OOD")
    print(f"Nodes for testing (ID+OOD): {data.test_mask.sum().item()} -> {id_test_nodes.sum()} ID, {ood_test_nodes.sum()} OOD")
    
    # --- 4. Prepare the Unified Label Tensor (y) ---
    new_y = torch.zeros((num_nodes, num_id_classes), dtype=torch.float)
    original_id_labels = data.y[id_node_mask]
    remapped_id_labels = original_id_labels - min(IDclass)
    new_y[id_node_mask] = one_hot_encode(remapped_id_labels, num_id_classes)
    data.y = new_y
    
    # --- 5. Create DataLoaders ---
    requested_num_workers = int(config.get("num_workers", 11))
    num_workers = requested_num_workers

    # Windows multiprocessing can fail on this large graph when workers try to
    # share tensor storage across processes, so fall back to single-process loading.
    if os.name == "nt" and num_workers > 0:
        print(
            f"Windows detected: overriding num_workers from {requested_num_workers} "
            "to 0 for SNAP patents to avoid DataLoader shared-memory errors."
        )
        num_workers = 0

    dataloader_kwargs = {
        "num_workers": num_workers,
        "persistent_workers": num_workers > 0,
    }

    if config.get("batch_size", -1) <= 0:
        raise ValueError(
            "SNAP patents is too large for full-batch dense feature training in this "
            "pipeline. Please set a positive `batch_size` to use NeighborLoader."
        )
    else:
        batch_size = int(config["batch_size"])
        eval_batch_size = int(config.get("eval_batch_size", batch_size))
        num_neighbors = [int(config.get('num_neighbors', 10))] * int(config.get("num_layers", 2))

        print(f"Using NeighborLoader for mini-batch training with batch size {batch_size}.")
        train_loader_start = time.perf_counter()
        train_loader = NeighborLoader(
            data,
            input_nodes=data.train_mask,
            batch_size=batch_size,
            num_neighbors=num_neighbors,
            shuffle=True,
            **dataloader_kwargs,
        )
        print(f"Train NeighborLoader ready in {time.perf_counter() - train_loader_start:.1f}s.")

        print(f"Using NeighborLoader for validation/test with batch size {eval_batch_size}.")
        val_loader_start = time.perf_counter()
        val_loader = NeighborLoader(
            data,
            input_nodes=data.val_mask,
            batch_size=eval_batch_size,
            num_neighbors=num_neighbors,
            shuffle=False,
            **dataloader_kwargs,
        )
        print(f"Validation NeighborLoader ready in {time.perf_counter() - val_loader_start:.1f}s.")

        test_loader_start = time.perf_counter()
        test_loader = NeighborLoader(
            data,
            input_nodes=data.test_mask,
            batch_size=eval_batch_size,
            num_neighbors=num_neighbors,
            shuffle=False,
            **dataloader_kwargs,
        )
        print(f"Test NeighborLoader ready in {time.perf_counter() - test_loader_start:.1f}s.")
    
    return train_loader, val_loader, test_loader
