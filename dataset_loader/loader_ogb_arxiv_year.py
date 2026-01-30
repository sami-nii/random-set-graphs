from torch_geometric.utils import subgraph
from ogb.nodeproppred import NodePropPredDataset
import torch_geometric
import torch
from torch_geometric.loader import DataLoader, NeighborLoader
from .utils import one_hot_encode, even_quantile_labels


import torch
import torch.nn.functional as F
import torch_geometric
from torch_geometric.loader import DataLoader
import numpy as np # Numpy is required for quantile binning


def load_ogb_arxiv_year(DATASET_STORAGE_PATH, config):
    """
    Loads the ogbn-arxiv dataset and prepares it for transductive OOD detection,
    now including OOD samples in the validation set for hyperparameter tuning.

    Args:
        DATASET_STORAGE_PATH (str): Path to store the OGB dataset.
        config (dict): A configuration dictionary.

    Returns:
        A tuple of (train_loader, val_loader, test_loader).
    """
    # --- 1. Load the Full Graph Data from OGB ---
    ogb_dataset = NodePropPredDataset(name='ogbn-arxiv', root=DATASET_STORAGE_PATH)
    graph = ogb_dataset.graph

    nclass = 5
    original_labels = torch.tensor(
        even_quantile_labels(graph['node_year'].flatten(), nclass, verbose=True),
        dtype=torch.long
    )

    data = torch_geometric.data.Data(
        x=torch.as_tensor(graph['node_feat']),
        edge_index=torch.as_tensor(graph['edge_index']),
        y=original_labels
    )
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
    print("--- OGBN-Arxiv (Year) Dataset with OOD Validation ---")
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
    if config.get("batch_size", -1) <= 0:
        print("Using DataLoader for full-batch training.")
        train_loader = DataLoader([data], batch_size=1, shuffle=False)
    else:
        print(f"Using NeighborLoader for mini-batch training with batch size {config['batch_size']}.")
        train_loader = NeighborLoader(
            data,
            input_nodes=data.train_mask, # Crucial: sample starting nodes from the train mask
            batch_size=config["batch_size"],
            num_neighbors=[int(config.get('num_neighbors', 10))] * int(config.get("num_layers", 2)),
            shuffle=True
        )
    val_loader = DataLoader([data], batch_size=1, shuffle=False)
    test_loader = DataLoader([data], batch_size=1, shuffle=False)

    return train_loader, val_loader, test_loader