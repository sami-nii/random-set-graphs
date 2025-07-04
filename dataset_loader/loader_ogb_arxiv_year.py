from torch_geometric.utils import subgraph
from ogb.nodeproppred import NodePropPredDataset
import torch_geometric
import torch
from torch_geometric.loader import DataLoader
from .utils import one_hot_encode, even_quantile_labels


import torch
import torch.nn.functional as F
import torch_geometric
from torch_geometric.loader import DataLoader
import numpy as np # Numpy is required for quantile binning


def load_ogb_arxiv_year(DATASET_STORAGE_PATH, config):
    """
    Loads the ogbn-arxiv dataset and prepares it for transductive OOD detection
    using the correct MASKING approach. The class labels are derived from node_year quantiles.

    This function prepares a single graph object and attaches all necessary masks to it.

    Args:
        DATASET_STORAGE_PATH (str): Path to store the OGB dataset.
        config (dict): A configuration dictionary.

    Returns:
        A tuple of (train_loader, val_loader, test_loader).
    """
    # --- 1. Load the Full Graph Data from OGB ---
    ogb_dataset = NodePropPredDataset(name='ogbn-arxiv', root=DATASET_STORAGE_PATH)
    graph = ogb_dataset.graph

    # Define class splits based on publication year
    nclass = 5
    original_labels = torch.tensor(
        even_quantile_labels(graph['node_year'].flatten(), nclass, verbose=True),
        dtype=torch.long
    )

    # Create a single, unified Data object for the entire graph
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
    
    # Define split ratios for a random split
    train_ratio = 0.6
    val_ratio = 0.2

    indices = torch.randperm(num_nodes)
    train_size = int(train_ratio * num_nodes)
    val_size = int(val_ratio * num_nodes)
    
    # Create initial boolean masks for the entire graph
    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)
    
    train_mask[indices[:train_size]] = True
    val_mask[indices[train_size:train_size + val_size]] = True
    test_mask[indices[train_size + val_size:]] = True

    # Create a mask to identify all OOD nodes
    ood_node_mask = torch.isin(data.y, torch.tensor(OODclass))

    # Final train/val masks must only contain ID nodes
    data.train_mask = train_mask & ~ood_node_mask
    data.val_mask = val_mask & ~ood_node_mask
    
    # Test mask contains everything from its split
    data.test_mask = test_mask

    print(f"Nodes for training (ID only): {data.train_mask.sum().item()}")
    print(f"Nodes for validation (ID only): {data.val_mask.sum().item()}")
    print(f"Nodes for testing (ID + OOD): {data.test_mask.sum().item()}")

    # --- 3. Prepare the Unified Label Tensor (y) ---
    new_y = torch.zeros((num_nodes, num_id_classes), dtype=torch.float)
    id_node_mask = ~ood_node_mask
    
    original_id_labels = data.y[id_node_mask]
    remapped_id_labels = original_id_labels - min(IDclass)
    
    new_y[id_node_mask] = one_hot_encode(remapped_id_labels, num_id_classes)
    data.y = new_y # Replace original labels with the new formatted tensor
    
    # --- 4. Create DataLoaders ---
    # The loaders will yield the same single graph object.
    # The model is responsible for using the masks internally.
    train_loader = DataLoader([data], batch_size=1, shuffle=False)
    val_loader = DataLoader([data], batch_size=1, shuffle=False)
    test_loader = DataLoader([data], batch_size=1, shuffle=False)

    return train_loader, val_loader, test_loader