from torch_geometric.utils import subgraph
from ogb.nodeproppred import NodePropPredDataset
import torch_geometric
import torch
from torch_geometric.loader import DataLoader
from .utils import one_hot_encode, even_quantile_labels
import scipy
from torch_geometric.loader import NeighborLoader


def loader_snap_patents_year(DATASET_STORAGE_PATH, config):
    """
    Loads the snap-patents dataset and prepares it for transductive OOD detection
    using the correct MASKING approach.

    This function prepares a single graph object and attaches all necessary masks to it.
    It supports both full-batch training (DataLoader) and mini-batching with neighborhood
    sampling (NeighborLoader).

    Args:
        DATASET_STORAGE_PATH (str): Path to the directory containing 'snap-patents.mat'.
        config (dict): A configuration dictionary for batch_size, num_neighbors, etc.

    Returns:
        A tuple of (train_loader, val_loader, test_loader).
    """
    # --- 1. Load the Full Graph Data ---
    fulldata = scipy.io.loadmat(f'{DATASET_STORAGE_PATH}/snap-patents.mat')

    edge_index = torch.tensor(fulldata['edge_index'], dtype=torch.long)
    node_feat = torch.tensor(fulldata['node_feat'].todense(), dtype=torch.float)
    years = fulldata['years'].flatten()

    # Create original class labels based on year quantiles
    original_labels = torch.tensor(even_quantile_labels(years, nclasses=5, verbose=False), dtype=torch.long)

    # Create a single, unified Data object for the entire graph
    data = torch_geometric.data.Data(x=node_feat, edge_index=edge_index, y=original_labels)
    num_nodes = data.num_nodes

    # --- 2. Prepare Masks ---
    OODclass = [0, 1]
    IDclass = [2, 3, 4]
    num_id_classes = len(IDclass)
    
    # Define split ratios
    train_ratio = 0.6
    val_ratio = 0.2

    # Create a random permutation for splitting
    indices = torch.randperm(num_nodes)
    train_size = int(train_ratio * num_nodes)
    val_size = int(val_ratio * num_nodes)
    
    # Create initial boolean masks based on the random split
    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)
    
    train_mask[indices[:train_size]] = True
    val_mask[indices[train_size:train_size + val_size]] = True
    test_mask[indices[train_size + val_size:]] = True

    # Create a mask to identify all OOD nodes in the entire graph
    ood_node_mask = torch.isin(data.y, torch.tensor(OODclass))

    # The final train and val masks must ONLY include ID-class nodes
    data.train_mask = train_mask & ~ood_node_mask
    data.val_mask = val_mask & ~ood_node_mask
    
    # The test mask should include BOTH ID and OOD nodes for evaluation
    data.test_mask = test_mask

    print(f"Nodes for training (ID only): {data.train_mask.sum().item()}")
    print(f"Nodes for validation (ID only): {data.val_mask.sum().item()}")
    print(f"Nodes for testing (ID + OOD): {data.test_mask.sum().item()}")

    # --- 3. Prepare the Unified Label Tensor (y) ---
    # The model expects a one-hot vector for ID classes and a zero-vector for OOD classes.
    new_y = torch.zeros((num_nodes, num_id_classes), dtype=torch.float)
    
    id_node_mask = ~ood_node_mask
    original_id_labels = data.y[id_node_mask]
    remapped_id_labels = original_id_labels - min(IDclass)
    
    new_y[id_node_mask] = one_hot_encode(remapped_id_labels, num_id_classes)
    
    # Replace the original `y` on the data object with the correctly formatted one
    data.y = new_y
    
    # --- 4. Create DataLoaders ---
    if config.get("batch_size", 1) <= 0: # Use get for safety
        print("Using DataLoader for full-batch training.")
        train_loader = DataLoader([data], batch_size=1, shuffle=False)
    else:
        print(f"Using NeighborLoader for mini-batch training with batch size {config['batch_size']}.")
        # Use the train_mask to specify the seed nodes for each mini-batch
        train_loader = NeighborLoader(
            data,
            input_nodes=data.train_mask, # This is the crucial change
            batch_size=config["batch_size"],
            num_neighbors=[int(config.get('num_neighbors', 10))] * int(config.get("num_layers", 2)),
            shuffle=True
        )

    # Validation and testing are usually done on the full graph
    val_loader = DataLoader([data], batch_size=1, shuffle=False)
    test_loader = DataLoader([data], batch_size=1, shuffle=False)
    
    return train_loader, val_loader, test_loader
