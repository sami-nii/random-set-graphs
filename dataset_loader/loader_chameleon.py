from torch_geometric.utils import subgraph
import torch
from torch_geometric.loader import DataLoader
from .utils import one_hot_encode
from torch_geometric.datasets import WikipediaNetwork

def loader_chameleon(DATASET_STORAGE_PATH, config, split_index=0):
    """
    Loads the Chameleon dataset and prepares it for transductive OOD detection,
    now including OOD samples in the validation set for hyperparameter tuning.

    Args:
        DATASET_STORAGE_PATH (str): Path to store the WikipediaNetwork dataset.
        config (dict): A configuration dictionary (batch_size is ignored).
        split_index (int): The dataset provides 10 different train/val/test splits.
                           This selects which one to use. Defaults to 0.

    Returns:
        A tuple of (train_loader, val_loader, test_loader).
    """
    # --- 1. Load the Full Graph Data ---
    dataset = WikipediaNetwork(root=DATASET_STORAGE_PATH, name='chameleon')
    data = dataset[0]  # Get the single graph object

    # --- 2. Prepare Masks based on specified ID/OOD classes ---
    OODclass = [0, 1]
    IDclass = [2, 3, 4]
    num_id_classes = len(IDclass)
    
    # The dataset has 10 splits; we select one using the split_index
    original_train_mask = data.train_mask[:, split_index]
    original_val_mask = data.val_mask[:, split_index]
    original_test_mask = data.test_mask[:, split_index]

    # Create masks for the entire node population
    ood_node_mask = torch.isin(data.y, torch.tensor(OODclass))
    id_node_mask = ~ood_node_mask

    # --- 3. Create the Final Masks for the Data Object ---

    # The final train mask must ONLY include ID-class nodes
    data.train_mask = original_train_mask & id_node_mask
    
    # The validation mask should now include ALL nodes (ID and OOD) from its original split
    data.val_mask = original_val_mask
    
    # The test mask should also include ALL nodes from its original split
    data.test_mask = original_test_mask

    # --- Reporting for verification ---
    print("--- Chameleon Dataset with OOD Validation ---")
    id_val_nodes = data.val_mask & id_node_mask
    ood_val_nodes = data.val_mask & ood_node_mask
    id_test_nodes = data.test_mask & id_node_mask
    ood_test_nodes = data.test_mask & ood_node_mask
    
    print(f"Nodes for training (ID only): {data.train_mask.sum().item()}")
    print(f"Nodes for validation (ID+OOD): {data.val_mask.sum().item()} -> {id_val_nodes.sum()} ID, {ood_val_nodes.sum()} OOD")
    print(f"Nodes for testing (ID+OOD): {data.test_mask.sum().item()} -> {id_test_nodes.sum()} ID, {ood_test_nodes.sum()} OOD")

    # --- 4. Prepare the Unified Label Tensor (y) ---
    new_y = torch.zeros((data.num_nodes, num_id_classes), dtype=torch.float)
    
    original_id_labels = data.y[id_node_mask]
    remapped_id_labels = original_id_labels - min(IDclass)
    
    new_y[id_node_mask] = one_hot_encode(remapped_id_labels, num_id_classes)
    data.y = new_y
    
    # --- 5. Create DataLoaders ---
    train_loader = DataLoader([data], batch_size=1, shuffle=False)
    val_loader = DataLoader([data], batch_size=1, shuffle=False)
    test_loader = DataLoader([data], batch_size=1, shuffle=False)

    return train_loader, val_loader, test_loader