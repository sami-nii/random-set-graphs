from torch_geometric.datasets import Planetoid
import torch_geometric
from torch_geometric.loader import DataLoader
from .utils import one_hot_encode
import torch
from torch_geometric.data import Data

# Load the Planetoid dataset (Cora)
def loader_cora(DATASET_STORAGE_PATH, config):
    """
    Loads the Cora dataset and prepares it for transductive OOD detection
    using the correct MASKING approach, including OOD samples in the validation set.

    Args:
        DATASET_STORAGE_PATH (str): Path to store the Planetoid dataset.
        config (dict): A configuration dictionary (batch_size is ignored for this loader).

    Returns:
        A tuple of (train_loader, val_loader, test_loader).
    """
    # --- 1. Load the Full Graph Data ---
    # Using split="full" gives us the standard public masks for Cora
    dataset = Planetoid(root=DATASET_STORAGE_PATH, name="Cora", split="full")
    data = dataset[0]  # Get the single graph object

    # --- 2. Prepare Masks based on specified ID/OOD classes ---
    OODclass = [0, 1, 2, 3]
    IDclass = [4, 5, 6]
    num_id_classes = len(IDclass)
    
    # Get the original masks provided by the dataset
    original_train_mask = data.train_mask
    original_val_mask = data.val_mask
    original_test_mask = data.test_mask

    # Create a mask to identify all OOD nodes in the entire graph
    ood_node_mask = torch.isin(data.y, torch.tensor(OODclass))
    id_node_mask = ~ood_node_mask

    # --- 3. Create the Final Masks for the Data Object ---

    # The final train mask must ONLY include ID-class nodes from the original split
    data.train_mask = original_train_mask & id_node_mask
    
    # The validation mask should include ALL nodes (ID and OOD) from the original validation split
    data.val_mask = original_val_mask
    
    # The test mask should include ALL nodes (ID and OOD) from the original test split
    data.test_mask = original_test_mask

    # --- Reporting for verification ---
    print("--- Cora Dataset with OOD Validation ---")
    id_val_nodes = data.val_mask & id_node_mask
    ood_val_nodes = data.val_mask & ood_node_mask
    id_test_nodes = data.test_mask & id_node_mask
    ood_test_nodes = data.test_mask & ood_node_mask
    
    print(f"Nodes for training (ID only): {data.train_mask.sum().item()}")
    print(f"Nodes for validation (ID+OOD): {data.val_mask.sum().item()} -> {id_val_nodes.sum()} ID, {ood_val_nodes.sum()} OOD")
    print(f"Nodes for testing (ID+OOD): {data.test_mask.sum().item()} -> {id_test_nodes.sum()} ID, {ood_test_nodes.sum()} OOD")

    # --- 4. Prepare the Unified Label Tensor (y) ---
    # The model expects a one-hot vector for ID classes and a zero-vector for OOD classes.
    new_y = torch.zeros((data.num_nodes, num_id_classes), dtype=torch.float)
    
    # Get original labels for all ID nodes and remap them to the range [0, 2]
    original_id_labels = data.y[id_node_mask]
    remapped_id_labels = original_id_labels - min(IDclass)
    
    # One-hot encode and place them in the correct rows of the new_y tensor
    new_y[id_node_mask] = one_hot_encode(remapped_id_labels, num_id_classes)
    
    # Replace the original `y` on the data object
    data.y = new_y
    
    # --- 5. Create DataLoaders ---
    # For this transductive setup, all loaders yield the same single graph object.
    train_loader = DataLoader([data], batch_size=1, shuffle=False)
    val_loader = DataLoader([data], batch_size=1, shuffle=False)
    test_loader = DataLoader([data], batch_size=1, shuffle=False)

    return train_loader, val_loader, test_loader