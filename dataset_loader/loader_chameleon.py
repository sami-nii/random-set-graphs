from torch_geometric.utils import subgraph
import torch
from torch_geometric.loader import DataLoader
from .utils import one_hot_encode
from torch_geometric.datasets import WikipediaNetwork

def loader_chameleon(DATASET_STORAGE_PATH, config):
    """
    Loads the Chameleon dataset and prepares it for transductive OOD detection.

    This function prepares a single graph object and attaches all necessary masks to it.
    It returns DataLoaders that yield this same, single graph.

    Args:
        DATASET_STORAGE_PATH (str): Path to store the dataset.
        config (dict): A configuration dictionary (e.g., for batch_size).

    Returns:
        A tuple of (train_loader, val_loader, test_loader).
    """
    dataset = WikipediaNetwork(root=DATASET_STORAGE_PATH, name='chameleon')
    data = dataset[0]  # Get the single graph object

    # Define OOD and ID classes
    OODclass = [0, 1]
    IDclass = [2, 3, 4]
    num_id_classes = len(IDclass)
    
    # --- 1. Prepare Masks ---
    # The original masks from the dataset select nodes for each split # TODO use a seed for std
    original_train_mask = data.train_mask[:, 0]
    original_val_mask = data.val_mask[:, 0]
    original_test_mask = data.test_mask[:, 0]

    # Create a mask to identify all OOD nodes
    ood_node_mask = torch.isin(data.y, torch.tensor(OODclass))

    # The final train and val masks should ONLY include ID-class nodes
    data.train_mask = original_train_mask & ~ood_node_mask
    data.val_mask = original_val_mask & ~ood_node_mask
    
    # The test mask should include BOTH ID and OOD nodes for evaluation
    data.test_mask = original_test_mask
    
    print(f"Nodes for training (ID only): {data.train_mask.sum().item()}")
    print(f"Nodes for validation (ID only): {data.val_mask.sum().item()}")
    print(f"Nodes for testing (ID + OOD): {data.test_mask.sum().item()}")

    # --- 2. Prepare Labels (y tensor) ---
    # The model expects a one-hot vector for ID classes and a zero-vector for OOD classes.
    new_y = torch.zeros((data.num_nodes, num_id_classes), dtype=torch.float)
    
    # Find all nodes belonging to ID classes
    id_node_mask = ~ood_node_mask
    
    # Get the original labels for the ID nodes and remap them to the new range [0, 1, 2]
    original_id_labels = data.y[id_node_mask]
    remapped_id_labels = original_id_labels - min(IDclass)
    
    # One-hot encode the remapped labels and place them in the correct rows of the new_y tensor
    new_y[id_node_mask] = one_hot_encode(remapped_id_labels, num_id_classes)
    
    # Assign the new, correctly formatted y-tensor to the data object
    data.y = new_y

    # --- 3. Create DataLoaders ---
    # For single-graph transductive learning, the loader yields the same graph object each time.
    # The model then uses the appropriate mask internally.
    # batch_size is effectively 1, as we process the whole graph at once.
    train_loader = DataLoader([data], batch_size=1, shuffle=False)
    val_loader = DataLoader([data], batch_size=1, shuffle=False)
    test_loader = DataLoader([data], batch_size=1, shuffle=False)

    return train_loader, val_loader, test_loader