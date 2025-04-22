from torch_geometric.utils import subgraph
import torch
from torch_geometric.loader import DataLoader
from .utils import one_hot_encode
from torch_geometric.datasets import WikipediaNetwork
from torch_geometric.data import Data


def laoder_squirrel(DATASET_STORAGE_PATH, config, split_test):
    dataset = WikipediaNetwork(root=DATASET_STORAGE_PATH, name='squirrel')
    dataset = dataset[0]  # there is only one graph in the dataset

    # Define OOD and ID classes
    OODclass = [0, 1]
    IDclass = [2, 3, 4]
    
    train_mask = dataset.train_mask[:, 0].clone()
    val_mask = dataset.val_mask[:, 0].clone()
    test_mask = dataset.test_mask[:, 0].clone()

    # Mask for OOD nodes
    ood_mask = torch.isin(dataset.y, torch.tensor(OODclass))

    # Update masks to filter out OOD nodes for training and validation
    train_mask = train_mask & ~ood_mask
    val_mask = val_mask & ~ood_mask

    # Create subgraphs based on filtered masks
    train_data = dataset.subgraph(train_mask)
    val_data = dataset.subgraph(val_mask)
    test_data = dataset.subgraph(test_mask)

    # One-hot encode labels for training and validation data (only ID classes)
    train_data.y = one_hot_encode(train_data.y - min(IDclass), len(IDclass))
    val_data.y = one_hot_encode(val_data.y - min(IDclass), len(IDclass))

    # Encode labels for test data: one-hot for ID classes, all zero for OOD classes
    
    is_ood_test = torch.isin(test_data.y, torch.tensor(OODclass))

    # Initialize test labels to zeros
    test_labels_encoded = torch.zeros((test_data.y.size(0), len(IDclass)))

    # Encode ID class nodes in test data
    id_test_indices = (~is_ood_test).nonzero(as_tuple=True)[0]
    test_labels_encoded[id_test_indices] = one_hot_encode(
        test_data.y[id_test_indices] - min(IDclass), len(IDclass)
    )

    test_data.y = test_labels_encoded

    train_data.validate()
    val_data.validate()
    test_data.validate()

    train_loader = DataLoader([train_data], batch_size=config["batch_size"], shuffle=True)
    val_loader = DataLoader([val_data], batch_size=config["batch_size"], shuffle=False)
    test_loader = DataLoader([test_data], batch_size=config["batch_size"], shuffle=False)

    if split_test:
        num_nodes = test_data.num_nodes
        all_indices = torch.randperm(num_nodes)
        half = num_nodes // 2

        idx_1 = all_indices[:half]
        idx_2 = all_indices[half:]

        # Get subgraphs for each half
        edge_index_1, _ = subgraph(idx_1, test_data.edge_index, relabel_nodes=True)
        edge_index_2, _ = subgraph(idx_2, test_data.edge_index, relabel_nodes=True)

        data_1 = Data(
            x=test_data.x[idx_1],
            edge_index=edge_index_1,
            y=test_data.y[idx_1]
        )

        data_2 = Data(
            x=test_data.x[idx_2],
            edge_index=edge_index_2,
            y=test_data.y[idx_2]
        )

        OOD_train_loader = DataLoader([data_1], batch_size=1, shuffle=False),
        OOD_test_loader = DataLoader([data_2], batch_size=1, shuffle=False)
        
        return train_loader, val_loader, OOD_train_loader, OOD_test_loader
    else:
        # Return the loaders for training, validation, and test
        return train_loader, val_loader, test_loader

