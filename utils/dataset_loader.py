import torch_geometric.data
from torch_geometric.datasets import Planetoid
import os
from torch_geometric.loader import DataLoader
import torch_geometric
import torch
import torch.nn.functional as F


DATASET_STORAGE_PATH = "../dataset/"


def one_hot_encode(labels, num_classes):
    """One-hot encode a tensor of labels."""
    return F.one_hot(labels, num_classes=num_classes).float()

def dataset_loader(dataset_name: str):
    if dataset_name == 'cora':
        # Load the Planetoid dataset (Cora)
        dataset = Planetoid(root=DATASET_STORAGE_PATH, name="Cora", split="full")
        dataset : torch_geometric.data.Data = dataset[0]  # there is only one graph in the dataset

        # Define OOD and ID classes
        OODclass = [0, 1, 2, 3]
        IDclass = [4, 5, 6]

        # Clone masks
        train_mask = dataset.train_mask.clone()
        val_mask = dataset.val_mask.clone()
        test_mask = dataset.test_mask.clone()

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

        return train_data, val_data, test_data

    else:
        raise ValueError(f"Dataset {dataset_name} not supported")
