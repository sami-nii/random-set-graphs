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

        # Clone the train and validation masks to avoid altering the original data
        train_mask = dataset.train_mask.clone()
        val_mask = dataset.val_mask.clone()
        test_mask = dataset.test_mask.clone()

        # Mask for OOD nodes
        ood_mask = torch.isin(dataset.y, torch.tensor(OODclass))

        # Update train and validation masks to filter out OOD nodes
        train_mask = train_mask & ~ood_mask
        val_mask = val_mask & ~ood_mask

        # Create subgraphs based on filtered masks
        train_data = dataset.subgraph(train_mask)
        val_data = dataset.subgraph(val_mask)
        test_data = dataset.subgraph(test_mask)

        # The labels should be one-hot encoded

        # onlt ID classes during training and validation
        train_data.y = one_hot_encode(train_data.y - min(IDclass), len(IDclass))
        val_data.y = one_hot_encode(val_data.y - min(IDclass), len(IDclass))
           
        # Where the test data is OOD, we will set the label to 7
        test_labels_ood = torch.isin(test_data.y, torch.tensor(OODclass))
        test_labels = torch.where(test_labels_ood, torch.tensor(7), test_data.y)
        
        # Create a one-hot encoding for ID classes
        test_data.y = one_hot_encode(test_labels - min(IDclass), len(IDclass) + 1)  # +1 for the OOD class

        train_data.validate()
        val_data.validate()
        test_data.validate()

        return train_data, val_data, test_data
    
    else:
        raise ValueError(f"Dataset {dataset_name} not supported")
    
 