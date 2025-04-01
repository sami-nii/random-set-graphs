from torch_geometric.datasets import Planetoid
import os
from torch_geometric.loader import DataLoader
import torch_geometric



DATASET_STORAGE_PATH = "../dataset/"


def dataset_loader(dataset_name: str):
    if dataset_name == 'cora':
        batch_size = 32  # Puoi personalizzare il batch size

        # Load the Planetoid dataset (Cora)
        dataset = Planetoid(root=DATASET_STORAGE_PATH, name="Cora", split="full")[0]

        # Crea DataLoader direttamente dal dataset senza incapsularlo
        train_mask = dataset.train_mask
        val_mask = dataset.val_mask
        test_mask = dataset.test_mask

        train_data = dataset.subgraph(train_mask)
        val_data = dataset.subgraph(val_mask)
        test_data = dataset.subgraph(test_mask)

        train_loader = DataLoader([train_data], batch_size=batch_size, shuffle=False)
        val_loader = DataLoader([val_data], batch_size=batch_size, shuffle=False)
        test_loader = DataLoader([test_data], batch_size=batch_size, shuffle=False)

        return train_loader, val_loader, test_loader
    else:
        raise ValueError(f"Dataset {dataset_name} not supported")
