from torch_geometric.utils import subgraph
from ogb.nodeproppred import NodePropPredDataset
import torch_geometric
import torch
from torch_geometric.loader import DataLoader
from .utils import one_hot_encode, even_quantile_labels
import scipy


def loader_snap_patents(DATASET_STORAGE_PATH, config):
    
    nclass = 5
    train_ratio = 0.6
    val_ratio = 0.2

    OODclass = [0, 1]
    IDclass = [2, 3, 4]

    fulldata = scipy.io.loadmat(f'{DATASET_STORAGE_PATH}/snap-patents.mat')

    edge_index = torch.tensor(fulldata['edge_index'], dtype=torch.long)
    node_feat = torch.tensor(fulldata['node_feat'].todense(), dtype=torch.float)

    years = fulldata['years'].flatten()
    label = even_quantile_labels(years, nclass, verbose=False)
    label = torch.tensor(label, dtype=torch.long)

    data = torch_geometric.data.Data(
        edge_index=edge_index,
        x=node_feat,
        y=label
    )

    num_nodes = data.num_nodes

    # Step 1: Generate random permutation of nodes
    indices = torch.randperm(num_nodes)
    train_size = int(train_ratio * num_nodes)
    val_size = int(val_ratio * num_nodes)

    # Step 2: Split indices
    train_idx = indices[:train_size]
    val_idx = indices[train_size:train_size + val_size]
    test_idx = indices[train_size + val_size:]

    # Step 3: Filter out OOD nodes from training and validation sets
    train_idx = train_idx[~torch.isin(data.y[train_idx], torch.tensor(OODclass))]
    val_idx = val_idx[~torch.isin(data.y[val_idx], torch.tensor(OODclass))]

    # Step 4: Create subgraphs based on the filtered splits
    train_edge_index, train_edge_attr = subgraph(train_idx, data.edge_index, data.edge_attr, relabel_nodes=True)
    val_edge_index, val_edge_attr = subgraph(val_idx, data.edge_index, data.edge_attr, relabel_nodes=True)
    test_edge_index, test_edge_attr = subgraph(test_idx, data.edge_index, data.edge_attr, relabel_nodes=True)

    # Step 5: Create independent Data objects
    train_data = torch_geometric.data.Data(
        x=data.x[train_idx],
        edge_index=train_edge_index,
        edge_attr=train_edge_attr,
        y=one_hot_encode( data.y[train_idx] - min(IDclass), len(IDclass) )
    )

    val_data = torch_geometric.data.Data(
        x=data.x[val_idx],
        edge_index=val_edge_index,
        edge_attr=val_edge_attr,
        y= one_hot_encode(data.y[val_idx]- min(IDclass), len(IDclass) )
    )

    test_data = torch_geometric.data.Data(
        x=data.x[test_idx],
        edge_index=test_edge_index,
        edge_attr=test_edge_attr,
        y=data.y[test_idx]
    )

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

    train_loader = DataLoader([train_data], batch_size=config["batch_size"], shuffle=True)
    val_loader = DataLoader([val_data], batch_size=config["batch_size"], shuffle=False)
    test_loader = DataLoader([test_data], batch_size=config["batch_size"], shuffle=False)

    return train_loader, val_loader, test_loader
