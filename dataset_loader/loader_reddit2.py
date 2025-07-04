import torch
import torch.nn.functional as F
import json
import os.path as osp
from typing import Callable, List, Optional
import numpy as np

import torch_geometric
from torch_geometric.data import Data, InMemoryDataset, download_google_url
from torch_geometric.loader import DataLoader

import os
import os.path as osp
import ssl
import sys
import urllib
from typing import Optional

import fsspec

from torch_geometric.io import fs


##### CODE FROM THE PAPER: Revisiting Score Propagation in Graph Out-of-Distribution Detection 
# Longfei Ma, Yiyou Sun, Kaize Ding, Zemin Liu, Fei Wu 
# 38th Conference on Neural Information Processing Systems (NeurIPS 2024)

def download_url(
    url: str,
    folder: str,
    log: bool = True,
    filename: Optional[str] = None,
):
    r"""Downloads the content of an URL to a specific folder.

    Args:
        url (str): The URL.
        folder (str): The folder.
        log (bool, optional): If :obj:`False`, will not print anything to the
            console. (default: :obj:`True`)
        filename (str, optional): The filename of the downloaded file. If set
            to :obj:`None`, will correspond to the filename given by the URL.
            (default: :obj:`None`)
    """
    if filename is None:
        filename = url.rpartition('/')[2]
        filename = filename if filename[0] == '?' else filename.split('?')[0]

    path = osp.join(folder, filename)

    if fs.exists(path):  # pragma: no cover
        if log and 'pytest' not in sys.modules:
            print(f'Using existing file {filename}', file=sys.stderr)
        return path

    if log and 'pytest' not in sys.modules:
        print(f'Downloading {url}', file=sys.stderr)

    os.makedirs(folder, exist_ok=True)

    context = ssl._create_unverified_context()
    data = urllib.request.urlopen(url, context=context)

    with fsspec.open(path, 'wb') as f:
        # workaround for https://bugs.python.org/issue42853
        while True:
            chunk = data.read(10 * 1024 * 1024)
            if not chunk:
                break
            f.write(chunk)

    return path


def download_google_url(
    id: str,
    folder: str,
    filename: str,
    log: bool = True,
):
    r"""Downloads the content of a Google Drive ID to a specific folder."""
    url = f'https://drive.usercontent.google.com/download?id={id}&confirm=t'
    return download_url(url, folder, log, filename)

class Reddit2(InMemoryDataset):
    adj_full_id = '1sncK996BM5lpuDf75lDFqCiDZyErc1c2'
    feats_id = '1ZsHaJ0ussP1W722krmEIp_8pwKAoi5b3'
    class_map_id = '1JF3Pjv9OboMNYs2aXRQGbJbc4t_nDd5u'
    role_id = '1nJIKd77lcAGU4j-kVNx_AIGEkveIKz3A'

    def __init__(
        self, root: str, transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None, force_reload: bool = False,
    ) -> None:
        super().__init__(root, transform, pre_transform, force_reload=force_reload)
        self.load(self.processed_paths[0])

    @property
    def raw_file_names(self) -> List[str]:
        return ['adj_full.npz', 'feats.npy', 'class_map.json', 'role.json']

    @property
    def processed_file_names(self) -> str:
        return 'data.pt'

    def download(self) -> None:
        download_google_url(self.adj_full_id, self.raw_dir, 'adj_full.npz')
        download_google_url(self.feats_id, self.raw_dir, 'feats.npy')
        download_google_url(self.class_map_id, self.raw_dir, 'class_map.json')
        download_google_url(self.role_id, self.raw_dir, 'role.json')

    def process(self) -> None:
        import scipy.sparse as sp
        f = np.load(osp.join(self.raw_dir, 'adj_full.npz'))
        adj = sp.csr_matrix((f['data'], f['indices'], f['indptr']), f['shape'])
        adj = adj.tocoo()
        row = torch.from_numpy(adj.row).to(torch.long)
        col = torch.from_numpy(adj.col).to(torch.long)
        edge_index = torch.stack([row, col], dim=0)
        x = torch.from_numpy(np.load(osp.join(self.raw_dir, 'feats.npy'))).to(torch.float)
        ys = [-1] * x.size(0)
        with open(osp.join(self.raw_dir, 'class_map.json')) as f:
            class_map = json.load(f)
            for key, item in class_map.items():
                ys[int(key)] = item
        y = torch.tensor(ys)
        with open(osp.join(self.raw_dir, 'role.json')) as f:
            role = json.load(f)
        train_mask = torch.zeros(x.size(0), dtype=torch.bool)
        train_mask[torch.tensor(role['tr'])] = True
        val_mask = torch.zeros(x.size(0), dtype=torch.bool)
        val_mask[torch.tensor(role['va'])] = True
        test_mask = torch.zeros(x.size(0), dtype=torch.bool)
        test_mask[torch.tensor(role['te'])] = True
        data = Data(x=x, edge_index=edge_index, y=y, train_mask=train_mask,
                    val_mask=val_mask, test_mask=test_mask)
        data = data if self.pre_transform is None else self.pre_transform(data)
        self.save([data], self.processed_paths[0])



##### END OF THE CODE FROM THE PAPER: Revisiting Score Propagation in Graph Out-of-Distribution Detection 


def one_hot_encode(labels, num_classes):
    """Simple one-hot encoder."""
    return F.one_hot(labels, num_classes=num_classes).float()


def load_reddit2(DATASET_STORAGE_PATH, config):
    """
    Loads the Reddit2 dataset and prepares it for transductive OOD detection.
    """
    # --- 1. Load the Full Graph Data ---
    dataset = Reddit2(root=osp.join(DATASET_STORAGE_PATH, 'reddit2'))
    data = dataset[0]

    # --- 2. Prepare Masks based on CORRECTED ID/OOD classes ---
    # As per your request:
    OODclass = list(range(11))         # Classes 0 to 10 are OOD
    IDclass = list(range(11, 41))      # Classes 11 to 40 are ID
    num_id_classes = len(IDclass)
    
    print("--- Reddit2 Dataset Corrected Split ---")
    print(f"OOD Classes: {OODclass[0]}...{OODclass[-1]}")
    print(f"ID Classes: {IDclass[0]}...{IDclass[-1]}")
    
    original_train_mask = data.train_mask
    original_val_mask = data.val_mask
    original_test_mask = data.test_mask

    ood_node_mask = torch.isin(data.y, torch.tensor(OODclass))

    data.train_mask = original_train_mask & ~ood_node_mask
    data.val_mask = original_val_mask & ~ood_node_mask
    data.test_mask = original_test_mask

    print(f"Nodes for training (ID only): {data.train_mask.sum().item()}")
    print(f"Nodes for validation (ID only): {data.val_mask.sum().item()}")
    print(f"Nodes for testing (ID + OOD): {data.test_mask.sum().item()}")

    # --- 3. Prepare the Unified Label Tensor (y) ---
    new_y = torch.zeros((data.num_nodes, num_id_classes), dtype=torch.float)
    id_node_mask = ~ood_node_mask
    
    original_id_labels = data.y[id_node_mask]
    # This correctly remaps labels [11, ..., 40] to [0, ..., 29]
    remapped_id_labels = original_id_labels - min(IDclass)
    
    new_y[id_node_mask] = one_hot_encode(remapped_id_labels, num_id_classes)
    data.y = new_y
    
    # --- 4. Create DataLoaders ---
    train_loader = DataLoader([data], batch_size=1, shuffle=False)
    val_loader = DataLoader([data], batch_size=1, shuffle=False)
    test_loader = DataLoader([data], batch_size=1, shuffle=False)

    return train_loader, val_loader, test_loader