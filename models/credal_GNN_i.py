import torch
import lightning as L
from torch_geometric.nn.models import GCN, GraphSAGE, GAT, GIN, EdgeCNN
import torch.nn as nn
import torch.nn.functional as F
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.math import compute_uncertainties
from ood_metrics import fpr_at_95_tpr
from torchmetrics import AUROC
from models.credal_layer import CredalLayer
from models.credal_loss import CreNetLoss

models_map = {
    "GCN": GCN,  # GCN is built-in in torch_geometric
    "SAGE": GraphSAGE,  # SAGE is built-in in torch_geometric
    "GAT": GAT,  # GAT is built-in in torch_geometric
    "GIN": GIN,  # GIN is built-in in torch_geometric
    "EdgeCNN": EdgeCNN,  # EdgeCNN is built-in in torch_geometric
}

