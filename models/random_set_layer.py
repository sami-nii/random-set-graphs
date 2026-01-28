import torch
import lightning as L
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics import Accuracy, F1Score
from torch_geometric.nn.models import GCN, GraphSAGE, GAT, GIN, EdgeCNN
from utils.math import compute_uncertainties

models_map = {
    "GCN": GCN, "SAGE": GraphSAGE, "GAT": GAT, "GIN": GIN, "EdgeCNN": EdgeCNN
}
# Random set layer that outputs a valid mass function.
# Include at least all singletons

class RandomSetLayer(nn.Module):
    def __init__(self, input_dim, focal_sets):
        super().__init__()
        self.focal_sets = focal_sets
        self.num_sets = len(focal_sets)
        self.linear = nn.Linear(input_dim, self.num_sets)
        # Replace final layer with length of new classes

    def forward(self, x):
        scores = torch.sigmoid(self.linear(x))
        return scores / (scores.sum(dim=-1, keepdim=True) + 1e-8)
    
new_classes = focal_sets
# Classes = Classes from squirrel dataset

    
# def pignistic(m, focal_sets, num_classes):
#     device = m.device
#     N = m.size(0)
#     betp = torch.zeros(N, num_classes, device=device)
#     for i, A in enumerate(focal_sets):
#         if len(A) == 0:
#             continue
#         weight = m[:, i] / len(A)
#         for c in A:
#             betp[:, c] += weight
#     return betp

def mass_coeff(new_classes):
    for i, A in enumerate(new_classes):
        for j, B in enumerate(new_classes):
            leng = 0
            if set(B).issubset(set(A)):
                leng = (-1) ** (len(A) - len(B))
            mass_co[j][i] = leng
    return mass_co

mass_co = np.zeros((len(new_classes), len(new_classes)))
    # mass_co = np.zeros((len(new_classes_1024), len(new_classes_1024)))

mass_coeff_matrix = mass_coeff(new_classes)
mass_coeff_matrix = tf.cast(mass_coeff_matrix, tf.float32)

#Mobius inverse function
def belief_to_mass(test_preds, new_classes):
    mass_coeff_matrix = mass_coeff(new_classes)
    
    test_preds_mass = test_preds @ mass_coeff_matrix

    test_preds_mass[test_preds_mass<0] = 0
    sums_ = 1 - np.sum(test_preds_mass, axis=-1)
    sums_[sums_<0] = 0

    test_preds_mass = np.append(test_preds_mass, sums_[:, None], axis=-1)
    test_preds_mass = test_preds_mass/np.sum(test_preds_mass, axis=-1)[:, None]
    
    return test_preds_mass

def betp_approx(classes):
  mask = []
  for j, n in enumerate(classes):
      mask.append([])
      for k, A in enumerate(new_classes):
          if set([n]).issubset(A):
              mask[-1].append( 1 / len(A))
          else:
              mask[-1].append(0)

  mask = np.array(mask).T
  return mask

# Pignistic probability
def pignistic(mass, classes, new_classes):
    betp_matrix = np.zeros((len(new_classes), len(classes)))
    for i,c in enumerate(classes): 
        for j,A in enumerate(new_classes):
            if set([c]).issubset(A):
                betp_matrix[j,i] = 1/len(A)
    
    final_bet_p = mass @ betp_matrix

    return final_bet_p

class BinaryCrossEntropy(nn.Module):
    ALPHA = 0.01
    BETA = 0.01

    def computeBCE(y_true, y_pred):

        y_true = tf.cast(y_true, tf.float32)
        y_true = K.clip(y_true, K.epsilon(), 1)
        y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
        term_0 = (1 - y_true) * K.log(1 - y_pred + K.epsilon())  
        term_1 = y_true * K.log(y_pred + K.epsilon())
        bce_loss = -K.mean(term_0 + term_1, axis=0)
        
        mass = tf.matmul(y_pred, mass_coeff_matrix)

        # alpha = tf.cast(tf.where(mass >= 0, tf.ones_like(mass), tf.zeros_like(mass)), dtype=tf.float32)
        
        mass_reg = K.mean(tf.nn.relu(-mass))

        mass_sum = tf.nn.relu(K.mean(K.sum(mass, axis=-1)) - 1)
        
        #add alpha to bce term 1 or 2
        # alpha_reg = -K.mean((1 - alpha) * K.log(1 - y_true + K.epsilon()) + alpha * K.log(y_true + K.epsilon()), axis = 0)
        

        total_loss = bce_loss + ALPHA * mass_reg + BETA * mass_sum
        # tf.print(K.mean(bce_loss), K.sum(mass_reg), K.mean(total_loss))

        return total_loss
    
class RandomSetGNN(L.LightningModule):
    def __init__(self, gnn_type, in_channels, hidden_channels, num_layers, focal_sets, num_classes, lr=0.001, weight_decay=0.0, act=F.relu):
        super().__init__()
        self.save_hyperparameters(ignore=["focal_sets"])

        self.gnn_model = models_map[gnn_type](
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            num_layers=num_layers,
            out_channels=hidden_channels,
            act=act,
        )
        self.random_set_layer = RandomSetLayer(input_dim=hidden_channels, focal_sets=focal_sets)
        self.criterion = BinaryCrossEntropy(focal_sets=focal_sets, num_classes=num_classes)
        self.acc = Accuracy(task="multiclass", num_classes=num_classes)
        self.f1 = F1Score(task="multiclass", num_classes=num_classes)
        self.lr = lr
        self.weight_decay = weight_decay
        self.focal_sets = focal_sets
        self.num_classes = num_classes

    def forward(self, data):
        h = self.gnn_model(data.x, data.edge_index)
        m = self.random_set_layer(h)
        return m
                
    def training_step(self, batch, batch_idx):
        q = self(batch)
        
        y_train = batch.y[batch.train_mask]
        q_train = q[batch.train_mask]

        loss = self.criterion(q_train, y_train) 
        # betp = pignistic(m_train, self.focal_sets, self.num_classes)
        # preds = betp.argmax(dim=1)

        # Return belief = model prediction instead of pignistic

        # Convert labels to indices
        train_preds = torch.argmax(q_train, dim=1)
        train_labels = torch.argmax(y_train, dim=1)
        
        accuracy = (train_preds == train_labels).float().mean()
        
        f1 = self.f1_score_metric(train_preds, train_labels)
        
        num_train_nodes = batch.train_mask.sum()
        self.log("train_loss", loss, batch_size=num_train_nodes)
        self.log("train_acc", accuracy, batch_size=num_train_nodes)
        self.log("train_f1", f1, batch_size=num_train_nodes)
        return loss
    
    def validation_step(self, batch, batch_idx):
        q, = self(batch)

        # Select the outputs and labels for the validation nodes
        q_val = q[batch.val_mask].detach()
        y_val = batch.y[batch.val_mask].detach()
        
        # --- OOD Detection Metrics ---
        # if self.ood_in_val:
        #     TU, AU, EU = compute_uncertainties(q_val.cpu().numpy())
        #     targets = 1 - y_val.sum(axis=1) # 1 for OOD, 0 for ID
            
        #     auroc_EU = self.auroc_metric(torch.from_numpy(EU).to(self.device), targets)
        #     auroc_AU = self.auroc_metric(torch.from_numpy(AU).to(self.device), targets)
        #     auroc_TU = self.auroc_metric(torch.from_numpy(TU).to(self.device), targets)
            
        #     self.log("val_auroc_EU", auroc_EU, prog_bar=True) # Monitor this during training
        #     self.log("val_auroc_AU", auroc_AU)
        #     self.log("val_auroc_TU", auroc_TU)
        
        # --- Classification Metrics (for ID nodes within the validation set) ---
        id_mask_in_val = (y_val.sum(axis=1) == 1)
        
        # Calculate validation loss ONLY on ID nodes
        loss = self.criterion(q_val[id_mask_in_val], y_val[id_mask_in_val])
        self.log("val_loss", loss, prog_bar=True)

        # Calculate classification metrics
        val_labels = torch.argmax(y_val[id_mask_in_val], dim=1)
        
        val_preds = torch.argmax(q_val[id_mask_in_val], dim=1)

        val_f1 = self.f1_score_metric(val_preds, val_labels)
        
        self.log("val_f1", val_f1, prog_bar=True)


        return loss
    
    def test_step(self, batch, batch_idx):
        m = self(batch)
        y_test = batch.y
        test_preds_mass = tf.matmul(m, mass_coeff_matrix)
        test_preds_mass = np.array(test_preds_mass)
        test_preds_mass[test_preds_mass<0] = 0

        betp = test_preds_mass @ betp_approx(classes)

        # betp = pignistic(m, self.focal_sets, self.num_classes)
        preds = betp.argmax(dim=1)

        labels_idx = y_test.argmax(dim=1) if y_test.ndim > 1 else y_test

        acc = self.acc(preds, labels_idx)
        f1 = self.f1(preds, labels_idx)

        self.log("test_acc", acc)
        self.log("test_f1", f1)

        return {"test_acc": acc, "test_f1": f1}
    
    def configure_optimizers(self):
        return torch.optim.Adam(
        self.parameters(), lr=self.lr, weight_decay=self.weight_decay)