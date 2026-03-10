import torch
import inspect
import numpy as np
from sklearn.manifold import TSNE
from sklearn.mixture import GaussianMixture
from scipy import linalg

def train_embeddings(aux_model, x_train, batch_size, device='cpu', edge_index=None):
    aux_model.to(device)
    x_train = x_train.to(device)
    if edge_index is not None:
        edge_index = edge_index.to(device)
    
    with torch.no_grad():
        if edge_index is None:
            train_embeddings = aux_model(x_train).cpu().numpy()
        else:
            train_embeddings = aux_model(x_train, edge_index).cpu().numpy()
    
    tsne_kwargs = {"n_components": 3, "init": "random", "perplexity": 30}
    if "n_jobs" in inspect.signature(TSNE.__init__).parameters:
        tsne_kwargs["n_jobs"] = -1
    train_embedded_tsne = TSNE(**tsne_kwargs).fit_transform(train_embeddings)
    return torch.tensor(train_embedded_tsne, dtype=torch.float32)


def fit_gmm(classes, train_embedded_tsne, y_train):
    individual_gms = []
    train_embedded_tsne_np = train_embedded_tsne.detach().cpu().numpy()
    y_train_np = y_train.detach().cpu().numpy()
    
    for i in range(len(classes)):
        class_points = train_embedded_tsne_np[y_train_np == i]
        if class_points.shape[0] == 0:
            raise ValueError(f"No training samples found for class {i} in budgeting GMM fit.")
        gm = GaussianMixture(n_components=1, random_state=7)
        gm.fit(class_points)
        individual_gms.append(gm)
    
    return individual_gms


def ellipse(individual_gms, num_classes, device='cpu'):
    means = []
    eigen_vecs = []
    stds = []
    feature_space = 3
    
    for i_gm in individual_gms:
        means.append(i_gm.means_[0])
        v, w = linalg.eigh(i_gm.covariances_[0])
        v = 2.0 * torch.sqrt(torch.tensor(7.815)) * torch.sqrt(torch.tensor(v, dtype=torch.float32))
        stds.append(v)
        eigen_vecs.append(torch.tensor(w, dtype=torch.float32))
    
    means = torch.from_numpy(np.asarray(means, dtype=np.float32)).to(device)
    eigen_vecs = torch.stack(eigen_vecs).to(device)
    stds = torch.stack(stds).to(device)
    
    max_std = torch.max(stds)
    max_len = int(max_std.item()) + 2
    reg_shape = (max_len,) * feature_space
    center = (torch.tensor(reg_shape, device=device, dtype=torch.float32) / 2.0)
    
    # Generate grid indices
    indices = torch.stack(torch.meshgrid(*[torch.arange(s, device=device) for s in reg_shape], indexing='ij'), dim=-1)
    indices = indices.reshape(-1, feature_space).to(torch.float32)
    
    regions = []
    vecs = indices - center
    vec_norms = torch.norm(vecs, dim=-1, keepdim=True) + 1e-31
    
    for i in range(num_classes):
        ell = torch.matmul(vecs, eigen_vecs[i])
        ell = torch.abs(ell / (vec_norms * torch.norm(eigen_vecs[i], dim=-1)))
        ell = torch.norm(torch.sum((ell * (stds[i]/2)).unsqueeze(-1) * eigen_vecs[i], dim=1), dim=-1) + 1e-25
        ell = (vec_norms.squeeze(-1) <= ell).float().reshape(reg_shape)
        regions.append(ell)
    
    return regions, means, max_len


def overlaps(k, classes, num_clusters, classes_dict, regions, means, max_len):
    clusters = classes
    overlaps_dict = {}
    top_sets = [set([c]) for c in clusters]

    for cardinality in range(2, num_clusters + 1):
        new_top_sets = []
        for ts in top_sets:
            for clus in clusters:
                s = sorted(ts | {clus})
                s_key = ",".join(s)
                if len(s) == cardinality and s_key not in overlaps_dict:
                    region = torch.zeros_like(regions[0])
                    smallest_region = float('inf')
                    for num, name in enumerate(s):
                        c = classes_dict[name]
                        reg = regions[c]
                        if num == 0:
                            region += reg
                            reg_cen = means[c]
                        else:
                            top_corner = means[c] - reg_cen
                            limits = torch.stack([torch.clamp(top_corner, -max_len, max_len).int() for _ in range(2)], dim=1)
                            # simplified approximation for overlap
                            region += reg
                        smallest_region = min(smallest_region, torch.sum(reg))
                    
                    intersection = torch.sum(region == len(s)).item()
                    union = torch.sum(region != 0).item()
                    op = 0.0 if union == 0 else intersection / union
                    overlaps_dict[s_key] = op
                    if op > 0:
                        new_top_sets.append(set(s))
        top_sets = new_top_sets
    
    keys = list(overlaps_dict.keys())
    keys = [set(cl.split(",")) for cl in keys]
    values = list(overlaps_dict.values())
    arg_sorted = torch.argsort(torch.tensor(values), descending=True)
    new_k = min(k, torch.sum(torch.tensor(values)[arg_sorted[:k]] != 0).item())
    new_classes = [set([c]) for c in classes] + [keys[i] for i in arg_sorted[:new_k]]
    
    return new_classes
