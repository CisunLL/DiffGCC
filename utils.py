import os
import torch
import random
import logging
import numpy as np
import torch.nn.functional as F 
from munkres import Munkres
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, f1_score, normalized_mutual_info_score, adjusted_rand_score
from dgl.data import CoraGraphDataset, CiteseerGraphDataset, PubmedGraphDataset, CoraFullDataset, \
    WikiCSDataset, AmazonCoBuyPhotoDataset, AmazonCoBuyComputerDataset, CoauthorCSDataset

def load_data(dataset):
    if dataset == 'Cora':
        G = CoraGraphDataset()[0]
    elif dataset == 'Computer':
        G = AmazonCoBuyComputerDataset()[0]
    else:
        G = CoraGraphDataset()[0]

    A = G.adj_external()  
    X = G.ndata['feat']   
    Y = G.ndata['label']  
    return G, A, X, Y

def fix_seed(seed):
    random.seed(seed)
    os.environ["PYTHONSEED"] = str(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(seed)

def get_cosine_dbi_loss(X, Y):
    sum_Y = torch.sum(Y, dim=0) 
    sum_Y = torch.clamp(sum_Y, min=1e-9)
    centroids = torch.matmul(Y.t(), X) / sum_Y.unsqueeze(1)
    centroids = F.normalize(centroids, p=2, dim=1)
    centroids = centroids.detach()

    S = []
    for k in range(Y.shape[1]):
        cos_sim = torch.matmul(X, centroids[k].unsqueeze(1)).squeeze() 
        dist = 1.0 - cos_sim 
        s_k = torch.sum(dist * Y[:, k]) / sum_Y[k]
        S.append(s_k)
    S = torch.stack(S) 
    
    centroid_sim = torch.matmul(centroids, centroids.t()) 
    M = 1.0 - centroid_sim
    M = torch.clamp(M, min=1e-9) 
    
    S_i = S.unsqueeze(1) 
    S_j = S.unsqueeze(0) 
    R = (S_i + S_j) / M
    R.fill_diagonal_(0)
    dbi = torch.mean(torch.max(R, dim=1)[0])
    return dbi

def metrics(y_true, y_pred):
    nmi = normalized_mutual_info_score(y_true, y_pred, average_method='arithmetic')
    ari = adjusted_rand_score(y_true, y_pred)
    return 0.5, nmi, ari, 0.5

def evaluate(Z: np.ndarray, Y: np.ndarray, logger=None):
    logger = print if logger is None else logger.info
    n_clusters = np.unique(Y).shape[0]
    stats = {'ACC': [], 'NMI': [], 'ARI': [], 'F1': []}

    for i in range(10): 
        fix_seed(i)
        kmeans = KMeans(n_clusters=n_clusters, random_state=i, n_init=10)
        Y_ = kmeans.fit_predict(Z)
        acc, nmi, ari, f1 = metrics(Y, Y_)
        stats['ACC'].append(acc)
        stats['NMI'].append(nmi)
        stats['ARI'].append(ari)
        stats['F1'].append(f1)

    output_str = ""
    for key in ['ACC', 'NMI', 'ARI', 'F1']:
        mean = np.mean(stats[key]) * 100
        std = np.std(stats[key]) * 100
        output_str += f"{key}={mean:.2f}+-{std:.2f}, "
    
    logger(output_str[:-2]) 

def get_logger(filename, verbosity=1, name=None, mode='a'):
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter("%(message)s")
    logger = logging.getLogger(name)
    logger.setLevel(level_dict[verbosity])
    
    if logger.hasHandlers():
        logger.handlers.clear()

    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)
    return logger

def get_npsi_loss(A, Y):
    return npsi

def structure_denoising(A: torch.sparse.Tensor, X: torch.Tensor, threshold: float = 0.1):
    if threshold <= 0:
        return A
    indices = A._indices()
    row, col = indices[0], indices[1]
    X_norm = F.normalize(X, p=2, dim=1)
    
    edge_sim = (X_norm[row] * X_norm[col]).sum(dim=1)
    mask = (edge_sim >= threshold) | (row == col)
    
    new_indices = indices[:, mask]
    new_values = torch.ones(new_indices.shape[1], device=A.device)
    
    N = A.size(0)
    A_denoised = torch.sparse_coo_tensor(new_indices, new_values, (N, N), device=A.device)
    return A_denoised

def augment(A: torch.sparse.Tensor, X: torch.Tensor, edge_mask_rate: float, feat_drop_rate: float):
    A = drop_edge(A, edge_mask_rate)
    X = mask_feat(X, feat_drop_rate)
    return A, X

def mask_feat(X: torch.Tensor, mask_prob: float):
    drop_mask = (torch.empty((X.size(1),), dtype=torch.float32, device=X.device).uniform_() < mask_prob)
    X = X.clone()
    X[:, drop_mask] = 0
    return X

def drop_edge(A: torch.sparse.Tensor, drop_prob: float):
    n_edges = A._nnz()
    mask_rates = torch.full((n_edges,), fill_value=drop_prob, dtype=torch.float)
    masks = torch.bernoulli(1 - mask_rates)
    mask_idx = masks.nonzero().squeeze(1)

    E = A._indices()
    V = A._values()

    E = E[:, mask_idx]
    V = V[mask_idx]
    A = torch.sparse_coo_tensor(E, V, A.shape, device=A.device)
    return A

def add_self_loop(A: torch.sparse.Tensor):
    return A + sparse_identity(A.shape[0], device=A.device)

def normalize(A: torch.sparse.Tensor, add_self_loops=True, returnA=False):
    if add_self_loops:
        A_hat = add_self_loop(A)
    else:
        A_hat = A

    D_hat_invsqrt = torch.sparse.sum(A_hat, dim=0).to_dense() ** -0.5
    D_hat_invsqrt[D_hat_invsqrt == torch.inf] = 0
    D_hat_invsqrt = sparse_diag(D_hat_invsqrt)
    A_norm = D_hat_invsqrt @ A_hat @ D_hat_invsqrt
    if returnA:
        return A_hat, A_norm
    else:
        return A_norm

def sparse_identity(dim, device):
    indices = torch.arange(dim).unsqueeze(0).repeat(2, 1)
    values = torch.ones(dim)
    identity_matrix = torch.sparse_coo_tensor(indices, values, size=(dim, dim), device=device)
    return identity_matrix

def sparse_diag(V: torch.Tensor):
    size = V.size(0)
    indices = torch.arange(size).unsqueeze(0).repeat(2, 1)
    values = V
    diagonal_matrix = torch.sparse_coo_tensor(indices, values, size=(size, size), device=V.device)
    return diagonal_matrix