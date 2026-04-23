import torch
import torch.nn as nn
import torch.nn.functional as F
from SGformer import SGFormer

class GATConv(nn.Module):
    def __init__(self, in_dim, out_dim, heads=1, dropout=0.0, activation=None, concat=True):
        super(GATConv, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.heads = heads
        self.dropout = dropout
        self.activation = activation
        self.concat = concat
        
        self.W = nn.Parameter(torch.FloatTensor(in_dim, heads * out_dim))
        self.a_src = nn.Parameter(torch.FloatTensor(out_dim, 1))
        self.a_dst = nn.Parameter(torch.FloatTensor(out_dim, 1))

        if concat:
            self.bias = nn.Parameter(torch.FloatTensor(heads * out_dim))
        else:
            self.bias = nn.Parameter(torch.FloatTensor(out_dim))
        
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.W, gain=1.414)
        nn.init.xavier_uniform_(self.a_src, gain=1.414)
        nn.init.xavier_uniform_(self.a_dst, gain=1.414)
        nn.init.zeros_(self.bias)

    def forward(self, A_norm, X, return_attn=False):
        N = X.size(0)
        h = torch.mm(X, self.W).view(N, self.heads, self.out_dim)
        edge_index = A_norm._indices()
        src, dst = edge_index[0], edge_index[1]

        if src.device != X.device:
            src = src.to(X.device)
            dst = dst.to(X.device)
        
        h_src = h[src]
        h_dst = h[dst]

        a_src_expanded = h_src @ self.a_src
        a_dst_expanded = h_dst @ self.a_dst

        e = self.leaky_relu(a_src_expanded + a_dst_expanded).squeeze(-1)
        
        e_max = torch.zeros(N, self.heads, device=e.device)
        e_max.index_reduce_(0, dst, e, 'amax', include_self=False)
        e = e - e_max[dst].clamp(max=10)
        
        e_exp = torch.exp(e.clamp(max=10))
        denominator = torch.zeros(N, self.heads, device=e_exp.device)
        denominator.index_add_(0, dst, e_exp)
        denominator = denominator.clamp(min=1e-8)
        
        alpha = e_exp / denominator[dst]
        alpha_drop = F.dropout(alpha, self.dropout, training=self.training)

        h_agg = torch.zeros(N, self.heads, self.out_dim, device=h.device)
        h_agg.index_add_(0, dst, h_src * alpha_drop.unsqueeze(-1))
        
        if self.concat:
            output = h_agg.view(N, self.heads * self.out_dim) + self.bias
        else:
            output = h_agg.mean(dim=1) + self.bias
        
        if self.activation is not None:
            output = self.activation(output)
        
        if return_attn:
            return output, alpha.mean(dim=1)
        else:
            return output

class GAT(nn.Module):
    def __init__(self, in_dim, hid_dims, heads=1, dropout=0.0):
        super(GAT, self).__init__()
        self.encoder = nn.ModuleList()
        all_dims = [in_dim] + list(hid_dims)
        
        for i in range(len(all_dims) - 1):
            is_last = (i == len(all_dims) - 2)
            self.encoder.append(GATConv(
                all_dims[i] * heads if i > 0 else all_dims[i], 
                all_dims[i + 1], 
                heads=1 if is_last else heads,
                dropout=dropout,
                activation=None if is_last else F.selu,
                concat=not is_last
            ))

    def embed(self, A_norm, X, return_attn=False):
        Z = X
        attn_weights = None
        for i, layer in enumerate(self.encoder):
            if return_attn and i == len(self.encoder) - 1:
                Z, attn_weights = layer(A_norm, Z, return_attn=True)
            else:
                Z = layer(A_norm, Z)
        
        Z = F.normalize(Z, p=2, dim=1)
        if return_attn:
            return Z, attn_weights
        return Z

class DataWrapper:
    def __init__(self, x, edge_index):
        self.graph = {}
        self.graph['node_feat'] = x
        self.graph['edge_index'] = edge_index
        self.graph['edge_weight'] = None 

class DualEncoder(nn.Module):
    def __init__(self, in_dim, hid_dims, heads=1, dropout=0.0, 
                 trans_num_layers=1, trans_heads=1, trans_dropout=0.5, 
                 n_classes=None): 
        super(DualEncoder, self).__init__()
        
        self.local_gat = GAT(in_dim, hid_dims, heads, dropout)
        self.global_trans = SGFormer(
            in_channels=in_dim, 
            hidden_channels=hid_dims[0], 
            out_channels=hid_dims[-1], 
            num_layers=trans_num_layers, 
            num_heads=trans_heads, 
            dropout=trans_dropout, 
            use_graph=False,
            aggregate='add'
        )
        
        self.global_proj = nn.Linear(hid_dims[-1], hid_dims[-1])
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hid_dims[-1])
        self.fusion_weight = nn.Parameter(torch.tensor(0.01))
        
        self.n_classes = n_classes
        if n_classes is not None:
            self.cluster_proj = nn.Linear(hid_dims[-1], n_classes)
            nn.init.orthogonal_(self.cluster_proj.weight)

    def _run_trans(self, A_norm, X):
        edge_index = A_norm._indices()
        data = DataWrapper(X, edge_index)
        out = self.global_trans(data)
        out = self.global_proj(out)
        out = self.dropout(out)
        out = self.layer_norm(out)
        return F.normalize(out, p=2, dim=1)

    def forward(self, A_norm1, A_norm2, X1, X2):
        Z1_local = self.local_gat.embed(A_norm1, X1)
        Z1_global = self._run_trans(A_norm1, X1)
        Z1_fused = F.normalize(Z1_local + self.fusion_weight * Z1_global, p=2, dim=1)
        
        Z2_local = self.local_gat.embed(A_norm2, X2)
        Z2_global = self._run_trans(A_norm2, X2)
        Z2_fused = F.normalize(Z2_local + self.fusion_weight * Z2_global, p=2, dim=1)
        
        if hasattr(self, 'cluster_proj'):
            Y1 = F.softmax(self.cluster_proj(Z1_fused), dim=1)
            Y2 = F.softmax(self.cluster_proj(Z2_fused), dim=1)
            return Z1_fused, Z2_fused, Z1_local, Z1_global, Y1, Y2
        

        return Z1_fused, Z2_fused, Z1_local, Z1_global

    def embed(self, A_norm, X, return_attn=False):
        if return_attn:
            Z_local, gat_attn = self.local_gat.embed(A_norm, X, return_attn=True)
        else:
            Z_local = self.local_gat.embed(A_norm, X, return_attn=False)
        
        Z_global = self._run_trans(A_norm, X)
        Z_final = Z_local + self.fusion_weight * Z_global
        Z_final = F.normalize(Z_final, p=2, dim=1)

        if return_attn:
            return Z_final, gat_attn
        return Z_final