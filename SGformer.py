import math
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import degree

def full_attention_conv(qs, ks, vs, output_attn=False):
    qs = qs / torch.norm(qs, p=2) 
    ks = ks / torch.norm(ks, p=2) 
    N = qs.shape[0]

    kvs = torch.einsum("lhm,lhd->hmd", ks, vs)
    attention_num = torch.einsum("nhm,hmd->nhd", qs, kvs) 
    attention_num += N * vs

    all_ones = torch.ones([ks.shape[0]]).to(ks.device)
    ks_sum = torch.einsum("lhm,l->hm", ks, all_ones)
    attention_normalizer = torch.einsum("nhm,hm->nh", qs, ks_sum)

    attention_normalizer = torch.unsqueeze(attention_normalizer, len(attention_normalizer.shape)) 
    attention_normalizer += torch.ones_like(attention_normalizer) * N
    attn_output = attention_num / attention_normalizer 

    if output_attn:
        attention=torch.einsum("nhm,lhm->nlh", qs, ks).mean(dim=-1) 
        normalizer=attention_normalizer.squeeze(dim=-1).mean(dim=-1,keepdims=True)
        attention=attention/normalizer
        return attn_output, attention
    else:
        return attn_output


class TransConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, num_heads, use_weight=True):
        super().__init__()
        self.Wk = nn.Linear(in_channels, out_channels * num_heads)
        self.Wq = nn.Linear(in_channels, out_channels * num_heads)
        if use_weight:
            self.Wv = nn.Linear(in_channels, out_channels * num_heads)
        self.out_channels = out_channels
        self.num_heads = num_heads
        self.use_weight = use_weight

    def reset_parameters(self):
        self.Wk.reset_parameters()
        self.Wq.reset_parameters()
        if self.use_weight:
            self.Wv.reset_parameters()

    def forward(self, query_input, source_input, edge_index=None, edge_weight=None, output_attn=False):
        query = self.Wq(query_input).reshape(-1, self.num_heads, self.out_channels)
        key = self.Wk(source_input).reshape(-1, self.num_heads, self.out_channels)
        if self.use_weight:
            value = self.Wv(source_input).reshape(-1, self.num_heads, self.out_channels)
        else:
            value = source_input.reshape(-1, 1, self.out_channels)

        if output_attn:
            attention_output, attn = full_attention_conv(query, key, value, output_attn)
        else:
            attention_output = full_attention_conv(query, key, value)

        final_output = attention_output.mean(dim=1)
        if output_attn:
            return final_output, attn
        else:
            return final_output


class TransConv(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_layers=2, num_heads=1,
                 alpha=0.5, dropout=0.5, use_bn=True, use_residual=True, use_weight=True, use_act=False):
        super().__init__()
        self.convs = nn.ModuleList()
        self.fcs = nn.ModuleList()
        self.fcs.append(nn.Linear(in_channels, hidden_channels))
        self.bns = nn.ModuleList()
        self.bns.append(nn.LayerNorm(hidden_channels))
        for i in range(num_layers):
            self.convs.append(TransConvLayer(hidden_channels, hidden_channels, num_heads=num_heads, use_weight=use_weight))
            self.bns.append(nn.LayerNorm(hidden_channels))
        self.dropout = dropout
        self.activation = F.relu
        self.use_bn = use_bn
        self.residual = use_residual
        self.alpha = alpha
        self.use_act=use_act

    def forward(self, data):
        x = data.graph['node_feat']
        edge_weight = data.graph['edge_weight'] if 'edge_weight' in data.graph else None
        layer_ = []

        x = self.fcs[0](x)
        if self.use_bn:
            x = self.bns[0](x)
        x = self.activation(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        layer_.append(x)

        for i, conv in enumerate(self.convs):
            x = conv(x, x, edge_index, edge_weight) 
            if self.residual:
                x = self.alpha * x + (1-self.alpha) * layer_[i]
            if self.use_bn:
                x = self.bns[i+1](x)
            if self.use_act:
                x = self.activation(x) 
            x = F.dropout(x, p=self.dropout, training=self.training)
            layer_.append(x)

        return x

    def get_attentions(self, x):
        layer_, attentions = [], []
        x = self.fcs[0](x)
        if self.use_bn:
            x = self.bns[0](x)
        x = self.activation(x)
        layer_.append(x)
        for i, conv in enumerate(self.convs):
            x, attn = conv(x, x, output_attn=True)
            attentions.append(attn)
            if self.residual:
                x = self.alpha * x + (1 - self.alpha) * layer_[i]
            if self.use_bn:
                x = self.bns[i + 1](x)
            layer_.append(x)
        return torch.stack(attentions, dim=0)

class SGFormer(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=2, num_heads=1, 
                 alpha=0.5, dropout=0.5, use_bn=True, use_residual=True, use_weight=True, use_graph=True, use_act=False, graph_weight=0.8, gnn=None, aggregate='add'):
        super().__init__()
        self.trans_conv=TransConv(in_channels,hidden_channels,num_layers,num_heads,alpha,dropout,use_bn,use_residual,use_weight)
        self.gnn=gnn
        self.use_graph=use_graph
        self.graph_weight=graph_weight
        self.use_act=use_act
        self.aggregate=aggregate

        if aggregate=='add':
            self.fc=nn.Linear(hidden_channels,out_channels)
        elif aggregate=='cat':
            self.fc=nn.Linear(2*hidden_channels,out_channels)
        else:
            raise ValueError(f'Invalid aggregate type:{aggregate}')

    def forward(self,data):
        x1=self.trans_conv(data)
        if self.use_graph and self.gnn is not None:
            x2=self.gnn(data)
            if self.aggregate=='add':
                x=self.graph_weight*x2+(1-self.graph_weight)*x1
            else:
                x=torch.cat((x1,x2),dim=1)
        else:
            x=x1
        x=self.fc(x)
        return x
    
    def get_attentions(self, x):
        return self.trans_conv.get_attentions(x)