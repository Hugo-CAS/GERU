# -*- coding: utf-8 -*-
"""
Created on Thu Aug 19 11:17:58 2021

@author: Hugo
"""

import torch
import torch.nn as nn
from torch_geometric.utils import dense_to_sparse
from torch_geometric.nn import GCNConv
from pygcn.models import GCN
from sparse_attn import EncoderLayer

class AGRNNCell(nn.Module):
    def __init__(self, nhead, dim_in, dim_out, top_k):
        super(AGRNNCell, self).__init__()
        # self.node_num = node_num
        self.input_dim = dim_in
        self.hidden_dim = dim_out
        self.head = nhead
       
        self.linear = nn.Linear(self.input_dim, self.hidden_dim)
        
        self.gcngate1 = GCNConv(2*self.hidden_dim, self.hidden_dim)
        self.gcngate2 = GCNConv(2*self.hidden_dim, self.hidden_dim)
        
        self.update = GCNConv(3*self.hidden_dim, self.hidden_dim)
        self.gcor = EncoderLayer(1, 2*self.hidden_dim, self.hidden_dim, self.hidden_dim, top_k, 0.1)

    def forward(self, x, state):
        #x: B, num_nodes, input_dim
        #state: B, num_nodes, hidden_dim
        # print("x shape", x.shape)
        # print("state shape", state.shape)
        device = x.device
        state = state.to(x.device)
        x = self.linear(x)
        # print(x.shape, state.shape)
        input_and_state = torch.cat((x, state), dim=-1)
        
        xx, A, mask_k = self.gcor(input_and_state) #A [b,nh,n,n]
        xx = xx.view(-1, self.hidden_dim * 2)
        num_nodes = A.shape[2]
        A = A.sum(1)#.squeeze() #A [b,n,n]
        A = A.transpose(1,2).contiguous()
        mask_k = torch.logical_not(mask_k)
        # print(mask_k[0])
        mask_k = mask_k.sum(1)#.squeeze()
        # print(mask_k[0])
        if torch.cuda.is_available():
            mask_k = torch.cuda.BoolTensor(mask_k>0)
        else:
            mask_k = torch.BoolTensor(mask_k>1)
        mask_k = mask_k.transpose(1, 2).contiguous()
        edge_weight = torch.masked_select(A, mask_k).to(device)
        
        batch_index, row_index, col_index = torch.where(mask_k)
        row_index = num_nodes * batch_index + row_index
        col_index = num_nodes * batch_index + col_index
        edge_index = torch.cat((row_index.view(1,-1), col_index.view(1,-1)),axis=0)
        edge_index = edge_index.to(device)
        assert len(edge_weight) == edge_index.size()[1]
        
        z = torch.sigmoid(self.gcngate1(xx, edge_index, edge_weight))
        r = torch.sigmoid(self.gcngate2(xx, edge_index, edge_weight))
        
        state = state.view(-1, self.hidden_dim)
        candidate = torch.cat((xx, z*state), dim=-1)
        hc = torch.tanh(self.update(candidate, edge_index, edge_weight))
        h = r*state + (1-r)*hc
        return h.view(-1, num_nodes, self.hidden_dim), A

    def init_hidden_state(self, batch_size, node_num):
        return torch.zeros(batch_size, node_num, self.hidden_dim)
    