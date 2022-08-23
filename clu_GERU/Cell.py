# -*- coding: utf-8 -*-
"""
Created on Mon Dec 28 20:58:29 2020

@author: Hugo
"""

import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv
from pygcn.models import GCN
from My_ICA import MyImprovedClusteredAttention
from cluster_topk_att import TransformerEncoderLayer
from my_attention_layer import AttentionLayer


class AGRNNCell(nn.Module):
    def __init__(self, nhead, dim_in, dim_out, num_clusters, top_k):
        super(AGRNNCell, self).__init__()
        # self.node_num = node_num
        self.input_dim = dim_in
        self.hidden_dim = dim_out
        self.head = nhead
        self.num_clusters = num_clusters
        self.top_k = top_k
        #add a graph constractor A = GCOR() [b, num_nodes, num_nodes]
        #self.gate = GNN()
        #self.update = GNN()
        
        self.linear = nn.Linear(self.input_dim, self.hidden_dim)
        
        # self.gate1 = GCN(2*self.hidden_dim, self.hidden_dim)
        # self.gate2 = GCN(2*self.hidden_dim, self.hidden_dim)
        
        self.gcngate1 = GCNConv(2*self.hidden_dim, self.hidden_dim)
        self.gcngate2 = GCNConv(2*self.hidden_dim, self.hidden_dim)
        
        
        # self.update = GCN(2*self.hidden_dim, self.hidden_dim) 
        # self.update = GCNConv(2*self.hidden_dim, self.hidden_dim)
        self.update = GCNConv(3*self.hidden_dim, self.hidden_dim)
        # self.gcor = EncoderLayer(1, 2*self.hidden_dim, self.hidden_dim, self.hidden_dim, top_k, 0.1)
        
        
        self.gcor = TransformerEncoderLayer(
                AttentionLayer(
                    MyImprovedClusteredAttention(
                        clusters=num_clusters,
                        topk=top_k
                    ),
                    self.hidden_dim * 2,
                    self.head
                ),
                self.hidden_dim * 2,
                self.head
            )
        
        
        # self.gate = AVWGCN(dim_in+self.hidden_dim, 2*dim_out, cheb_k, embed_dim)
        # self.update = AVWGCN(dim_in+self.hidden_dim, dim_out, cheb_k, embed_dim)

    def forward(self, x, state):
        #x: B, num_nodes, input_dim
        #state: B, num_nodes, hidden_dim
        
        state = state.to(x.device)
        x = self.linear(x)
        # print(x.shape, state.shape)
        input_and_state = torch.cat((x, state), dim=-1)
        
        xx, sparse_edges, A = self.gcor(input_and_state) #A [b,nh,n,topk]
        # print("shape of xx:", xx.shape)
        xx = xx.view(-1, self.hidden_dim * 2)
        A = A.permute(0, 2, 1, 3).contiguous() #[b,n,nh,topk]
        # print("shape of A:", A.shape)
        batch_size, num_nodes, nheads, topk = A.shape
        
        
        # A = A.reshape(batch_size, num_nodes, -1) #[b,n,nh*topk]
        edges_1 = torch.arange(batch_size*num_nodes).unsqueeze(-1).repeat(1, nheads*topk)
        # print(edges_1)
        # print("shape of edges_1:", edges_1.shape)
        edges_1 = edges_1.view(1,-1).to(x.device)
        
        # print("nheads*topk:", nheads*topk)
        flat = torch.arange(batch_size).unsqueeze(-1).repeat(1, nheads*topk).view(batch_size, 1, -1).to(x.device) #(B,1,h*topk)
        # print("shape of flat:", flat.shape)
        # print(flat[0])
        sparse_edges = sparse_edges.permute(0,2,1,3).reshape(batch_size, num_nodes,-1)
        # print("shape of sparse_edges:", sparse_edges.shape)
        edges_2 = sparse_edges + flat
        # print(sparse_edges)
        # print(edges_2)
        edges_2 = edges_2.reshape(1,-1)
        edge_index_1 = torch.cat((edges_1, edges_2), axis=1)
        edge_index_2 = torch.cat((edges_2, edges_1), axis=1)
        edge_index = torch.cat((edge_index_1, edge_index_2),axis=0)
        # edge_index = edge_index.T
        # print("edge_index.shape:", edge_index.shape)
        edge_weight_1 = A.reshape(-1)
        edge_weight_2 = A.reshape(-1)
        edge_weight = torch.cat((edge_weight_1, edge_weight_2))
        # print("edge_weight.shape:", edge_weight.shape)
  
    
        # print(edge_index[0])
        # print(edge_weight[0])
        # z = torch.sigmoid(self.gate1(input_and_state, A))
        # r = torch.sigmoid(self.gate2(input_and_state, A))
        # print(input_and_state.shape, edge_index.shape, edge_weight.shape)
        z = torch.sigmoid(self.gcngate1(xx, edge_index, edge_weight))
        # print("z.shape:", z.shape)
        r = torch.sigmoid(self.gcngate2(xx, edge_index, edge_weight))
        # print("r.shape:", r.shape)
        state = state.view(-1, self.hidden_dim)
        # print("state.shape:", state.shape)
        
        # x = x.view(-1, self.hidden_dim)
        # candidate = torch.cat((x, z*state), dim=-1) #可以调整
        
        candidate = torch.cat((xx, z*state), dim=-1)
        
        
        # print("shape of candidate:", candidate.shape)
        # hc = torch.tanh(self.update(candidate, A))
        hc = torch.tanh(self.update(candidate, edge_index, edge_weight))
        h = r*state + (1-r)*hc
        # h = torch.cat((x, h), dim=-1)
        # return h, A
        return h.view(-1, num_nodes, self.hidden_dim), (edge_index, edge_weight)

    def init_hidden_state(self, batch_size, node_num):
        return torch.zeros(batch_size, node_num, self.hidden_dim)
    