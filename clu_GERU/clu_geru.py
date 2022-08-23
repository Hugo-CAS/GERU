# -*- coding: utf-8 -*-
"""
Created on Fri Aug 20 14:54:13 2021

@author: Hugo
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from Cell_1 import AGRNNCell

class AGRNN(nn.Module):
    def __init__(self, nhead, dim_in, dim_out, num_clusters, top_k, num_layers=1):
        super(AGRNN, self).__init__()
        assert num_layers >= 1, 'At least one DCRNN layer in the Encoder.'
        # self.node_num = node_num
        self.input_dim = dim_in #7
        self.num_layers = num_layers #1
        self.dcrnn_cells = nn.ModuleList()
        self.dcrnn_cells.append(AGRNNCell(nhead, dim_in, dim_out, num_clusters, top_k))
        for _ in range(1, num_layers):
            self.dcrnn_cells.append(AGRNNCell(nhead, dim_out, dim_out, num_clusters, top_k))

    def forward(self, x, init_state): #初始state=0
        #shape of x: (B, T, N, D)
        #shape of init_state: (num_layers, B, N, hidden_dim)
        # print(x.shape)
        # assert x.shape[2] == self.node_num and x.shape[3] == self.input_dim
        seq_length = x.shape[0]
        current_inputs = x
        output_hidden = []
        for i in range(self.num_layers):
            state = init_state[i] #(B,N,d)
            inner_states = []
            A_list = []
            for t in range(seq_length):
                state, A = self.dcrnn_cells[i](current_inputs[t, :, :, :], state)
                inner_states.append(state)
                # A_list.append(A.squeeze().detach().cpu().numpy())
            output_hidden.append(state)
            current_inputs = torch.stack(inner_states, dim=1)                                                                              
        #current_inputs: the outputs of last layer: (B, T, N, hidden_dim)
        #output_hidden: the last state for each layer: (num_layers, B, N, hidden_dim)
        #last_state: (B, N, hidden_dim)
        return current_inputs, output_hidden#, A_list

    def init_hidden(self, batch_size, num_node):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.dcrnn_cells[i].init_hidden_state(batch_size,num_node))
        return torch.stack(init_states, dim=0)      #(num_layers, B, N, hidden_dim) B=1, N=num_node * bs
    
class KAGRNN(nn.Module):
    def __init__(self, args):
        super(KAGRNN, self).__init__()
        self.head = args.head
        # self.num_node = args.num_nodes
        self.input_dim = args.input_dim
        self.hidden_dim = args.rnn_units
        self.output_dim = args.output_dim
        # self.horizon = args.horizon
        self.num_layers = args.num_layers
        self.num_clusters = args.num_clusters
        self.top_k = args.top_k
        
        self.encoder = AGRNN(self.head, self.input_dim, self.hidden_dim, self.num_clusters, self.top_k, self.num_layers)
        self.fc = nn.Sequential(nn.Linear(self.hidden_dim, self.output_dim), nn.ReLU(True))
        # self.fc = nn.Sequential(nn.Linear(self.hidden_dim+42, self.output_dim), nn.ReLU(True))
        
    def forward(self, x):
        
        init_state = self.encoder.init_hidden(x.shape[1], x.shape[2])
        
        # output, _, A_list = self.encoder(x, init_state)
        output, _ = self.encoder(x, init_state)
        
        output = output[:, -1:, :, :]
        # output = torch.squeeze(output)
        # x = x.squeeze().transpose(0,1).reshape((output.shape[0],-1))
        # print(output.shape, x.shape)
        # xo = torch.cat((x,output), dim=-1)
        output = self.fc(output)
        output = output.view(-1,2)
        
        return F.log_softmax(output,dim=1)#, A_list