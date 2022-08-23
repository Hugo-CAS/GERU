# -*- coding: utf-8 -*-
"""
Created on Sun Dec 27 22:07:22 2020

@author: Hugo
"""
import numpy as np
import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

def sparse_dot_product(query, key, value, scale_factor, top_k=3):
    
    '''
    key [batch, n_heads, key_len, dim]
    value [batch, n_heads, key_len, dim]
    query [batch, n_heads, query_len, dim]
    '''
    
    scores = torch.matmul(query, key.transpose(2,3)) / scale_factor
    # print(scores)
    # print("shape of scores:", scores.shape)
    if top_k > key.size()[-1]:
        top_k = key.size()[-1]
    if top_k:
        v, _ = torch.topk(scores, top_k)
        # print(v)
        # vk = v[:,:,:,-1].unsqueeze(2).expand_as(scores)
        vk = v[:,:,:,-1].unsqueeze(3).expand_as(scores)
        # print(vk)
        mask_k = torch.lt(scores, vk)
        scores = scores.masked_fill(mask_k, -1e18)
    
    attn = F.softmax(scores,dim=3)
    context = torch.matmul(attn, value)
    return context, attn, mask_k

class ScaledDotProductAttention(nn.Module):
    def __init__(self, top_k, scale_factor, dropout=.1):
        super(ScaledDotProductAttention, self).__init__()
        self.scale_factor = scale_factor
        self.top_k = top_k
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, q, k, v):
        context, attn, mask_k = sparse_dot_product(q, k, v, self.scale_factor, self.top_k)
        return context, attn, mask_k #(b,nh,n,n)

class MultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, n_head, d_model, d_k, d_v, top_k, dropout=0.1):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)
        self.fc = nn.Linear(n_head * d_v, d_model, bias=False)

        self.attention = ScaledDotProductAttention(top_k, d_k ** 0.5)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)


    def forward(self, q, k, v, mask=None):
        # print("MultiHeadAttention:", q.shape, k.shape, v.shape)
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)

        residual = q

        # Pass through the pre-attention projection: b x lq x (n*dv)
        # Separate different heads: b x lq x n x dv
        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        # Transpose for attention dot product: b x n x lq x dv
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        if mask is not None:
            mask = mask.unsqueeze(1)   # For head axis broadcasting.

        q, attn, mask_k = self.attention(q, k, v)
        # print(q.shape)
        # Transpose to move the head dimension back: b x lq x n x dv
        # Combine the last two dimensions to concatenate all the heads together: b x lq x (n*dv)
        q = q.transpose(1, 2).contiguous().view(sz_b, len_q, -1)
        # print(q.shape)
        q = self.dropout(self.fc(q))
        q += residual

        q = self.layer_norm(q)

        return q, attn, mask_k
    
class EncoderLayer(nn.Module):

    def __init__(self, n_head, d_model, d_k, d_v, top_k, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.slf_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, top_k, dropout=dropout)

    def forward(self, enc_input, slf_attn_mask=None):
        enc_output, enc_slf_attn, mask_k = self.slf_attn(
            enc_input, enc_input, enc_input)
        return enc_output, enc_slf_attn, mask_k
    