# -*- coding: utf-8 -*-
"""
Created on Mon Aug  2 16:12:54 2021

@author: Hugo
"""

#
# Copyright (c) 2020 Idiap Research Institute, http://www.idiap.ch/
# Written by Angelos Katharopoulos <angelos.katharopoulos@idiap.ch>,
# Apoorv Vyas <avyas@idiap.ch>
#

"""Implement improved clustered self attention."""

from math import sqrt

import numpy as np
import torch
import torch.autograd
from torch.nn import Dropout, Module
from torch.nn.init import normal_

from fast_transformers.attention_registry import AttentionRegistry, Optional, Float, Int, \
    Bool, EventDispatcherInstance
from fast_transformers.events import EventDispatcher
from fast_transformers.masking import FullMask
from fast_transformers.aggregate import clustered_aggregate, clustered_broadcast
from fast_transformers.clustering.hamming import cluster
from fast_transformers.hashing import compute_hashes
from fast_transformers.sparse_product import sparse_dot_product, sparse_weighted_average
from fast_transformers.sparse_product import clustered_sparse_dot_product, \
    clustered_sparse_weighted_average


class _GroupQueries(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, clusters, counts, lengths):
        factors = 1./counts.float()
        q_grouped = clustered_aggregate(Q, clusters, factors, lengths)
        ctx.save_for_backward(clusters, counts, factors)

        return q_grouped

    @staticmethod
    def backward(ctx, grad_q_grouped):
        clusters, counts, factors = ctx.saved_tensors
        grad_q = clustered_broadcast(grad_q_grouped, clusters, counts, factors)

        return grad_q, None, None, None


class _BroadcastValues(torch.autograd.Function):
    @staticmethod
    def forward(ctx, v_grouped, clusters, counts, lengths):
        factors = torch.ones_like(counts, dtype=v_grouped.dtype)
        V = clustered_broadcast(v_grouped, clusters, counts, factors)
        ctx.save_for_backward(clusters, counts, factors, lengths)

        return V

    @staticmethod
    def backward(ctx, grad_v):
        clusters, counts, factors, lengths = ctx.saved_tensors
        grad_v_grouped = clustered_aggregate(grad_v, clusters, factors, lengths)

        return grad_v_grouped, None, None, None, None


def re_orgnizedA(L, num_clusters, A_c, A_topk, sorted_indx, topk, counts):
    N, H, C, S = A_c.shape
    A_c = A_c.cpu()
    A_topk = A_topk.cpu()
    sorted_indx = sorted_indx.cpu()
    topk = topk.cpu()
    counts = counts.cpu()
    # print(counts.shape)
    # print(C)
    
    A = torch.zeros((N,H,L,S))
    num_point = torch.zeros((N,H,C+1)) #countsd(N, H, C)
    for i in range(C):
        num_point[:,:,i+1] = num_point[:,:,i] + counts[:,:,i]
    for n in range(N):
        for h in range(H):
            for i in range(L):
                for j in range(C):
                    if i >= num_point[n,h,j] and i < num_point[n,h,j+1]:
                        A[n,h,i,:] = A_c[n,h,j,:]
                        A[n,h,i,topk[n,h,j,:]] = A_topk[n,h,i,:]
    # A[topk] = A_topk
    sorted_rev_indx = torch.argsort(sorted_indx, dim=-1)
    q_offset = torch.arange(N*H).unsqueeze(-1) * L
    q_rev_flat = (sorted_rev_indx.view(N*H, -1) + q_offset).reshape(-1)
    A_new = A.reshape(-1, S).index_select(0, q_rev_flat).view(N,H,L,S)
    return A_new

class MyImprovedClusteredAttention(Module):
    """
    Immproved clustered attention approximation by recompution attention
    for each query with the top-k keys for the corresponding cluster.
    Given the queries, keys, and values as Q, K, and V respectively, we
    first cluster the queries in "C" groups and compute the "C" query centroids
    Q_c.
    We now use to the centroids Q_c to identify the top-k keys with highest
    dot products.
    Subsequently, for each query we compute the sparse dot product with
    the corresponding top-k keys to improve the attention approximation.
    Arguments
    ---------
        clusters: How many clusters to group the queries into
        iterations: The number of lloyd iterations to perform (default: 10)
        bits: How many bits to use for the hash (default: 32)
        hash_bias: If true, hamming distance proportional to L2 distance
                   If false, hamming distance proportional to cosine distance
                   (default: True)
        topk: Number of top-k keys to for improved approximation (default: 32)
        softmax_temp: The temperature to use for the softmax attention.
                      (default: 1/sqrt(d_keys) where d_keys is computed at
                      runtime)
        attention_dropout: The dropout rate to apply to the attention
                           (default: 0.1)
        event_dispatcher: str or EventDispatcher instance to be used by this
                          module for dispatching events (default: the default
                          global dispatcher)
    """
    def __init__(self, clusters, iterations=10, bits=32,
                 hash_bias=True, topk=32, softmax_temp=None,
                 attention_dropout=0.1, event_dispatcher=""):
        super(MyImprovedClusteredAttention, self).__init__()
        self.clusters = clusters
        self.iterations = iterations
        self.bits = bits
        self.hash_bias = hash_bias
        self.topk = topk
        self.softmax_temp = softmax_temp
        self.dropout = Dropout(attention_dropout)
        self.event_dispatcher = EventDispatcher.get(event_dispatcher)

    def _create_query_groups(self, Q, query_lengths):
        N, H, L, E = Q.shape

        # Compute the hashes for all the queries
        planes = Q.new_empty((self.bits, E+1))
        normal_(planes)
        if not self.hash_bias:
            planes[:, -1] = 0
        hashes = compute_hashes(Q.view(N*H*L, E), planes).view(N, H, L)

        # Cluster the hashes and return the cluster index per query
        clusters, counts =  cluster(
            hashes,
            query_lengths._lengths.int(),
            clusters=self.clusters,
            iterations=self.iterations,
            bits=self.bits
        )
        sorted_clusters, sorted_indx = torch.sort(clusters, dim=-1)
        return (sorted_clusters, counts), sorted_indx

    def _topk_attention(self, Q, K, V,
                        clusters, counts,
                        topk, topk_values,
                        A_bottomk, softmax_temp,
                        query_lengths):
        """Return the attention with just the topk heads."""
        # Extract some indices
        N, H, L, E = Q.shape
        _, _, S, _ = K.shape
        _, _, C, k = topk.shape

        # We need to pass the output tensor to initialize to 0
        QK = clustered_sparse_dot_product(
            Q, K, topk,
            clusters, counts,
            query_lengths._lengths.int()
        ) #与topk相乘之后的结果
        # print("shape of top QK:", QK.shape)
        # We need to mask the topk dot products if topk > input_length
        QK = QK.masked_fill(
            torch.isinf(topk_values[:,0,0,:]).view(N, 1, 1, k),
            float("-inf")
        )
        A = torch.softmax(softmax_temp * QK, dim=-1) 
        # print("A.shape:", A.shape)
        # print(A[0,0,:,:])
        assert A_bottomk.is_contiguous()
        # print("A_bottomk.shape:", A_bottomk.shape)
        A_bottomk = clustered_broadcast(
            A_bottomk.unsqueeze(3),
            clusters,
            counts,
            torch.ones_like(counts, dtype=torch.float32)
        )
        # print("A_bottomk.shape:", A_bottomk.shape)
        A = A * (1.0 - A_bottomk) #1-bottom的和就是mj Eq(10)
        A_topk = A.clone()
        # print("A_1.shape:", A.shape)
        A = self.dropout(A)
        assert A.is_contiguous()
        V_new = clustered_sparse_weighted_average(A, V, topk, clusters, counts)
        # print("V_new.shape:", V_new.shape)
        return V_new, A_topk

    def _broadcast_values(self, V, clusters, counts, lengths):
        """Broadcast the values back to the correct positions but make sure
        that the gradient flows properly."""
        V_new = _BroadcastValues.apply(V.contiguous(), clusters, counts, lengths)
        return V_new

    def _bottomk_attention(self, QK, V, clusters, counts, query_lengths, topk, softmax_temp):
        """Return the attention with just the bottomk keys."""
        N, H, C, S = QK.shape

        A = torch.softmax(softmax_temp * QK, dim=-1)
        mask = QK.new_ones(QK.shape)
        mask[
            torch.arange(N, device=QK.device).view(N, 1, 1, 1),
            torch.arange(H, device=QK.device).view(1, H, 1, 1),
            torch.arange(C, device=QK.device).view(1, 1, C, 1),
            topk,
        ] = 0
        A = A * mask #(N,H,C,S)
        A_t = A.detach().cpu()
        A_bottomk = A.sum(-1) #(N,H,C,S)
        # print("shape of A_bottomk:", A_bottomk.shape)
        A = self.dropout(A)
        # Compute the values
        V_new = torch.einsum("nhls,nhse->nhle", A, V)
        # print("shape of V_new:", V_new.shape)
        # Broadcast the values back depending on the groups
        V_new = self._broadcast_values(V_new, clusters, counts, query_lengths._lengths.int())

        return V_new, A_bottomk, A_t #A_bottomk (N,H,C)

    def forward(self, queries, keys, values, attn_mask, query_lengths,
                key_lengths):
        # Make sure that there is no attention mask
        assert attn_mask.all_ones, ("Improved-clustered attention cannot "
                                    "use an arbitrary attention mask.")

        queries = queries.permute(0,2,1,3).contiguous()
        keys = keys.permute(0,2,1,3).contiguous()
        values = values.permute(0,2,1,3).contiguous()
        N, H, L, E = queries.shape
        _, _, S, D = values.shape
        softmax_temp = self.softmax_temp or 1./sqrt(E)
        # print("N:{}, H:{}, L:{}, E:{}, S:{}, D:{}, softmax_temp:{}".format(N, H, L, E, S, D, softmax_temp))
        # print("query_lengths shape:", query_lengths.shape)
        # print("query_lengths:", query_lengths) #(N, L)
        # print("query_lengths.lengths:", query_lengths.lengths)
        
        
        # Cluster the queries into groups
        groups, sorted_indx = self._create_query_groups(queries, query_lengths)#clusters(N,H,L)的每个元素在原来的L中的索引
        clusters, counts = groups #counts每个cluster对于的个数, clusters(N,H,L),指定所属的cluster
        
        # print("clusters shape:", clusters.shape)
        # print(clusters[0,0,:])
        # print("counts shape:", counts.shape) 
        # print(counts[0,:,:])
        # print("type of sorted_index:", type(sorted_indx))
        # print("shape of sorted_index:", sorted_indx.shape) #(N,H,L) 每个位置为对应cluster[i,j]所对应的index
        # print(sorted_indx[0,0,:])

        # Re-organize queries so that first group belong to first cluster
        # next to second cluster and so on. This improves kernel implementations.
        # Note that this step is introduced after NeurIPS submission and
        # now the complexity is O(N log(N)).
        q_offset = torch.arange(N*H, device=queries.device).unsqueeze(-1) * L
        q_flat = (sorted_indx.view(N*H, -1) + q_offset).reshape(-1)
        s_queries = queries.reshape(-1, E).index_select(0, q_flat).view(N,H,L,E)#妙(N,H,L,E) sorted queries

        # Aggregate the re-arranged queries.
        Q_grouped = _GroupQueries.apply(s_queries, *groups, query_lengths.lengths. int())
        # Compute the attention
        # print("Q_grouped shape:{}, key shape:{}".format(Q_grouped.shape, keys.shape))
        QK = torch.einsum("nhle,nhse->nhls", Q_grouped, keys) #先计算了A^c (N,H,L,S)
        # print("shape of QK:", QK.shape)
        # print("shape of key_lengths.additive_matrix", key_lengths.additive_matrix.shape)
        QK = QK + key_lengths.additive_matrix[:, None, None, :]
        topk_values, topk = torch.topk(QK, min(self.topk, S), sorted=False, dim=-1)#S中的topk,是sorted之后的topk(N,H,C,k)
        # print(topk[clusters].shape)
        # print("shape topk_values:", topk_values.shape)
        # print("shape topk:", topk.shape)
        # print(topk[0,0,:,:])
        # print(topk.shape)
        # topk[torch.arnage(N).view(N,1,1,1), torch.arange(H).view(1,H,1,1), clusters[:,:,]]
        #print(torch.arange(N).view(N,1,1,1), torch.arange(H).view(1,H,1,1), clusters.view(N,H,L,1), torch.arange(self.topk).view(1,1,1,self.topk))
        sparse_index = topk[torch.arange(N).view(N,1,1,1), torch.arange(H).view(1,H,1,1), clusters.long().view(N,H,L,1), torch.arange(self.topk).view(1,1,1,self.topk)]
        # print("shape of sparse_index:", sparse_index.shape)
        
        assert topk.is_contiguous()

        # Now compute the attention with only the bottom keys
        V_bottomk, A_bottomk, A_c = self._bottomk_attention(
            QK, values,
            clusters, counts,
            query_lengths,
            topk,
            softmax_temp
        )

        # Now compute the attention with only the top keys
        V_topk, A_topk = self._topk_attention(
            s_queries, keys, values,
            clusters, counts,
            topk, topk_values,
            A_bottomk,
            softmax_temp,
            query_lengths
        )
        # print(V_topk.shape, V_bottomk.shape)
        V_sorted_new = V_topk + V_bottomk

        # Reverse the previous mapping
        sorted_rev_indx = torch.argsort(sorted_indx, dim=-1)
        q_rev_flat = (sorted_rev_indx.view(N*H, -1) + q_offset).reshape(-1)
        V_new = V_sorted_new.reshape(-1, D).index_select(0, q_rev_flat).view(N,H,L,D)
        sparse_index_new = sparse_index.reshape(-1,self.topk).index_select(0, q_rev_flat).view(N,H,L,self.topk)
        A_topk_new = A_topk.reshape(-1, self.topk).index_select(0, q_rev_flat).view(N,H,L,self.topk)
        # print("shape of sparse_index_new:", sparse_index_new.shape)
        # A_new = re_orgnizedA(L, self.clusters, A_c, A_topk, sorted_indx, topk, counts)
        
        # print(A_new[0,0,:,:])
        
        return V_new.permute(0, 2, 1, 3).contiguous(), sparse_index_new, A_topk_new


# Register the attention implementation so that it becomes available in our
# builders
AttentionRegistry.register(
    "myimproved-clustered", MyImprovedClusteredAttention,
    [
        ("clusters", Int),
        ("iterations", Optional(Int, 10)),
        ("bits", Optional(Int, 63)),
        ("hash_bias", Optional(Bool, True)),
        ("topk", Optional(Int, 32)),
        ("softmax_temp", Optional(Float)),
        ("attention_dropout", Optional(Float, 0.1)),
        ("event_dispatcher", Optional(EventDispatcherInstance, ""))
    ]
)