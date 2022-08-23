# -*- coding: utf-8 -*-
"""
Created on Tue Aug  3 13:58:57 2021

@author: Hugo

Implement transformer encoders and decoders that are going to be used with
different attention mechanisms.
In all cases the batch dimension is first and the sequence dimension is second.
"""

import torch
from torch.nn import Dropout, LayerNorm, Linear, Module, ModuleList
import torch.nn.functional as F

from fast_transformers.events import EventDispatcher
from fast_transformers.masking import FullMask, LengthMask


class TransformerEncoderLayer(Module):
    """Self attention and feed forward network with skip connections.
    This transformer encoder layer implements the same encoder layer as
    PyTorch but is a bit more open for extension by receiving the attention
    implementation as a constructor argument.
    Arguments
    ---------
        attention: The attention implementation to use given as a nn.Module
        d_model: The input feature dimensionality
        d_ff: The dimensionality of the intermediate features after the
              attention (default: d_model*4)
        dropout: The dropout rate to apply to the intermediate features
                 (default: 0.1)
        activation: {'relu', 'gelu'} Which activation to use for the feed
                    forward part of the layer (default: relu)
        event_dispatcher: str or EventDispatcher instance to be used by this
                          module for dispatching events (default: the default
                          global dispatcher)
    """
    def __init__(self, attention, d_model, d_ff=None, dropout=0.1,
                 activation="relu", event_dispatcher=""):
        super(TransformerEncoderLayer, self).__init__()
        d_ff = d_ff or 4*d_model
        self.attention = attention
        self.linear1 = Linear(d_model, d_ff)
        self.linear2 = Linear(d_ff, d_model)
        self.norm1 = LayerNorm(d_model)
        # self.norm2 = LayerNorm(d_model)
        self.dropout = Dropout(dropout)
        self.activation = getattr(F, activation)
        self.event_dispatcher = EventDispatcher.get(event_dispatcher)

    def forward(self, x, attn_mask=None, length_mask=None):
        """Apply the transformer encoder to the input x.
        Arguments
        ---------
            x: The input features of shape (N, L, E) where N is the batch size,
               L is the sequence length (padded) and E is d_model passed in the
               constructor.
            attn_mask: An implementation of fast_transformers.masking.BaseMask
                       that encodes where each element of x can attend to.
            length_mask: An implementation of
                         fast_transformers.masking.BaseMask that encodes how
                         many elements each sequence in the batch consists of.
        """
        # Normalize the masks
        N = x.shape[0]
        L = x.shape[1]
        attn_mask = attn_mask or FullMask(L, device=x.device)
        length_mask = length_mask or \
            LengthMask(x.new_full((N,), L, dtype=torch.int64))

        # Run self attention and add it to the input
        y, edge_index, A_topk = self.attention(
            x, x, x,
            attn_mask=attn_mask,
            query_lengths=length_mask,
            key_lengths=length_mask
        )
        x = x + self.dropout(y)
        x = self.norm1(x)
        # # Run the fully connected part of the layer
        # y = x = self.norm1(x)
        # y = self.dropout(self.activation(self.linear1(y)))
        # y = self.dropout(self.linear2(y))

        # return self.norm2(x+y) 
        # print("x shape:", x.shape)
        return x, edge_index, A_topk

