"""
CSIT5210 - Data Mining and Knowledge Discovery
@author: HUANG, Yanzhen | Deng Zhenxiao
@date: Nov 6, 2025
@description: Key components for implementing the
downstream models. They adheres to the original 
implementation of the source code for consistence.

@reference:

https://github.com/HappyPointer/LLM2Rec/blob/main

(See full citation in README)
"""

import math
from typing import cast, List

import torch
from torch import nn
from run_LLM.downstream_model_class.data_classes import SASRecModelArgs


class FNN(nn.Module):
    """
    Transformer's feed-forward layer.
    """

    def __init__(
            self,
            mha_hidden_size: int,
            fnn_hidden_size: int,
            fnn_hidden_dropout: float,
            layer_norm_eps: float
    ):
        super(FNN, self).__init__()

        self.fnn1 = nn.Linear(mha_hidden_size, fnn_hidden_size)

        self.fnn2 = nn.Linear(fnn_hidden_size, mha_hidden_size)
        self.dropout = nn.Dropout(fnn_hidden_dropout)
        self.layer_norm = nn.LayerNorm(
            mha_hidden_size, eps=layer_norm_eps)
        
    def gelu(self, x):
        """
        Note: This adheres to the original implementation.
        """
        return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))
    
    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """
        Forward-through the neuronetwork.
        """
        hidden_states = self.fnn1(input_tensor)
        hidden_states = self.gelu(hidden_states)

        hidden_states = self.fnn2(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.layer_norm(hidden_states + input_tensor)

        return hidden_states


class MultiHeadAttention(nn.Module):
    """
    A multi-head attention block.
    """

    def __init__(
            self,
            n_heads: int,
            mha_hidden_size: int,
            fnn_hidden_dropout: float,
            attn_dropout: float,
            layer_norm_eps: float
    ):
        super(MultiHeadAttention, self).__init__()

        assert mha_hidden_size % n_heads == 0

        # Multi-head attention size distribution
        self.n_heads = n_heads
        self.attn_head_size = mha_hidden_size // n_heads
        self.all_head_size = mha_hidden_size

        # Attention scaling
        self.sqrt_dk = math.sqrt(self.attn_head_size)

        # Q, K, V matrices
        self.w_q = nn.Linear(mha_hidden_size, self.all_head_size)
        self.w_k = nn.Linear(mha_hidden_size, self.all_head_size)
        self.w_v = nn.Linear(mha_hidden_size, self.all_head_size)

        # FNN
        self.softmax = nn.Softmax(dim=-1) # row-wise
        self.fnn = nn.Linear(mha_hidden_size, mha_hidden_size)

        # Layer norm
        self.layer_norm = nn.LayerNorm(mha_hidden_size, eps=layer_norm_eps)

        # Dropouts
        self.attn_dropout = nn.Dropout(attn_dropout)
        self.dropout = nn.Dropout(fnn_hidden_dropout)

    def _distribute_mat(self, x: torch.Tensor) -> torch.Tensor:
        """
        Divide tensor's hidden size (last dimension)
        into #. attn_heads x attn_head_size

        (#. batch, #. seq, hidden_size)
        => (#. batch, #. seq, #. attn_head, attn_head_size)
        """
        x = x.view(x.size()[:-1] + (self.n_heads, self.attn_head_size))
        return x
    
    def _collect_mat(self, x: torch.Tensor) -> torch.Tensor:
        """
        Collect tensor's distributed size (last two dimensions)
        into a single dimension size.

        (#. batch, #. seq, #. attn_head, attn_head_size)
        => (#. batch, #. seq, hidden_size)
        """

        x = x.view(x.size()[:-2] + (self.all_head_size,))
        return x
    
    def forward(self, input_tensor: torch.Tensor, 
                attn_mask: torch.Tensor) -> torch.Tensor:
        """
        Given a batch of input sequences,
        output a batch of hidden state sequences.
        """
        
        _Q = self.w_q(input_tensor)
        _K = self.w_k(input_tensor)
        _V = self.w_v(input_tensor)

        # (#. batch, #. attn_head, ...)
        Q = self._distribute_mat(_Q).permute(0, 2, 1, 3)
        KT = self._distribute_mat(_K).permute(0, 2, 3, 1)
        V = self._distribute_mat(_V).permute(0, 2, 1, 3)

        # attention = softmax((QK^T)/√dk)V
        _attn_weights = (torch.matmul(Q, KT)) / self.sqrt_dk
        _attn_weights += attn_mask

        attn_weights = self.softmax(_attn_weights)
        attn_weights = self.attn_dropout(attn_weights)
        attn_weights = cast(torch.Tensor, attn_weights)

        attention = torch.matmul(attn_weights, V)
        attention = attention.permute(0, 2, 1, 3).contiguous()

        # FNN
        attention = self._collect_mat(attention)
        hidden_states = self.fnn(attention)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.layer_norm(hidden_states + input_tensor)

        return hidden_states


class TransformerBlock(nn.Module):
    """
    A transformer block containing a
    multi-head attention layer (mha) and a
    feed-forward layer (fnn).
    """

    def __init__(
            self, n_heads: int,
            mha_hidden_size: int,
            fnn_hidden_size: int,
            fnn_hidden_dropout: float,
            attn_dropout: float,
            layer_norm_eps: float
    ):
        super(TransformerBlock, self).__init__()
        self.mha = MultiHeadAttention(
            n_heads, 
            mha_hidden_size=mha_hidden_size,
            fnn_hidden_dropout=fnn_hidden_dropout,
            attn_dropout=attn_dropout,
            layer_norm_eps=layer_norm_eps
        )

        self.fnn = FNN(
            mha_hidden_size=mha_hidden_size,
            fnn_hidden_size=fnn_hidden_size,
            fnn_hidden_dropout=fnn_hidden_dropout,
            layer_norm_eps=layer_norm_eps
        )

    def forward(self, hidden_states: torch.Tensor, 
                attn_mask: torch.Tensor) -> torch.Tensor:
        """
        Forward a batch of embedding sequences.
        Output a batch of hidden state sequences.
        """
        mha_output = self.mha(hidden_states, attn_mask)
        fnn_output = self.fnn(mha_output)
        return fnn_output


class TransformerEncoder(nn.Module):
    """
    A transformer encoder.
    """

    def __init__(self, config: SASRecModelArgs):
        super(TransformerEncoder, self).__init__()

        self.layers = nn.ModuleList([
            TransformerBlock(
                n_heads=config.num_heads,
                mha_hidden_size=config.hidden_size,
                fnn_hidden_size=256,
                fnn_hidden_dropout=config.dropout,
                attn_dropout=config.dropout,
                layer_norm_eps=1e-12
            ) for _ in range(config.layer_num)])
    
    def forward(self, hidden_states: torch.Tensor,
                attention_mask: torch.Tensor, ) -> List[torch.Tensor]:
        """
        Given a batch of embedding sequences,
        output all transformer block's output batch 
        of hidden state sequences.
        """
        
        all_layer_hiddenstates = []
        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask)
            all_layer_hiddenstates.append(hidden_states)
        return all_layer_hiddenstates
    

    def get_attn_mask(self, item_seq: torch.Tensor, 
                      bidirectional: bool) -> torch.Tensor:
        """
        Given item sequence, get attention mask.
        """

        _attn_mask = (item_seq != 0)    # (#. batch, #. seq)

        # (#. batch, 1, 1, #. seq)
        attn_mask = _attn_mask.unsqueeze(1).unsqueeze(2)

        if not bidirectional:
            seq_len = item_seq.size(-1)

            # Make last dim a square mat of seq_len x seq_len
            attn_mask_ = attn_mask.expand((-1, -1, seq_len, -1))
            attn_mask = torch.tril(attn_mask_)
        
        # bidirectional: (#. batch, 1, 1, #. seq)
        # not bidirectional: (#. batch, #. seq, #. seq)
        attn_mask = torch.where(attn_mask, 0.0, -10_000.0)
        return attn_mask


    def gather_batch_indices(
            self, output: torch.Tensor, 
            idx: torch.Tensor) -> torch.Tensor:
        """
        For all squences in the mini-batch do:

        For a specific sequence i, find the i-th index.
        Extract the hidden_size-length embedding from the
        sequence.

        Result in a batch_size x hidden_size tensor.
        """

        idx = idx.view(-1, 1, 1).expand(
            -1, -1, output.shape[-1]) # (batch_size, ) => (batch_size, 1, 1)

        output = output.gather(dim=1, index=idx) # (batch_size, 1, hidden_size)

        return output.squeeze(1) # (batch_size, hidden_size)