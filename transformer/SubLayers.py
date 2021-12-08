''' Define the sublayers in encoder/decoder layer '''
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from .Modules import ScaledDotProductAttention

__author__ = "Yu-Hsiang Huang"

class SPA(nn.Module):
    ''' Selective parallel attention '''
    def __init__(
            self,
            n_head: int=8,
            d_v: int=64):
        super().__init__()

        self.gap = nn.AdaptiveAvgPool1d(1)
        self.sk = nn.Linear(d_v, n_head * d_v)
        self.softmax = nn.Softmax(dim=1)

    def forward(
            self,
            x # multi-head feature of shape [bs, n_head, lq=len_seq, d_v=channels]
        ):

        bs, n_head, lq, d_v = x.size()

        u = x.sum(dim=1) # [bs, lq, d_v]
        s = self.gap(u.transpose(1, 2)).view(bs, d_v) # [bs, d_v]
        v = self.sk(s) # [bs, n_head*d_v]
        v = v.view(bs, n_head, d_v) # [bs, n_head, d_v]
        v = self.softmax(v) # [bs, n_head, d_v]
        v = v.unsqueeze(2) # [bs, n_head, 1, d_v]

        f = x * v.expand_as(x) # [bs, n_head, lq, d_v]
        return f

class MultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(
            self,
            n_head: int=8,
            d_model: int=512,
            d_k: int=64,
            d_v: int=64,
            dropout: float=0.1):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)

        # >> Selective heads
        if n_head > 1:
            self.spa = SPA(n_head=n_head, d_v=d_v)
            self.fc = nn.Linear(d_v, d_model, bias=False)
        else:
            self.fc = nn.Linear(n_head * d_v, d_model, bias=False)

        self.attention = ScaledDotProductAttention(temperature=d_k ** 0.5)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)


    def forward(self, q, k, v, mask=None):

        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)

        residual = q

        # Pass through the pre-attention projection: sz_b x lq x (n*dv)
        # Separate different heads: sz_b x lq x n x dv
        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        # >> Transpose for attention dot product: sz_b x n x lq x dv
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        if mask is not None:
            mask = mask.unsqueeze(1)   # For head axis broadcasting.

        q, attn = self.attention(q, k, v, mask=mask)

        if n_head > 1:
            # >> Selective heads
            q = self.spa(q) # [sz_b, n_head, lq, dv]
            q = q.sum(dim=1, keepdim=True) # [sz_b, 1, lq, dv]

        # >> Transpose to move the head dimension back: sz_b x lq x n x dv
        # >> Combine the last two dimensions to concatenate all the heads together: sz_b x lq x (n*dv)
        q = q.transpose(1, 2).contiguous().view(sz_b, len_q, -1)
        q = self.dropout(self.fc(q))
        q += residual

        q = self.layer_norm(q)

        return q, attn

class PositionwiseFeedForward(nn.Module):
    ''' A two-feed-forward-layer module '''

    def __init__(self, d_in, d_hid, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_in, d_hid) # position-wise
        self.w_2 = nn.Linear(d_hid, d_in) # position-wise
        self.layer_norm = nn.LayerNorm(d_in, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):

        residual = x

        x = self.w_2(F.relu(self.w_1(x)))
        x = self.dropout(x)
        x += residual

        x = self.layer_norm(x)

        return x
