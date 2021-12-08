import torch
import torch.nn as nn
import torch.nn.functional as F

__author__ = "Yu-Hsiang Huang"

class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(
            self,
            temperature: float,
            attn_dropout: float=0.1):
        super().__init__()
        self.temperature = temperature
        self.attn_dropout = attn_dropout
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q, k, v, mask=None):

        attn = torch.matmul(q / self.temperature, k.transpose(2, 3))

        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)

        attn = self.dropout(F.softmax(attn, dim=-1))
        output = torch.matmul(attn, v)

        return output, attn

    def __repr__(self):
        tmpstr = self.__class__.__name__ + "("
        tmpstr += "temperature=" + str(self.temperature)
        tmpstr += ", attn_dropout=" + str(self.attn_dropout)
        tmpstr += ")"
        return tmpstr
