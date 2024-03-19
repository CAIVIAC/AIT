import torch
import torch.nn as nn
import torch.nn.functional as F


class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, attn_dropout=0.1, dist='softmax'):
        super().__init__()
        self.dist = dist
        self.temperature = temperature
        self.attn_dropout = attn_dropout
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q, k, v, mask=None):

        attn = torch.matmul(q / self.temperature, k.transpose(2, 3))

        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)

        if self.dist == 'softmax':
            attn = self.dropout(F.softmax(attn, dim=-1))
        elif self.dist == 'division':
            attn = self.dropout(attn / attn.size(-1))
        output = torch.matmul(attn, v)

        return output, attn

    def __repr__(self):
        tmpstr = self.__class__.__name__ + "("
        tmpstr += "temperature=" + str(self.temperature)
        tmpstr += ", attn_dropout=" + str(self.attn_dropout)
        tmpstr += ", dist=" + str(self.dist)
        tmpstr += ")"
        return tmpstr
