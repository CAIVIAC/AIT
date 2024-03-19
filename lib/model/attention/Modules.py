import torch
import torch.nn as nn
import torch.nn.functional as F


class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q, k, v, mask=None, ch_att=False):
        """
        :param q, k, v
            - size: [bs, n_head, T, C]
        """

        if ch_att:
            attn = torch.matmul(q.transpose(2, 3), k) # [bs, n_head, C^q, C^k]
        else:
            attn = torch.matmul(q / self.temperature, k.transpose(2, 3))
            if mask is not None:
                attn = attn.masked_fill(mask == 0, -1e9)

        attn = self.dropout(F.softmax(attn, dim=-1))
        if ch_att:
            output = torch.matmul(attn, v.transpose(2, 3)) # [bs, n_head, C^q, T]
            output = output.transpose(2, 3) # [bs, n_head, T, C]
        else:
            output = torch.matmul(attn, v)

        return output, attn
