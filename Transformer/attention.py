import torch
import torch.nn.functional as F
import torch.nn as nn
import math
from utils import clones

def attention(query, key, value, mask=None, dropout=None):
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2,-1)) / math.sqrt(d_k)

    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)

    p_attn = F.softmax(scores, dim=-1)

    if dropout is not None:
        p_attn = dropout(p_attn)

    return torch.matmul(p_attn, value), p_attn


class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        super().__init__()
        assert d_model % h == 0
        self.h = h
        self.d_k = d_model // h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, mask=None):
        if mask is not None:
            mask = mask.unsqueeze(1)

        nbatches = query.size(0)

        query, key, value = [
            linear(x).view(nbatches, -1, self.h, self.d_k).transpose(1,2)
            for linear, x in zip(self.linears, (query,key,value))
        ]

        x, attn = attention(query, key, value, mask, dropout=self.dropout)

        x = x.transpose(1,2).contiguous().view(nbatches, -1, self.h * self.d_k)

        return self.linears[-1](x)
