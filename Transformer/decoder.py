import torch.nn as nn
from utils import clones
from layers import SublayerConnection, LayerNorm

class DecoderLayer(nn.Module):
    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        super().__init__()
        self.self_attn = self_attn
        self.src_attn  = src_attn
        self.feed_forward = feed_forward
        self.sublayers = clones(SublayerConnection(size, dropout), 3)
        self.size = size

    def forward(self, x, memory, src_mask, tgt_mask):
        m = memory
        x = self.sublayers[0](x,
            lambda x: self.self_attn(x,x,x,tgt_mask))
        x = self.sublayers[1](x,
            lambda x: self.src_attn(x, m, m, src_mask))
        x = self.sublayers[2](x, self.feed_forward)
        return x


class Decoder(nn.Module):
    def __init__(self, layer, N):
        super().__init__()
        self.layers = clones(layer,N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, memory, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)
