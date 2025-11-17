import torch.nn as nn
import copy
from encoder import Encoder, EncoderLayer
from decoder import Decoder, DecoderLayer
from attention import MultiHeadedAttention
from embeddings import Embeddings, PositionalEncoding
from feedforward import PositionwiseFeedForward
from generator import Generator   # if separated
# or inline it here

class EncoderDecoder(nn.Module):
    def __init__(self, encoder, decoder, src_embed, tgt_embed, generator):
        super().__init__()
        self.encoder   = encoder
        self.decoder   = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator

    def encode(self, src, src_mask):
        return self.encoder(self.src_embed(src), src_mask)

    def decode(self, memory, src_mask, tgt, tgt_mask):
        return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)

    def forward(self, src, tgt, src_mask, tgt_mask):
        memory = self.encode(src, src_mask)
        return self.decode(memory, src_mask, tgt, tgt_mask)


def make_model(src_vocab, tgt_vocab, N=2,  # small for toy example
               d_model=128, d_ff=512, h=4, dropout=0.1):

    c = copy.deepcopy
    attn = MultiHeadedAttention(h, d_model, dropout)
    ff   = PositionwiseFeedForward(d_model, d_ff, dropout)
    pos  = PositionalEncoding(d_model, dropout)

    model = EncoderDecoder(
        Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N),
        Decoder(DecoderLayer(d_model, c(attn), c(attn), c(ff), dropout), N),
        nn.Sequential(Embeddings(d_model, src_vocab), c(pos)),
        nn.Sequential(Embeddings(d_model, tgt_vocab), c(pos)),
        Generator(d_model, tgt_vocab)
    )

    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    return model
