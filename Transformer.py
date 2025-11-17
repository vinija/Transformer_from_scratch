import torch.nn as nn
import copy
import numpy as np
import torch

###
# 1. Utility Functions
# 1.1 clones (deep copy module list)
# 1.2 subsequent_mask (future mask for decoder)

# 2. Layer Normalization
# 2.1 LayerNorm

# 3. SublayerConnection (Residual + Norm)
# 3.1 SublayerConnection module

# 4. Attention Mechanisms
# 4.1 Scaled dot-product attention
# 4.2 Multi-headed attention

# 5. Position-wise Feed Forward Network
# 5.1 PositionwiseFeedForward

# 6. Embedding and Positional Encoding
# 6.1 Token Embeddings
# 6.2 PositionalEncoding

# 7. Encoder Components
# 7.1 EncoderLayer
# 7.2 Encoder (stack)

# 8. Decoder Components
# 8.1 DecoderLayer
# 8.2 Decoder (stack)

# 9. Full Encoder-Decoder Wrapper
# 9.1 EncoderDecoder class

# 10. Generator (final linear + log softmax)

# 11. Model Assembly
# 11.1 make_model

# 12. Training Components
# 12.1 Batch (mask creation)
# 12.2 Label smoothing
# 12.3 Loss computation wrapper
# 12.4 Optimizer (NoamOpt)
# 12.5 Training loop

# 13. Inference
# 13.1 Greedy decoding
###


#Section 1 Utility Functions
def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

# mask for decoder to not look at future tokens
def subsequent_mask(size):
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0

#Section 2 Layer Normalization
class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-6):
        super().__init__()
        self.a = nn.Parameter(torch.ones(features))
        self.b = nn.Parameter(torch.zeros(features))
        self.eps = eps
    
    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a * (x - mean) / (std + self.eps) + self.b

#Section 3 SublayerConnection
class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """
    def __init__(self, size, dropout):
        super().__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))

#Section 4 Attention Mechanism - scaled dot-product attention
def attention(query, key, value, mask=None, dropout=None):
    d_k = query.size(-1)

    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)

    if mask is not None:
        scores = scores.masked_full(mask == 0, -1e9)
    
    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)

    return torch.matmul(p_attn, value), p_attn

# Scaled dot-product attention is the core mechanism; multi-head attention runs multiple instances in parallel to capture diverse patterns.
class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        super().__init__()
        assert d_model % h == 0
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, query, key, value, mask=None):
        if mask is not None:
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        query, key, value = [
            lin(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
            for lin, x in zip(self.linears, (query, key, value))
        ]

        x, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout)

        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)
    
# Position-wise Feed Forward Network
class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))

# Embedding and Positional Encoding
#Token Embeddings
class Embeddings(nn.Module):
    def __init__(self, d_model, vocab):
        super().__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model
    
    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)

# Positional Encoding
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

#Encoder Components
class EncoderLayer(nn.Module):
    def __init__(self, size, self_attn, feed_forward, dropout):
        super().__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
    
    def forward(self, x, mask):
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)

class Encoder(nn.Module):
    def __init__(self, layer, N):
        super().__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)
    
    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)

#Decoder Components
class DecoderLayer(nn.Module):
    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        super().__init__()
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 3)
    
    def forward(self, x, memory, src_mask, tgt_mask):
        m = memory
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask))
        return self.sublayer[2](x, self.feed_forward)

class Decoder(nn.Module):
    def __init__(self, layer, N):
        super().__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)
    
    def forward(self, x, memory, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)

#Encoder-Decoder Components
class EncoderDecoder(nn.Module):
    def __init__(self, encoder, decoder, src_embed, tgt_embed, generator):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator
    
    def forward(self, src, tgt, src_mask, tgt_mask):
        memory = self.encoder(src, src_mask)
        return self.decoder(memory, tgt, src_mask, tgt_mask)

#Generator
class Generator(nn.Module):
    def __init__(self, d_model, vocab):
        super().__init__()
        self.proj = nn.Linear(d_model, vocab)
    
    def forward(self, x):
        return F.log_softmax(self.proj(x), dim=-1)

#Model Assembly
def make_model(src_vocab, tgt_vocab, N=6, d_model=512, d_ff=2048, h=8, dropout=0.1):
    c = copy.deepcopy
    attn = MultiHeadedAttention(h, d_model)
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    encoder = Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N)
    decoder = Decoder(DecoderLayer(d_model, c(attn), c(attn), c(ff), dropout), N)
    src_embed = Embeddings(d_model, src_vocab)
    tgt_embed = Embeddings(d_model, tgt_vocab)
    src_embed = PositionalEncoding(d_model, dropout)
    tgt_embed = PositionalEncoding(d_model, dropout)
    return EncoderDecoder(encoder, decoder, src_embed, tgt_embed, Generator(d_model, tgt_vocab))

#Training Components
def run_epoch(data, model, loss_compute):
    start = time.time()
    total_tokens = 0
    total_loss = 0
    tokens = 0
    for batch, i in enumerate(range(0, data.size(1) - 1, bptt)):
        data, targets = get_batch(data, i)
        size, loss = loss_compute(model.generate(data, targets), targets, batch_size)
        total_loss += loss
        total_tokens += size
        tokens += size
        if i % 50 == 1:
            elapsed = time.time() - start
            print("Epoch Step: %d Loss: %f Tokens per sec: %f" % (i, loss / size, tokens / elapsed))
            start = time.time()
            tokens = 0
    return total_loss / total_tokens

#Loss computation wrapper
class SimpleLossCompute:
    def __init__(self, generator, criterion, opt=None):
        self.generator = generator
        self.criterion = criterion
        self.opt = opt

    def __call__(self, x, y, norm):
        x = self.generator(x)
        loss = self.criterion(x.view(-1, x.size(-1)), y.view(-1)) / norm
        if self.opt is not None:
            loss.backward()
            self.opt.step()
            self.opt.zero_grad()
        return loss.data[0] * norm

# Greedy decoding
def greedy_decode(model, src, src_mask, max_len, start_symbol):
    memory = model.encode(src, src_mask)
    ys = torch.zeros(1, 1).fill_(start_symbol).type_as(src.data)
    for i in range(max_len - 1):
        out = model.decode(memory, ys, src_mask, subsequent_mask(ys.size(1)).type_as(src.data))
        prob = model.generate(out[:, -1])
        next_word = torch.argmax(prob, dim=-1)
        next_word = next_word.data[0]
        ys = torch.cat([ys, torch.zeros(1, 1).type_as(src.data).fill_(next_word)], dim=1)
    return ys
