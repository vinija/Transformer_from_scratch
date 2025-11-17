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

