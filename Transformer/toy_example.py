import torch
from model import make_model
from utils import subsequent_mask

# tiny vocab
SRC_VOCAB = 11
TGT_VOCAB = 11

model = make_model(SRC_VOCAB, TGT_VOCAB, N=2)

src = torch.tensor([[1,2,3,4]])
tgt = torch.tensor([[1,5,6,7]])

src_mask = (src != 0).unsqueeze(-2)
tgt_mask = subsequent_mask(tgt.size(1)).unsqueeze(0)

out = model(src, tgt, src_mask, tgt_mask)

print("Output shape:", out.shape)
