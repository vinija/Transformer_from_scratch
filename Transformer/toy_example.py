import torch
from model import make_model
from utils import subsequent_mask


# ============================================================
# 1. Toy Vocabulary
# ============================================================

# ID → Token
itos = {
    0: "<pad>",
    1: "<s>",
    2: "A",
    3: "B",
    4: "C",
    5: "X",
    6: "Y",
    7: "Z"
}

# Token → ID
stoi = {tok: idx for idx, tok in itos.items()}

SRC_VOCAB = len(itos)   # 8 tokens
TGT_VOCAB = len(itos)   # same vocab for demo


# ============================================================
# 2. Build Model
# ============================================================

model = make_model(SRC_VOCAB, TGT_VOCAB, N=2, d_model=128, d_ff=512, h=4)
model.eval()

print("Model built successfully.")


# ============================================================
# 3. Create Toy Input
# ============================================================

# Example:
# src = <s> A B C
# tgt = <s> X Y Z

src = torch.tensor([[1, 2, 3, 4]])  # shape (1,4)
tgt = torch.tensor([[1, 5, 6, 7]])  # shape (1,4)

print("Source:", src)
print("Target:", tgt)


# ============================================================
# 4. Create Masks
# ============================================================

# Padding mask for src
src_mask = (src != 0).unsqueeze(-2)   # shape (1,1,4)

# Causal mask for tgt
tgt_mask = subsequent_mask(tgt.size(1)).unsqueeze(0)  # shape (1,4,4)

print("Source mask:", src_mask)
print("Target mask:", tgt_mask)


# ============================================================
# 5. Forward Pass
# ============================================================

out = model(src, tgt, src_mask, tgt_mask)

print("\nHidden output shape:", out.shape)
# Expected: (1, target_seq_len, d_model)
# Example: (1, 4, 128)


# ============================================================
# 6. Run Generator → log probs over vocab
# ============================================================

log_probs = model.generator(out)

print("Log-probs shape:", log_probs.shape)
# Expected: (1, 4, vocab_size = 8)


# ============================================================
# 7. Get Predictions
# ============================================================

pred_ids = torch.argmax(log_probs, dim=-1)  # shape (1,4)
pred_tokens = [itos[i.item()] for i in pred_ids[0]]

print("\nPredicted token IDs:", pred_ids.tolist())
print("Predicted tokens:", pred_tokens)


# ============================================================
# 8. (Optional) Pretty print
# ============================================================

print("\n--- Summary ---")
print("Input src:", [itos[i] for i in src[0].tolist()])
print("Input tgt:", [itos[i] for i in tgt[0].tolist()])
print("Predicted:", pred_tokens)
print("----------------")
