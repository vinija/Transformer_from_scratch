

# README.md

# Transformer From Scratch (PyTorch)

This project implements a **full Transformer** (encoder–decoder) architecture **from scratch**, following the *Annotated Transformer* by Harvard NLP.
Each component is hand-built using basic PyTorch primitives and separated into modular files.

```
      +------------------------+
      |   Encoder-Decoder      |
      |      Transformer       |
      +------------------------+
         ▲                │
         │ encode(src)    │ decode(memory, tgt)
         │                ▼
   +------------+     +-------------+
   |  Encoder   |     |   Decoder   |
   +------------+     +-------------+
```

---

# Project Structure

```
Transformer/
│
├── __init__.py
├── attention.py
├── decoder.py
├── embeddings.py
├── encoder.py
├── feedforward.py
├── generator.py
├── inference.py
├── layers.py
├── model.py
├── train_utils.py
├── utils.py
│
└── toy_example.py
└── README.md
```

**Visual overview of modules:**

```
attention.py      → Multi-head Attention + Scaled Dot-Product
feedforward.py    → Position-wise FFN
layers.py         → LayerNorm + Residual Sublayer wrapper
embeddings.py     → Token Embeddings + Positional Encoding
encoder.py        → EncoderLayer stack
decoder.py        → DecoderLayer stack
generator.py      → Final linear → log softmax
model.py          → Assembles full Transformer
utils.py          → clones(), subsequent_mask()
inference.py      → greedy decoding
toy_example.py    → demonstration script
```

---

# Installation

Clone the repository:

```bash
git clone https://github.com/<yourname>/Transformer_from_scratch.git
cd Transformer_from_scratch
```

Install PyTorch:

```bash
pip install torch
```

---

# Running the Toy Example

The toy example builds and executes a miniature Transformer with dummy sequences.

Run from the project root:

```bash
python toy_example.py
```

Expected output:

```
Output shape: torch.Size([1, 4, 128])
```

This verifies the entire Transformer pipeline is functioning.

---

# What the Toy Example Does

### 1. Creates a small Transformer model

```
make_model(
   src_vocab = 11,
   tgt_vocab = 11,
   N = 2 encoder layers
)
```

Tiny 2-layer encoder and decoder with 4-head attention.

---

### 2. Creates simple integer token sequences

```
src = [1, 2, 3, 4]
tgt = [1, 5, 6, 7]
```

These are embedded using **Embeddings + Positional Encoding**:

```
token → embedding
embedding + sinusoidal position → encoded input
```

---

### 3. Builds required masks

**Src mask:**

```
(batch=1, 1, seq_len)
[1 1 1 1]
```

**Tgt causal mask:**

```
subsequent_mask:
1 0 0 0
1 1 0 0
1 1 1 0
1 1 1 1
```

This prevents the decoder from “seeing the future.”

---

### 4. Runs a full forward pass

```
out = model(src, tgt, src_mask, tgt_mask)
```

Internally:

```
+-------------------+
|      Encoder      |
+-------------------+
       │
       ▼
   memory states
       │
       ▼
+-------------------+
|      Decoder      |
+-------------------+
       │
       ▼
final hidden states
```

Result shape:

```
(batch=1, tgt_len=4, d_model=128)
```

The output is **pre-softmax hidden states** which the `Generator` would turn into vocabulary probabilities.

---

# Building Your Own Model

### Create a model:

```python
from Transformer.model import make_model

model = make_model(src_vocab=11, tgt_vocab=11, N=2)
```

### Forward pass:

```python
out = model(src, tgt, src_mask, tgt_mask)
```

### Greedy decoding:

```
src → encoder → decoder (step by step)
```

```python
from Transformer.inference import greedy_decode

decoded = greedy_decode(model, src, src_mask, max_len=10, start_symbol=1)
print(decoded)
```


