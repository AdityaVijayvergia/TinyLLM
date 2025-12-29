# Rotary Positional Embeddings (RoPE)


---

## Overview

RoPE encodes positional information by **rotating** query and key vectors in the attention mechanism. Unlike additive positional embeddings, RoPE naturally captures **relative positions** through the rotation angles.

---

## Part 1: Precomputing Rotary Embeddings

The `_precompute_rotary_embeddings` method creates `cos` and `sin` lookup tables.

### Step 1: Create frequency bands for each channel pair

```python
channel_range = torch.arange(0, head_dim, 2, dtype=torch.float32, device=device)
inv_freq = 1.0 / (base ** (channel_range / head_dim))
```

- `channel_range` = `[0, 2, 4, ..., head_dim-2]` — every other index
- `inv_freq` computes **inverse frequencies** for each pair of dimensions:
  - Dimension pair 0: `1/(10000^(0/head_dim))` = 1.0 (fastest rotation)
  - Dimension pair 2: `1/(10000^(2/head_dim))` ≈ slower
  - Each subsequent pair rotates slower (geometric progression)

**Intuition:** Early dimensions rotate fast (encode fine-grained position), later dimensions rotate slow (encode coarse position).

### Step 2: Create position indices

```python
t = torch.arange(seq_len, dtype=torch.float32, device=device)
```

`t` = `[0, 1, 2, ..., seq_len-1]` — each position in the sequence.

### Step 3: Compute rotation angles

```python
freqs = torch.outer(t, inv_freq)
```

**Outer product** creates a `(seq_len, head_dim/2)` matrix where:
- `freqs[pos, dim_pair]` = `position × inverse_frequency`

This gives the **rotation angle (in radians)** for each (position, dimension-pair).

### Step 4: Compute cosine and sine

```python
cos, sin = freqs.cos(), freqs.sin()
```

These represent the **rotation matrices** decomposed into their components.

### Step 5: Reshape for broadcasting

```python
cos, sin = cos[None, :, None, :], sin[None, :, None, :]
```

Reshapes from `(seq_len, head_dim/2)` to `(1, seq_len, 1, head_dim/2)` for broadcasting over:
- Batch dimension
- Sequence position (matched)
- Attention heads
- Channel pairs (matched)

---

## Part 2: Applying Rotary Embeddings

The `apply_rotary_emb` function rotates Q and K vectors.

### Step 1: Split vector into two halves

```python
d = x.shape[3] // 2
x1, x2 = x[..., :d], x[..., d:]
```

Given input `x` of shape `(B, T, H, head_dim)`:
- `x1` = first half of channels (dims 0 to head_dim/2)
- `x2` = second half (dims head_dim/2 to head_dim)

### Step 2: Apply 2D rotation

```python
y1 = x1 * cos + x2 * sin
y2 = x1 * (-sin) + x2 * cos
```

This is the **2D rotation formula**:

```
┌     ┐   ┌              ┐ ┌     ┐
│ y₁  │   │ cos θ   sin θ│ │ x₁  │
│     │ = │              │ │     │
│ y₂  │   │-sin θ   cos θ│ │ x₂  │
└     ┘   └              ┘ └     ┘
```

Each pair of dimensions is treated as a **2D plane** that gets rotated by θ.

### Step 3: Reassemble

```python
out = torch.cat([y1, y2], 3)
```

---

## Part 3: Why This Encodes Relative Position

The key insight is in **attention score computation**:

When computing `Q @ K^T`, tokens at positions `i` and `j` have a dot product that depends on the **relative distance** `|i - j|`.

**Mathematical intuition:**
- Rotating Q by θᵢ and K by θⱼ
- The dot product `Q·K` depends on the angle difference θᵢ - θⱼ
- This difference only depends on position difference, not absolute positions

This is why RoPE naturally encodes **relative** positions.

---

## Benefits of RoPE

1. **Relative position encoding** — Better generalization
2. **Extrapolation** — Can handle longer sequences than seen during training
3. **No additional parameters** — Just uses precomputed cos/sin tables
4. **Efficient** — Simple element-wise operations

---

## References

- [RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864)
- [Rotary Position Embeddings (EleutherAI Blog)](https://blog.eleuther.ai/rotary-embeddings/)
