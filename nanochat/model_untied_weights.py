"""
Architecture Overview:
1. Embedding: Token IDs -> Vectors (wte)
2. Stack of Blocks (Repeated L times):
   - RMSNorm
   - Attention (Mixing info between tokens)
   - RMSNorm
   - MLP (Processing info within a token)
3. Final Norm
4. LMHead: Vectors -> Logits (Probabilities)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
import math
from functools import partial

from nanochat.common import get_dist_info, print0
from nanochat.muon import Muon, DistMuon

@dataclass
class GPTConfig:
    """
    Hyperparameters for the model.
    """
    # ┌─────────────────────────────────────────────────────────┐
    # │           321M CONVERSATIONAL MODEL                     │
    # ├─────────────────────────────────────────────────────────┤
    # │  hidden_dim:        1024                                │
    # │  layers:            20                                  │
    # │  heads:             8                                   │
    # │  head_dim:          128                                 │
    # │  mlp_ratio:         3x                                  │
    # │  vocab_size:        32K                                 │
    # │  context_length:    1024                                │
    # │  embedding:         tied (input = output projection)    │
    # │  activation:        relu squared                        │
    # │  position encoding: RoPE                                │
    # ├─────────────────────────────────────────────────────────┤
    # │  TOTAL PARAMETERS:  243,269,632                         │
    # └─────────────────────────────────────────────────────────┘
    # No KV cache
    # No GQA

    hidden_dim: int = 1024 # hidden dimension
    n_layers: int = 20 # May need to reduce to 22 or 20
    n_heads: int = 8 # head dimension = hidden_dim / n_heads = 128
    mlp_ratio: int = 3
    vocab_size: int = 32*1024
    sequence_len: int = 1024
    n_kv_head: int = 8


def norm(x):
    """
    RMSNorm (Root Mean Square Layer Normalization).
    Used to stabilize training by normalizing activation magnitudes.
    """
    # Purely functional rmsnorm with no learnable params
    return F.rms_norm(x, (x.size(-1),))


def apply_rotatory_positional_encoding(x, cos, sin):
    """
    Applies Rotary Positional Embeddings (RoPE).
    Rotates the query and key vectors to encode relative positions.
    """
    assert x.ndim == 4  # multihead attention
    d = x.shape[3] // 2
    x1, x2 = x[..., :d], x[..., d:] # split up last time into two halves
    y1 = x1 * cos + x2 * sin # rotate pairs of dims
    y2 = x1 * (-sin) + x2 * cos
    out = torch.cat([y1, y2], 3) # re-assemble
    out = out.to(x.dtype) # ensure input/output dtypes match
    return out

# device_type = 'cuda' if torch.cuda.is_available() else 'cpu'

class MultiHeadAttention(nn.Module):
    """
    Multi-Head Causal Self Attention.

    1. Projects input to Q, K, V.
    2. Applies RoPE to Q, K for position info.
    3. Computes attention scores (Q @ K) to see how much each token cares about others. Aggregates values (V) based on scores.
    4. Projects output to mix information across heads.
    """
    def __init__(self, config: GPTConfig, layer_idx):
        super().__init__()
        self.layer_idx = layer_idx
        self.n_heads = config.n_heads
        self.hidden_dim = config.hidden_dim
        self.head_dim = config.hidden_dim // config.n_heads

        # Linear projections for Query, Key, Value
        self.key = nn.Linear(self.hidden_dim, self.head_dim * self.n_heads, bias=False)
        self.query = nn.Linear(self.hidden_dim, self.head_dim * self.n_heads, bias=False)
        self.value = nn.Linear(self.hidden_dim, self.head_dim * self.n_heads, bias=False)

        # Output projection ("o"): mixes results from all heads back into n_embd
        self.proj = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)

    def forward(self, x, cos_sin, kv_cache = None) -> torch.Tensor:
        B, T, C = x.size()

        # 1. Projects input to Q, K, V.
        # reshape to (B, T, n_heads, head_dim)
        k = self.key(x).view(B, T, self.n_heads, self.head_dim)
        q = self.query(x).view(B, T, self.n_heads, self.head_dim)
        v = self.value(x).view(B, T, self.n_heads, self.head_dim)

        # 2. Applies RoPE to Q, K for position info.
        cos, sin = cos_sin
        k, q = apply_rotatory_positional_encoding(k, cos, sin), apply_rotatory_positional_encoding(q, cos, sin)

        # 3. Computes attention scores (Q @ K) to see how much each token cares about others.
        q, k = norm(q), norm(k) # QK norm

        # make head be batch dim, i.e. (B, T, n_heads, head_dim) -> (B, n_heads, T, head_dim)
        # We are making the n_heads into a batch dimension so pytorch treats it as batches and
        # applies the attention function on each head separately in parallel
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        # TODO: Simplify logic. Check if below 1 line can replace longer logic? 
        # y = F.scaled_dot_product_attention(q, k, v, is_causal=True)

        # Longer logic
        # Apply KV cache: insert current k,v into cache, get the full view so far
        if kv_cache is not None:
            k, v = kv_cache.insert_kv(self.layer_idx, k, v)
        Tq = q.size(2) # number of queries in this forward pass
        Tk = k.size(2) # number of keys/values in total (in the cache + current forward pass)

        # Attention: queries attend to keys/values autoregressively. A few cases to handle:
        if kv_cache is None or Tq == Tk:
            # During training (no KV cache), attend as usual with causal attention
            # And even if there is KV cache, we can still use this simple version when Tq == Tk
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        elif Tq == 1:
            # During inference but with a single query in this forward pass:
            # The query has to attend to all the keys/values in the cache
            y = F.scaled_dot_product_attention(q, k, v, is_causal=False)
        else:
            # During inference AND we have a chunk of queries in this forward pass:
            # First, each query attends to all the cached keys/values (i.e. full prefix)
            attn_mask = torch.zeros((Tq, Tk), dtype=torch.bool, device=q.device) # True = keep, False = mask
            prefix_len = Tk - Tq
            attn_mask[:, :prefix_len] = True
            # Then, causal attention within this chunk
            attn_mask[:, prefix_len:] = torch.tril(torch.ones((Tq, Tq), dtype=torch.bool, device=q.device))
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)

        # Re-assemble the heads side by side and project back to residual stream
        y = y.transpose(1, 2).contiguous().view(B, T, -1)

        # 4. Projects output to mix information across heads.
        y = self.proj(y)
        return y


class FeedForward(nn.Module):
    """
    Feed Forward Network (MLP).
    Processes each token independently (no mixing between tokens).
    Structure: Expand -> ReLU^2 -> Contract
    """
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.proj_up = nn.Linear(config.hidden_dim, config.hidden_dim * config.mlp_ratio, bias=False)
        self.proj_down = nn.Linear(config.hidden_dim * config.mlp_ratio, config.hidden_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj_up(x)
        x = F.relu(x).square()
        # Check if swiglu is better for 3 x hidden_dim -
        # gelu and silu are alternatives but difference seems marginal so sticking with relu^2
        x = self.proj_down(x)
        return x


class TransformerBlock(nn.Module):
    """
    A single Transformer Block.
    Contains:
    1. Attention (Communication)
    2. MLP (Computation)
    Both use Residual Connections (x + ...) and Pre-Norm.
    """
    def __init__(self, config: GPTConfig, layer_idx):
        super().__init__()
        self.attn = MultiHeadAttention(config, layer_idx)
        self.ff = FeedForward(config)

    def forward(self, x: torch.Tensor, cos_sin: torch.Tensor, kv_cache = None) -> torch.Tensor:
        # Attention with residual connection
        x = x + self.attn(norm(x), cos_sin, kv_cache)
        # MLP with residual connection
        x = x + self.ff(norm(x))
        return x


class GPT(nn.Module):
    """
    The full GPT model.
    Contains:
    1. Token Embedding
    2. Transformer Blocks (stacked)
    3. Final Normalization
    4. LM Head - Tied weights with token embedding
    """
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
        self.max_seq_len = config.sequence_len
        self.token_embedding = nn.Embedding(config.vocab_size, config.hidden_dim)
        self.blocks = nn.ModuleList([TransformerBlock(config, layer_idx) for layer_idx in range(config.n_layers)])
        self.lm_head = nn.Linear(config.hidden_dim, config.vocab_size, bias=False)
        # self.lm_head.weight = self.token_embedding.weight

        self.rotary_seq_len = config.sequence_len * 20 # 10X over-compute
        # Why 10x? This provides a generous buffer for inference/generation, allowing the model
        # to generate sequences longer than its training length without recomputing embeddings.
        # Note: While the embeddings support 10x length, the model's quality degrades beyond ~1.5-2x
        # the training length due to unseen attention patterns. This buffer is for convenience,
        # not an expectation of good performance at 10x length. Memory cost is negligible.

        head_dim = config.hidden_dim // config.n_heads
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim)
        self.register_buffer("cos", cos, persistent=False) # persistent=False means it's not saved to the checkpoint
        self.register_buffer("sin", sin, persistent=False)

    def forward(self, idx, targets=None, kv_cache=None, loss_reduction="mean") -> torch.Tensor:
        T = idx.shape[1]
        cos_sin = self.cos[:, :T], self.sin[:, :T] # truncate cache to current sequence length

        x = self.token_embedding(idx)
        x = norm(x)
        for block in self.blocks:
            x = block(x, cos_sin, kv_cache)
        x = norm(x)

        softcap = 15 # smoothly cap the logits to the range [-softcap, softcap]
        logits = self.lm_head(x)
        logits = logits.float() # switch to fp32 for logit softcap and loss computation
        logits = softcap * torch.tanh(logits / softcap) # squash the logits

        if targets is not None:
            # training: given the targets, compute and return the loss
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1, reduction=loss_reduction)
            return loss
        else:
            # inference: just return the logits directly
            return logits

    def init_weights(self):
        """
        Initialize the full model in this one function for maximum clarity.

        embedding:     normal, std=1.0
        for each block:
            attn.c_q:        uniform, std=1/sqrt(n_embd)
            attn.c_k:        uniform, std=1/sqrt(n_embd)
            attn.c_v:        uniform, std=1/sqrt(n_embd)
            attn.c_proj:     zeros
            mlp.c_fc:        uniform, std=1/sqrt(n_embd)
            mlp.c_proj:      zeros
        """

        """
        Explanation:
        The initialization logic deviates from PyTorch defaults (Kaiming defaults) to improve training 
        stability and convergence for deep Transformers.
        
        Key Differences:
        1. Zero Initialization for Output Projections (c_proj):
           - Function: Sets the weights of the final linear layer in each block to zero.
           - Why: This ensures that at initialization, the residual blocks contribute nothing to the 
             residual stream (y = x + 0). The model effectively starts as an identity function, allowing 
             unimpeded gradient flow from top to bottom. This prevents vanishing/exploding gradients 
             and provides a stable starting point for the model to gradually learn features.

        2. Zero Initialization for LM Head:
           - Function: Sets the classifier weights to zero.
           - Why: Ensures all logits are initially zero, leading to a uniform probability distribution (1/V) 
             for the next token. This minimizes the initial loss to exactly log(V) and prevents the model 
             from starting with random biases towards arbitrary tokens.

        Custom initialization for Linear and Embedding layers.
        
        1. Controlled Variance (Linear Layers):
           - Formula: std = 1 / sqrt(fan_in) * min(1, sqrt(fan_out / fan_in))
           - Why: Standard Kaiming init often leads to activation variance that grows with depth in 
             Transformers. This custom initialization (ref: https://arxiv.org/pdf/2310.17813) stabilizes 
             activation variance across layers, specifically accounting for the network width.

        2. Unit Variance (Embeddings):
           - Function: Normal distribution with std=1.0.
           - Why: Ensures strong initial signal strength before it enters the first normalization layer.
        """
        # Embedding
        torch.nn.init.normal_(self.token_embedding.weight, mean=0.0, std=1.0)
        # self.lm_head.weight = self.token_embedding.weight
        torch.nn.init.normal_(self.lm_head.weight, mean=0.0, std=0.001)

        # Transformer blocks: uniform init with bound = sqrt(3) * std (same standard deviation as normal)
        n_embd = self.config.hidden_dim
        s = 3**0.5 * n_embd**-0.5 # sqrt(3) multiplier makes sure Uniform achieves the same std as Normal

        # Zero out the output projections of the blocks
        for block in self.blocks:
            torch.nn.init.zeros_(block.ff.proj_down.weight)
            torch.nn.init.zeros_(block.attn.proj.weight)
            torch.nn.init.uniform_(block.attn.query.weight, -s, s) # weights use Uniform to avoid outliers
            torch.nn.init.uniform_(block.attn.key.weight, -s, s)
            torch.nn.init.uniform_(block.attn.value.weight, -s, s)
            torch.nn.init.uniform_(block.ff.proj_up.weight, -s, s)

        # init the rotary embeddings
        head_dim = self.config.hidden_dim // self.config.n_heads
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim)
        self.cos, self.sin = cos, sin

        # Cast the embeddings from fp32 to bf16: optim can tolerate it and it saves memory: both in the model and the activations
        if self.token_embedding.weight.device.type == "cuda":
            self.token_embedding.to(dtype=torch.bfloat16)
        #     self.lm_head.weight = self.token_embedding.weight

    def _precompute_rotary_embeddings(self, seq_len, head_dim, base=10000, device=None):
        # autodetect the device from model embeddings
        if device is None:
            device = self.token_embedding.weight.device
        # stride the channels
        channel_range = torch.arange(0, head_dim, 2, dtype=torch.float32, device=device)
        inv_freq = 1.0 / (base ** (channel_range / head_dim))
        # stride the time steps
        t = torch.arange(seq_len, dtype=torch.float32, device=device)
        # calculate the rotation frequencies at each (time, channel) pair
        freqs = torch.outer(t, inv_freq)
        cos, sin = freqs.cos(), freqs.sin()
        cos, sin = cos.bfloat16(), sin.bfloat16() # keep them in bfloat16
        cos, sin = cos[None, :, None, :], sin[None, :, None, :] # add batch and head dims for later broadcasting
        return cos, sin

    def setup_optimizers(self, unembedding_lr=0.004, embedding_lr=0.2, matrix_lr=0.02, weight_decay=0.0):
        """
        Sets up the optimizers.
        Uses AdamW for embeddings/head and Muon for internal linear layers.

        Detailed Explanation of Hybrid Strategy:
        ----------------------------------------
        We use two different optimizers because different parts of the Transformer have different
        geometric properties and optimization landscapes.

        1. Muon (for internal 2D matrices):
           - Applied to: Attention projections (c_q, c_k, c_v, c_proj) and MLP weights (c_fc, c_proj).
           - Mechanism: Muon forces weight *updates* to be orthogonal. In linear algebra, orthogonal
             transformations (like rotation or reflection) preserve the magnitude (norm) of the vector
             they act on.
           - Benefit: Deep networks suffer from vanishing/exploding gradients because signals get
             scaled up or down at every layer. By forcing updates to be orthogonal, Muon ensures
             signals propagate through the network without exploding in magnitude, allowing for
             much faster and more stable training of deep layers.

        2. AdamW (for embeddings & head):
           - Applied to: Token embeddings (wte) and the final output head (lm_head).
           - Reason: These parameters are not dense 2D matrices in the same sense (embeddings are
             lookup tables). The concept of "orthogonal updates" is mathematically ill-defined or
             harmful for vectors/lookups. AdamW is ideal here as it adapts learning rates per-parameter
             based on update frequency (handling the sparse nature of token updates).

        Do they conflict?
        No. Both optimizers step in directions derived from the same global loss gradient, so they
        optimize the same function. The risk is learning speed mismatch (one part learning faster
        than the other), which we handle by manually scaling the AdamW learning rate below.
        """
        model_dim = self.config.hidden_dim
        # Separate out all parameters into 3 groups (matrix, embedding, lm_head)
        matrix_params = list(self.blocks.parameters())
        embedding_params = list(self.token_embedding.parameters())
        lm_head_params = list(self.lm_head.parameters())
        print(f"matrix parameters: {len(matrix_params)}, lm_head_parameters: {len(lm_head_params)}, embedding parameters: {len(embedding_params)}, total parameters: {len(list(self.parameters()))}")
        assert len(list(self.parameters())) == len(matrix_params) + len(embedding_params) + len(lm_head_params)

        # Create the AdamW optimizer for the embedding
        # Scale the LR for the AdamW parameters by ∝1/√dmodel (having tuned the LRs for 768 dim model)
        dmodel_lr_scale = (model_dim / 768) ** -0.5
        # if rank == 0:
        print(f"Scaling the LR for the AdamW parameters ∝1/√({model_dim}/768) = {dmodel_lr_scale:.6f}")
        adam_groups = [
            dict(params=lm_head_params, lr=unembedding_lr * dmodel_lr_scale),
            dict(params=embedding_params, lr=embedding_lr * dmodel_lr_scale),
        ]
        adamw_kwargs = dict(betas=(0.8, 0.95), eps=1e-10, weight_decay=weight_decay)
        AdamWFactory = partial(torch.optim.AdamW, fused=True)
        adamw_optimizer = AdamWFactory(adam_groups, **adamw_kwargs)

        # Create the Muon optimizer for the linear layers
        muon_kwargs = dict(lr=matrix_lr, momentum=0.95)
        muon_optimizer = Muon(matrix_params, **muon_kwargs)

        # Combine the two optimizers into one list
        optimizers = [adamw_optimizer, muon_optimizer]
        for opt in optimizers:
            for group in opt.param_groups:
                group["initial_lr"] = group["lr"]
        return optimizers

    def get_device(self):
        return self.token_embedding.weight.device

    def estimate_flops(self):
        """ Return the estimated FLOPs per token for the model. Ref: https://arxiv.org/abs/2204.02311 """
        nparams = sum(p.numel() for p in self.parameters())
        nparams_embedding = self.token_embedding.weight.numel()
        l, h, q, t = self.config.n_layers, self.config.n_heads, self.config.hidden_dim // self.config.n_heads, self.config.sequence_len
        num_flops_per_token = 6 * (nparams - nparams_embedding) + 12 * l * h * q * t
        return num_flops_per_token
    
    @torch.inference_mode()
    def generate(self, tokens, max_tokens, temperature=1.0, top_k=None, seed=42):
        """
        Naive autoregressive streaming inference.
        To make it super simple, let's assume:
        - batch size is 1
        - ids and the yielded tokens are simple Python lists and ints
        """
        assert isinstance(tokens, list)
        device = self.get_device()
        rng = None
        if temperature > 0:
            rng = torch.Generator(device=device)
            rng.manual_seed(seed)
        ids = torch.tensor([tokens], dtype=torch.long, device=device) # add batch dim
        for _ in range(max_tokens):
            logits = self.forward(ids) # (B, T, vocab_size)
            logits = logits[:, -1, :] # (B, vocab_size)
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            if temperature > 0:
                logits = logits / temperature
                probs = F.softmax(logits, dim=-1)
                next_ids = torch.multinomial(probs, num_samples=1, generator=rng)
            else:
                next_ids = torch.argmax(logits, dim=-1, keepdim=True)
            ids = torch.cat((ids, next_ids), dim=1)
            token = next_ids.item()
            yield token