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

@dataclass
class GPTConfig:
    """
    Hyperparameters for the model.
    """
    # ┌─────────────────────────────────────────────────────────┐
    # │           321M CONVERSATIONAL MODEL                     │
    # ├─────────────────────────────────────────────────────────┤
    # │  hidden_dim:        1024                                │
    # │  layers:            24                                  │
    # │  heads:             8                                   │
    # │  head_dim:          128                                 │
    # │  mlp_ratio:         3x                                  │
    # │  vocab_size:        65,536 (64K)                        │
    # │  context_length:    2048                                │
    # │  embedding:         tied (input = output projection)    │
    # │  activation:        relu squared    │
    # │  position encoding: RoPE                                │
    # ├─────────────────────────────────────────────────────────┤
    # │  TOTAL PARAMETERS:  ~321 Million                        │
    # └─────────────────────────────────────────────────────────┘
    hidden_dim: int = 1024 # hidden dimension  
    n_layers: int = 24 # May need to reduce to 22 or 20
    n_heads: int = 8 # head dimension = hidden_dim / n_heads = 128
    mlp_ratio: int = 3
    vocab_size: int = 64*1024
    sequence_len: int = 2048


# def apply_rotatory_positional_encoding(x: torch.Tensor, head_dim: int) -> torch.Tensor:
#     """
#     Apply RoPE to the input tensor.
#     """
#     B, T, C = x.size()
#     # TODO: Implement RoPE
    

class MultiHeadAttention(nn.Module):
    """
    Multi-Head Causal Self Attention.
    
    1. Projects input to Q, K, V.
    2. Applies RoPE to Q, K for position info.
    3. Computes attention scores (Q @ K) to see how much each token cares about others. Aggregates values (V) based on scores.
    4. Projects output to mix information across heads.
    """
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.n_heads = config.n_heads
        self.hidden_dim = config.hidden_dim
        self.head_dim = config.hidden_dim // config.n_heads

        # Linear projections for Query, Key, Value
        self.key = nn.Linear(self.hidden_dim, self.head_dim * self.n_heads, bias=False)
        self.query = nn.Linear(self.hidden_dim, self.head_dim * self.n_heads, bias=False)
        self.value = nn.Linear(self.hidden_dim, self.head_dim * self.n_heads, bias=False)

        # Output projection ("o"): mixes results from all heads back into n_embd
        self.proj = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.size()

        # 1. Projects input to Q, K, V.
        k = self.key(x).view(B, T, self.n_heads, self.head_dim)
        q = self.query(x).view(B, T, self.n_heads, self.head_dim)
        v = self.value(x).view(B, T, self.n_heads, self.head_dim)
        
        # 2. Applies RoPE to Q, K for position info.
        # TODO: Implement RoPE
        # k = apply_rotatory_positional_encoding(k, self.head_dim)
        # q = apply_rotatory_positional_encoding(q, self.head_dim)

        # 3. Computes attention scores (Q @ K) to see how much each token cares about others.
        q, k = norm(q), norm(k) # QK norm
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2) # make head be batch dim, i.e. (B, T, H, D) -> (B, H, T, D)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)

        # Re-assemble the heads side by side and project back to residual stream
        y = y.transpose(1, 2).contiguous().view(B, T, C)

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
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.attn = MultiHeadAttention(config)
        self.ff = FeedForward(config)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Attention with residual connection
        x = self.attn(norm(x)) + x
        # MLP with residual connection
        x = self.ff(norm(x)) + x
        return x
