import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn

AttentionCache = Tuple[torch.Tensor, torch.Tensor]
AttentionCacheSet = List[AttentionCache]


class MultiHeadAttention(nn.Module):
    """
    Multi-head attention allows the model to jointly attend to information from
    different representation subspaces at different positions.

    Args:
        hidden_dim: The number of hidden units.
        num_attn_heads: The number of attention heads.
        dropout_rate: The probability of dropping elements randomly. Default is `0.1`.
    """

    def __init__(self, hidden_dim: int, num_attn_heads: int, dropout_rate: float = 0.1):
        super().__init__()
        self.num_attn_heads = num_attn_heads

        self.linear_q = nn.Linear(hidden_dim, hidden_dim)
        self.linear_k = nn.Linear(hidden_dim, hidden_dim)
        self.linear_v = nn.Linear(hidden_dim, hidden_dim)
        self.linear_out = nn.Linear(hidden_dim, hidden_dim)

        self.dropout = nn.Dropout(dropout_rate)

    def forward(
        self,
        q: torch.Tensor,
        k: Optional[torch.Tensor] = None,
        v: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
        attn_cache: Optional[AttentionCache] = None,
    ) -> Tuple[torch.Tensor, AttentionCache]:
        q = self.linear_q(q)
        q = q.view(q.shape[:-1] + (self.num_attn_heads, -1)).transpose(-3, -2)

        if k is not None and v is not None:
            k, v = self.linear_k(k), self.linear_v(v)
            k = k.view(k.shape[:-1] + (self.num_attn_heads, -1)).transpose(-3, -2)
            v = v.view(v.shape[:-1] + (self.num_attn_heads, -1)).transpose(-3, -2)

            # Concatenate the attention cache to the projected key and value tensors.
            if attn_cache is not None:
                k = torch.cat((attn_cache[0], k), dim=-2)
                v = torch.cat((attn_cache[1], v), dim=-2)
        else:
            if attn_cache is None:
                raise ValueError(
                    "`attn_cache` must not be None when input key and value are None."
                )

            # Use previously calculated attention cache.
            k, v = attn_cache

        # Calculate an attention matrix.
        x = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(k.size(-1))

        if attn_mask is not None:
            x.masked_fill_(
                attn_mask.unsqueeze(-3), -1e4 if x.dtype == torch.float16 else -1e9
            )

        # Attend to the value vectors and merge the multi-heads into a single head.
        x = torch.matmul(self.dropout(x.softmax(-1)), v)
        x = x.transpose(-3, -2).contiguous().view(q.shape[:-3] + (q.size(-2), -1))

        return self.dropout(self.linear_out(x)), (k, v)
