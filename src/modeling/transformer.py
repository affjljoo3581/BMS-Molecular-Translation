from typing import Optional, Tuple

import torch
import torch.nn as nn

from modeling.attention import AttentionCache, AttentionCacheSet, MultiHeadAttention
from modeling.feedforward import PositionwiseFeedForward

try:
    from apex.normalization import FusedLayerNorm as LayerNorm
except ModuleNotFoundError:
    from torch.nn import LayerNorm


class TransformerLayer(nn.Module):
    """
    An implementation of Transformer encoder and decoder layers, including
    self-attention, encoder-decoder attention and positionwise feed-forward layers.

    Args:
        hidden_dim: The number of hidden units.
        num_attn_heads: The number of attention heads.
        expansion_ratio: The expansion ratio of the bottleneck layer. Default is `4`.
        dropout_rate: The probability of dropping elements randomly. Default is `0.1`.
        use_encoder_attn: The boolean that determines whether to use encoder attention
            layer or not. Default is `False`.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_attn_heads: int,
        expansion_ratio: float = 4,
        dropout_rate: float = 0.1,
        use_encoder_attn: bool = False,
    ):
        super().__init__()
        self.ln_self_attn = LayerNorm(hidden_dim)
        self.self_attn = MultiHeadAttention(hidden_dim, num_attn_heads, dropout_rate)

        if use_encoder_attn:
            self.ln_encoder_attn = LayerNorm(hidden_dim)
            self.encoder_attn = MultiHeadAttention(
                hidden_dim, num_attn_heads, dropout_rate
            )

        self.ln_ff = LayerNorm(hidden_dim)
        self.ff = PositionwiseFeedForward(hidden_dim, expansion_ratio, dropout_rate)

    def forward(
        self,
        x: torch.Tensor,
        y: Optional[torch.Tensor] = None,
        self_attn_mask: Optional[torch.Tensor] = None,
        enc_attn_mask: Optional[torch.Tensor] = None,
        self_attn_cache: Optional[AttentionCache] = None,
        enc_attn_cache: Optional[AttentionCache] = None,
    ) -> Tuple[torch.Tensor, Optional[AttentionCache], Optional[AttentionCache]]:
        """
        Args:
            x: The hidden representation tensor.
            y: The output representations from a transformer encoder model. Default is
                `None`.
            self_attn_mask: The attention mask for self-attention layer. Default is
                `None`.
            enc_attn_mask: The attention mask for encoder-decoder attention layer.
                Default is `None`.
            self_attn_cache: The attention cache of self-attention layer. Default is
                `None`.
            enc_attn_cache: The attention cache of encoder-decoder attention layer.
                Default is `None`.

        Returns:
            - A transformed hidden representation tensor.
            - A newly calculated attention cache from the self-attention layer. It
                consists of projected key and value vectors. It can be `None` if the
                model is training.
            - A newly calculated attention cache from the encoder-decoder attention
                layer. It consists of projected key and value vectors. It can be `None`
                if encoder-decoder attention is disabled or the model is training.
        """
        # Perform the self-attention layer.
        z = self.ln_self_attn(x)
        z, self_attn_cache = self.self_attn(z, z, z, self_attn_mask, self_attn_cache)
        x = x + z

        # Perform the encoder-decoder attention layer.
        if hasattr(self, "encoder_attn"):
            z, enc_attn_cache = self.encoder_attn(
                self.ln_encoder_attn(x), y, y, enc_attn_mask, enc_attn_cache
            )
            x = x + z
        else:
            enc_attn_cache = None

        # Perform the position-wise feed-forward layer.
        x = x + self.ff(self.ln_ff(x))

        # Do not return the attention caches to reduce the memory usage when the model
        # is training.
        if self.training:
            return x, None, None
        else:
            return x, self_attn_cache, enc_attn_cache


class Transformer(nn.Module):
    """
    Transformer is a model architecture eschewing recurrence and instead relying
    entirely on an attention mechanism to draw global dependencies between input and
    output.

    Args:
        num_layers: The number of transformer layers.
        hidden_dim: The number of hidden units.
        num_attn_heads: The number of attention heads.
        expansion_ratio: The expansion ratio of the bottleneck layer. Default is `4`.
        dropout_rate: The probability of dropping elements randomly. Default is `0.1`.
        use_encoder_attn: The boolean that determines whether to use encoder attention
            layer or not. Default is `False`.
        bidirectional: The boolean that determines whether to use bidirectional
            attention or not. If it is disabled, each position can only attend to the
            earlier positions in the sequence. Default is `True`.
    """

    def __init__(
        self,
        num_layers: int,
        hidden_dim: int,
        num_attn_heads: int,
        expansion_ratio: float = 4,
        dropout_rate: float = 0.1,
        use_encoder_attn: bool = True,
        bidirectional: bool = True,
    ):
        super().__init__()
        self.bidirectional = bidirectional

        self.transformer_layers = nn.ModuleList(
            TransformerLayer(
                hidden_dim=hidden_dim,
                num_attn_heads=num_attn_heads,
                expansion_ratio=expansion_ratio,
                dropout_rate=dropout_rate,
                use_encoder_attn=use_encoder_attn,
            )
            for _ in range(num_layers)
        )
        self.ln_head = LayerNorm(hidden_dim)

    def forward(
        self,
        x: torch.Tensor,
        y: Optional[torch.Tensor] = None,
        self_seq_mask: Optional[torch.Tensor] = None,
        enc_seq_mask: Optional[torch.Tensor] = None,
        self_attn_cache_set: Optional[AttentionCacheSet] = None,
        enc_attn_cache_set: Optional[AttentionCacheSet] = None,
    ) -> Tuple[torch.Tensor, Optional[AttentionCacheSet], Optional[AttentionCacheSet]]:
        """
        Args:
            x: The input representation tensor.
            y: The output representations from a transformer encoder model. Default is
                `None`.
            self_seq_mask: The element mask of the input sequence. Default is `None`.
            enc_seq_mask: The element mask of the encoder input sequence. Default is
                `None`.
            self_attn_cache_set: The set of attention caches for self-attention layers.
                Default is `None`.
            enc_attn_cache_set: The set of attention caches for encoder-decoder
                attention layers. Default is `None`.

        Returns:
            - An output representation tensor.
            - A set of newly calculated attention caches from self-attention layers. It
                can be `None` if the model is training.
            - A set of newly calculated attention caches from encoder-decoder attention
                layers. It can be `None` if encoder-decoder attention is disabled or the
                model is training.
        """
        new_self_attn_cache_set: AttentionCacheSet = []
        new_enc_attn_cache_set: AttentionCacheSet = []

        seq_offset = 0
        if self_attn_cache_set is not None:
            seq_offset = self_attn_cache_set[0][0].size(-2)

        # Create attention mask for self-attention layers and encoder-decoder attention
        # layers.
        self_attn_mask: Optional[torch.Tensor] = None
        if self_seq_mask is not None:
            self_attn_mask = self_seq_mask.unsqueeze(-2)
            shifted = self_seq_mask.new_zeros(self_attn_mask.shape[:-1] + (seq_offset,))

            self_attn_mask = torch.cat((shifted, self_attn_mask), dim=-1)

        enc_attn_mask: Optional[torch.Tensor] = None
        if enc_seq_mask is not None:
            enc_attn_mask = enc_seq_mask.unsqueeze(-2)

        # Create a future mask for the self-attention layers if bidirectional attention
        # is disabled.
        if not self.bidirectional:
            future_mask = torch.ones(
                (x.size(-2), x.size(-2) + seq_offset), dtype=torch.bool, device=x.device
            )
            future_mask = future_mask.triu(seq_offset + 1)
            future_mask = future_mask.view((1,) * (x.ndim - 2) + future_mask.shape)

            if self_attn_mask is None:
                self_attn_mask = future_mask
            else:
                self_attn_mask = self_attn_mask + future_mask

        # Perform the transformer layers and gather the self-attention caches.
        for i, transformer_layer in enumerate(self.transformer_layers):
            x, self_attn_cache, enc_attn_cache = transformer_layer(
                x,
                y,
                self_attn_mask,
                enc_attn_mask,
                self_attn_cache_set[i] if self_attn_cache_set is not None else None,
                enc_attn_cache_set[i] if enc_attn_cache_set is not None else None,
            )

            if self_attn_cache is not None:
                new_self_attn_cache_set.append(self_attn_cache)

            if enc_attn_cache is not None:
                new_enc_attn_cache_set.append(enc_attn_cache)

        return (
            self.ln_head(x),
            new_self_attn_cache_set if new_self_attn_cache_set else None,
            new_enc_attn_cache_set if new_enc_attn_cache_set else None,
        )
