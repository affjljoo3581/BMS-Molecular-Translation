from typing import List, Optional, Tuple

import torch
import torch.nn as nn
from tokenizers import Tokenizer

from modeling.attention import AttentionCacheSet
from modeling.embeddings import GPT2Embedding, ViTEmbedding
from modeling.heads import LMHead
from modeling.transformer import Transformer

try:
    from apex.normalization import FusedLayerNorm as LayerNorm
except ModuleNotFoundError:
    from torch.nn import LayerNorm


class MoT(nn.Module):
    """
    MoT is a transformer-based molecular translation model, which consists of
    ViT-encoder and GPT2-decoder.

    Args:
        image_size: The input image resolution.
        num_channels: The number of image channels.
        patch_size: The patch image resolution.
        vocab_size: The number of tokens in the vocabulary.
        num_encoder_layers: The number of transformer encoder layers.
        num_decoder_layers: The number of transformer decoder layers.
        hidden_dim: The number of hidden units.
        num_attn_heads: The number of attention heads in the layers.
        expansion_ratio: The expansion ratio of the bottleneck layer. Default is `4`.
        encoder_dropout_rate: The probability of dropping elements in transformer
            encoder layers randomly. Default is `0.1`.
        decoder_dropout_rate: The probability of dropping elements in transformer
            decoder layers randomly. Default is `0.1`.
        use_torchscript: The boolean determining whether to convert model to
            torchscript. Default is `False`.
    """

    def __init__(
        self,
        image_size: int,
        num_channels: int,
        patch_size: int,
        vocab_size: int,
        max_seq_len: int,
        num_encoder_layers: int,
        num_decoder_layers: int,
        hidden_dim: int,
        num_attn_heads: int,
        expansion_ratio: int,
        encoder_dropout_rate: float = 0.1,
        decoder_dropout_rate: float = 0.1,
        use_torchscript: bool = False,
    ):
        super().__init__()
        self.encoder_embedding = ViTEmbedding(
            image_size=image_size,
            num_channels=num_channels,
            patch_size=patch_size,
            hidden_dim=hidden_dim,
            dropout_rate=encoder_dropout_rate,
        )
        self.decoder_embedding = GPT2Embedding(
            vocab_size=vocab_size,
            hidden_dim=hidden_dim,
            max_seq_len=max_seq_len,
            dropout_rate=decoder_dropout_rate,
        )
        self.transformer_encoder = Transformer(
            num_layers=num_encoder_layers,
            hidden_dim=hidden_dim,
            num_attn_heads=num_attn_heads,
            expansion_ratio=expansion_ratio,
            dropout_rate=encoder_dropout_rate,
            use_encoder_attn=False,
            bidirectional=True,
        )
        self.transformer_decoder = Transformer(
            num_layers=num_decoder_layers,
            hidden_dim=hidden_dim,
            num_attn_heads=num_attn_heads,
            expansion_ratio=expansion_ratio,
            dropout_rate=decoder_dropout_rate,
            use_encoder_attn=True,
            bidirectional=False,
        )
        self.lm_head = LMHead(hidden_dim, vocab_size)

        # Initialize the parameters.
        self.reset_parameters()

        if use_torchscript:
            self.encoder_embedding = torch.jit.script(self.encoder_embedding)
            self.decoder_embedding = torch.jit.script(self.decoder_embedding)
            self.transformer_encoder = torch.jit.script(self.transformer_encoder)
            self.transformer_decoder = torch.jit.script(self.transformer_decoder)
            self.lm_head = torch.jit.script(self.lm_head)

    def reset_parameters(self):
        for name, module in self.named_modules():
            if isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, std=0.02)
            elif isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.02)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Conv2d):
                nn.init.kaiming_uniform_(module.weight, nonlinearity="conv2d")
                nn.init.zeros_(module.bias)
            elif isinstance(module, LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward_encoder(self, images: torch.Tensor) -> torch.Tensor:
        """
        Args:
            images: The input image tensor.

        Returns:
            An output representation tensor.
        """
        x, _, _ = self.transformer_encoder(self.encoder_embedding(images))
        return x

    def forward_decoder(
        self,
        input_ids: torch.Tensor,
        input_mask: Optional[torch.Tensor] = None,
        encoder_output: Optional[torch.Tensor] = None,
        self_attn_cache_set: Optional[AttentionCacheSet] = None,
        enc_attn_cache_set: Optional[AttentionCacheSet] = None,
    ) -> Tuple[torch.Tensor, AttentionCacheSet, Optional[AttentionCacheSet]]:
        """
        Args:
            input_ids: The input sequence tensor.
            input_mask: The element mask of the input sequence. Default is `None`.
            encoder_output: The output representations of a transformer encoder model.
                Default is `None`.
            self_attn_cache_set: The set of attention caches for self-attention layers.
                Default is `None`.
            enc_attn_cache_set: The set of attention caches for encoder-decoder
                attention layers. Default is `None`.

        Returns:
            - An output logits tensor.
            - A set of newly calculated attention caches from self-attention layers.
            - A set of newly calculated attention caches from encoder-decoder attention
                layers. It can be `None` if encoder-decoder attention is disabled.
        """
        seq_offset = 0
        if self_attn_cache_set is not None:
            seq_offset = self_attn_cache_set[0][0].size(-2)

        x, self_attn_cache_set, enc_attn_cache_set = self.transformer_decoder(
            self.decoder_embedding(input_ids, seq_offset),
            encoder_output,
            self_seq_mask=input_mask,
            self_attn_cache_set=self_attn_cache_set,
            enc_attn_cache_set=enc_attn_cache_set,
        )
        x = self.lm_head(x)

        return x, self_attn_cache_set, enc_attn_cache_set

    def generate(
        self,
        images: torch.Tensor,
        max_seq_len: int,
        tokenizer: Tokenizer,
    ) -> List[str]:
        self_attn_cache_set, enc_attn_cache_set = None, None
        encoder_output = self.forward_encoder(images)

        input_ids = torch.full(
            (images.size(0), 1),
            fill_value=tokenizer.token_to_id("[BOS]"),
            dtype=torch.long,
            device=images.device,
        )
        output_ids = input_ids.tolist()

        for i in range(max_seq_len):
            # Calculate the probabilites of next tokens and sample from them.
            logits, self_attn_cache_set, enc_attn_cache_set = self.forward_decoder(
                input_ids,
                input_mask=None,
                encoder_output=encoder_output if enc_attn_cache_set is None else None,
                self_attn_cache_set=self_attn_cache_set,
                enc_attn_cache_set=enc_attn_cache_set,
            )
            next_token_ids = logits[:, -1, :].argmax(-1)

            # Update the input tensor and attention cache sets.
            mask = next_token_ids != tokenizer.token_to_id("[EOS]")
            if not mask.any():
                break

            input_ids = next_token_ids[mask].unsqueeze(1)
            self_attn_cache_set = [(k[mask], v[mask]) for k, v in self_attn_cache_set]
            enc_attn_cache_set = [(k[mask], v[mask]) for k, v in enc_attn_cache_set]

            # Add the predicted tokens to the sequences.
            next_token_ids = next_token_ids.tolist()

            for target_output in output_ids:
                if len(target_output) == i + 1:
                    next_token_id = next_token_ids.pop(0)

                    if next_token_id != tokenizer.token_to_id("[EOS]"):
                        target_output.append(next_token_id)

        return [x.replace(" ", "") for x in tokenizer.decode_batch(output_ids)]
