from typing import List

import torch

from modeling.mot import MoT


class DummyTokenizer:
    def decode_batch(self, batch_ids: List[List[int]]) -> List[str]:
        return [
            " ".join([chr(i + ord("a") - 3) for i in ids if i not in [0, 1, 2]])
            for ids in batch_ids
        ]

    def token_to_id(self, token: str) -> int:
        if token == "[BOS]":
            return 0
        elif token == "[EOS]":
            return 1
        elif token == "[PAD]":
            return 2
        return ord(token) - ord("a")


def test_MoT_encoder_BCHW():
    model = MoT(
        image_size=224,
        num_channels=3,
        patch_size=32,
        vocab_size=8000,
        max_seq_len=64,
        num_encoder_layers=2,
        num_decoder_layers=2,
        hidden_dim=128,
        num_attn_heads=2,
        expansion_ratio=4,
        encoder_dropout_rate=0.1,
        decoder_dropout_rate=0.1,
    )

    assert model.forward_encoder(torch.rand((16, 3, 224, 224))).shape == (16, 49, 128)


def test_MoT_decoder_BT():
    model = MoT(
        image_size=224,
        num_channels=3,
        patch_size=32,
        vocab_size=8000,
        max_seq_len=256,
        num_encoder_layers=2,
        num_decoder_layers=2,
        hidden_dim=128,
        num_attn_heads=2,
        expansion_ratio=4,
        encoder_dropout_rate=0.1,
        decoder_dropout_rate=0.1,
    ).eval()

    input_ids = torch.randint(0, 8000, (16, 64), dtype=torch.long)
    input_mask = torch.randint(0, 2, (16, 64), dtype=torch.bool)
    encoder_output = torch.rand((16, 49, 128))

    logits, self_attn_cache_set, enc_attn_cache_set = model.forward_decoder(
        input_ids, input_mask, encoder_output
    )
    assert logits.shape == (16, 64, 8000)
    for self_attn_cache in self_attn_cache_set:
        assert self_attn_cache[0].shape == (16, 2, 64, 64)
        assert self_attn_cache[1].shape == (16, 2, 64, 64)
    for enc_attn_cache in enc_attn_cache_set:
        assert enc_attn_cache[0].shape == (16, 2, 49, 64)
        assert enc_attn_cache[1].shape == (16, 2, 49, 64)

    logits, self_attn_cache_set, enc_attn_cache_set = model.forward_decoder(
        input_ids, input_mask, None, self_attn_cache_set, enc_attn_cache_set
    )
    assert logits.shape == (16, 64, 8000)
    for self_attn_cache in self_attn_cache_set:
        assert self_attn_cache[0].shape == (16, 2, 128, 64)
        assert self_attn_cache[1].shape == (16, 2, 128, 64)
    for enc_attn_cache in enc_attn_cache_set:
        assert enc_attn_cache[0].shape == (16, 2, 49, 64)
        assert enc_attn_cache[1].shape == (16, 2, 49, 64)


def test_MoT_generate():
    vocab_size = ord("z") - ord("a") + 3
    model = MoT(
        image_size=224,
        num_channels=3,
        patch_size=32,
        vocab_size=vocab_size,
        max_seq_len=64,
        num_encoder_layers=2,
        num_decoder_layers=2,
        hidden_dim=128,
        num_attn_heads=2,
        expansion_ratio=4,
        encoder_dropout_rate=0.1,
        decoder_dropout_rate=0.1,
    ).eval()

    inchis = model.generate(
        images=torch.rand((16, 3, 224, 224)),
        max_seq_len=64,
        tokenizer=DummyTokenizer(),
    )
    assert all(len(x) <= 64 for x in inchis)


def test_MoT_generate_with_torchscript():
    vocab_size = ord("z") - ord("a") + 3
    model = MoT(
        image_size=224,
        num_channels=3,
        patch_size=32,
        vocab_size=vocab_size,
        max_seq_len=64,
        num_encoder_layers=2,
        num_decoder_layers=2,
        hidden_dim=128,
        num_attn_heads=2,
        expansion_ratio=4,
        encoder_dropout_rate=0.1,
        decoder_dropout_rate=0.1,
        use_torchscript=True,
    ).eval()

    inchis = model.generate(
        images=torch.rand((16, 3, 224, 224)),
        max_seq_len=64,
        tokenizer=DummyTokenizer(),
    )
    assert all(len(x) <= 64 for x in inchis)
