import torch

from modeling.transformer import Transformer, TransformerLayer


def test_TransformerLayer_BTC():
    model = TransformerLayer(
        hidden_dim=768,
        num_attn_heads=12,
        expansion_ratio=4,
        dropout_rate=0.1,
        use_encoder_attn=False,
    ).eval()

    x, self_attn_cache, enc_attn_cache = model(torch.rand((4, 128, 768)))
    assert x.shape == (4, 128, 768)
    assert self_attn_cache[0].shape == (4, 12, 128, 64)
    assert self_attn_cache[1].shape == (4, 12, 128, 64)
    assert enc_attn_cache is None

    x, self_attn_cache, enc_attn_cache = model(torch.rand((8, 256, 768)))
    assert x.shape == (8, 256, 768)
    assert self_attn_cache[0].shape == (8, 12, 256, 64)
    assert self_attn_cache[1].shape == (8, 12, 256, 64)
    assert enc_attn_cache is None

    x, self_attn_cache, enc_attn_cache = model(torch.rand((32, 16, 768)))
    assert x.shape == (32, 16, 768)
    assert self_attn_cache[0].shape == (32, 12, 16, 64)
    assert self_attn_cache[1].shape == (32, 12, 16, 64)
    assert enc_attn_cache is None


def test_TransformerLayer_BTC_attn_cache():
    model = TransformerLayer(
        hidden_dim=768,
        num_attn_heads=12,
        expansion_ratio=4,
        dropout_rate=0.1,
        use_encoder_attn=False,
    ).eval()

    x, self_attn_cache, enc_attn_cache = model(torch.rand((4, 32, 768)))
    x, self_attn_cache, enc_attn_cache = model(
        torch.rand((4, 16, 768)), self_attn_cache=self_attn_cache
    )
    assert x.shape == (4, 16, 768)
    assert self_attn_cache[0].shape == (4, 12, 48, 64)
    assert self_attn_cache[1].shape == (4, 12, 48, 64)
    assert enc_attn_cache is None


def test_TransformerLayer_BTC_BQK_mask():
    model = TransformerLayer(
        hidden_dim=768,
        num_attn_heads=12,
        expansion_ratio=4,
        dropout_rate=0.1,
        use_encoder_attn=False,
    ).eval()

    x, self_attn_cache, enc_attn_cache = model(
        torch.rand((4, 32, 768)),
        self_attn_mask=torch.randint(0, 2, (4, 32, 32), dtype=torch.bool),
    )
    assert x.shape == (4, 32, 768)
    assert self_attn_cache[0].shape == (4, 12, 32, 64)
    assert self_attn_cache[1].shape == (4, 12, 32, 64)
    assert enc_attn_cache is None


def test_TransformerLayer_BTC_1QK_mask():
    model = TransformerLayer(
        hidden_dim=768,
        num_attn_heads=12,
        expansion_ratio=4,
        dropout_rate=0.1,
        use_encoder_attn=False,
    ).eval()

    x, self_attn_cache, enc_attn_cache = model(
        torch.rand((4, 32, 768)),
        self_attn_mask=torch.randint(0, 2, (1, 32, 32), dtype=torch.bool),
    )
    assert x.shape == (4, 32, 768)
    assert self_attn_cache[0].shape == (4, 12, 32, 64)
    assert self_attn_cache[1].shape == (4, 12, 32, 64)
    assert enc_attn_cache is None


def test_TransformerLayer_BTC_B1K_mask():
    model = TransformerLayer(
        hidden_dim=768,
        num_attn_heads=12,
        expansion_ratio=4,
        dropout_rate=0.1,
        use_encoder_attn=False,
    ).eval()

    x, self_attn_cache, enc_attn_cache = model(
        torch.rand((4, 32, 768)),
        self_attn_mask=torch.randint(0, 2, (4, 1, 32), dtype=torch.bool),
    )
    assert x.shape == (4, 32, 768)
    assert self_attn_cache[0].shape == (4, 12, 32, 64)
    assert self_attn_cache[1].shape == (4, 12, 32, 64)
    assert enc_attn_cache is None


def test_TransformerLayer_BTC_11K_mask():
    model = TransformerLayer(
        hidden_dim=768,
        num_attn_heads=12,
        expansion_ratio=4,
        dropout_rate=0.1,
        use_encoder_attn=False,
    ).eval()

    x, self_attn_cache, enc_attn_cache = model(
        torch.rand((4, 32, 768)),
        self_attn_mask=torch.randint(0, 2, (1, 1, 32), dtype=torch.bool),
    )
    assert x.shape == (4, 32, 768)
    assert self_attn_cache[0].shape == (4, 12, 32, 64)
    assert self_attn_cache[1].shape == (4, 12, 32, 64)
    assert enc_attn_cache is None


def test_TransformerLayer_BTC_encoder_BTC():
    model = TransformerLayer(
        hidden_dim=768,
        num_attn_heads=12,
        expansion_ratio=4,
        dropout_rate=0.1,
        use_encoder_attn=True,
    ).eval()

    x, self_attn_cache, enc_attn_cache = model(
        torch.rand((4, 32, 768)),
        torch.rand((4, 16, 768)),
    )
    assert x.shape == (4, 32, 768)
    assert self_attn_cache[0].shape == (4, 12, 32, 64)
    assert self_attn_cache[1].shape == (4, 12, 32, 64)
    assert enc_attn_cache[0].shape == (4, 12, 16, 64)
    assert enc_attn_cache[1].shape == (4, 12, 16, 64)


def test_TransformerLayer_BTC_encoder_BTC_BQK_mask():
    model = TransformerLayer(
        hidden_dim=768,
        num_attn_heads=12,
        expansion_ratio=4,
        dropout_rate=0.1,
        use_encoder_attn=True,
    ).eval()

    x, self_attn_cache, enc_attn_cache = model(
        torch.rand((4, 32, 768)),
        torch.rand((4, 16, 768)),
        enc_attn_mask=torch.randint(0, 2, (4, 32, 16)),
    )
    assert x.shape == (4, 32, 768)
    assert self_attn_cache[0].shape == (4, 12, 32, 64)
    assert self_attn_cache[1].shape == (4, 12, 32, 64)
    assert enc_attn_cache[0].shape == (4, 12, 16, 64)
    assert enc_attn_cache[1].shape == (4, 12, 16, 64)


def test_TransformerLayer_BTC_BQK_mask_encoder_BTC_BQK_mask():
    model = TransformerLayer(
        hidden_dim=768,
        num_attn_heads=12,
        expansion_ratio=4,
        dropout_rate=0.1,
        use_encoder_attn=True,
    ).eval()

    x, self_attn_cache, enc_attn_cache = model(
        torch.rand((4, 32, 768)),
        torch.rand((4, 16, 768)),
        self_attn_mask=torch.randint(0, 2, (4, 32, 32)),
        enc_attn_mask=torch.randint(0, 2, (4, 32, 16)),
    )
    assert x.shape == (4, 32, 768)
    assert self_attn_cache[0].shape == (4, 12, 32, 64)
    assert self_attn_cache[1].shape == (4, 12, 32, 64)
    assert enc_attn_cache[0].shape == (4, 12, 16, 64)
    assert enc_attn_cache[1].shape == (4, 12, 16, 64)


def test_TransformerLayer_BTC_BQK_mask_attn_cache_encoder_BTC_BQK_mask():
    model = TransformerLayer(
        hidden_dim=768,
        num_attn_heads=12,
        expansion_ratio=4,
        dropout_rate=0.1,
        use_encoder_attn=True,
    ).eval()

    x, self_attn_cache, enc_attn_cache = model(
        torch.rand((4, 32, 768)),
        torch.rand((4, 16, 768)),
        self_attn_mask=torch.randint(0, 2, (4, 32, 32)),
        enc_attn_mask=torch.randint(0, 2, (4, 32, 16)),
    )
    x, self_attn_cache, enc_attn_cache = model(
        torch.rand((4, 64, 768)),
        self_attn_mask=torch.randint(0, 2, (4, 64, 96)),
        enc_attn_mask=torch.randint(0, 2, (4, 64, 16)),
        self_attn_cache=self_attn_cache,
        enc_attn_cache=enc_attn_cache,
    )
    assert x.shape == (4, 64, 768)
    assert self_attn_cache[0].shape == (4, 12, 96, 64)
    assert self_attn_cache[1].shape == (4, 12, 96, 64)
    assert enc_attn_cache[0].shape == (4, 12, 16, 64)
    assert enc_attn_cache[1].shape == (4, 12, 16, 64)


def test_Transformer_BERT_style():
    model = Transformer(
        num_layers=2,
        hidden_dim=256,
        num_attn_heads=4,
        expansion_ratio=4,
        dropout_rate=0.1,
        use_encoder_attn=False,
        bidirectional=True,
    ).eval()

    x = torch.rand((32, 128, 256))
    mask = torch.randint(0, 2, (32, 128), dtype=torch.bool)
    x, self_attn_cache_set, enc_attn_cache_set = model(x, self_seq_mask=mask)

    assert x.shape == (32, 128, 256)
    for self_attn_cache in self_attn_cache_set:
        assert self_attn_cache[0].shape == (32, 4, 128, 64)
        assert self_attn_cache[1].shape == (32, 4, 128, 64)
    assert enc_attn_cache_set is None


def test_Transformer_GPT2_style():
    model = Transformer(
        num_layers=2,
        hidden_dim=256,
        num_attn_heads=4,
        expansion_ratio=4,
        dropout_rate=0.1,
        use_encoder_attn=False,
        bidirectional=False,
    ).eval()

    x = torch.rand((32, 128, 256))
    mask = torch.randint(0, 2, (32, 128), dtype=torch.bool)
    x, self_attn_cache_set, enc_attn_cache_set = model(x, self_seq_mask=mask)

    assert x.shape == (32, 128, 256)
    for self_attn_cache in self_attn_cache_set:
        assert self_attn_cache[0].shape == (32, 4, 128, 64)
        assert self_attn_cache[1].shape == (32, 4, 128, 64)
    assert enc_attn_cache_set is None


def test_Transformer_GPT2_generation():
    model = Transformer(
        num_layers=2,
        hidden_dim=256,
        num_attn_heads=4,
        expansion_ratio=4,
        dropout_rate=0.1,
        use_encoder_attn=False,
        bidirectional=False,
    ).eval()

    self_attn_cache_set, enc_attn_cache_set = None, None
    for i in range(16):
        x, self_attn_cache_set, enc_attn_cache_set = model(
            torch.rand((32, 1, 256)),
            self_attn_cache_set=self_attn_cache_set,
            enc_attn_cache_set=enc_attn_cache_set,
        )

        assert x.shape == (32, 1, 256)
        for self_attn_cache in self_attn_cache_set:
            assert self_attn_cache[0].shape == (32, 4, i + 1, 64)
            assert self_attn_cache[1].shape == (32, 4, i + 1, 64)
        assert enc_attn_cache_set is None


def test_Transformer_BART_style():
    encoder = Transformer(
        num_layers=2,
        hidden_dim=256,
        num_attn_heads=4,
        expansion_ratio=4,
        dropout_rate=0.1,
        use_encoder_attn=False,
        bidirectional=True,
    ).eval()
    decoder = Transformer(
        num_layers=2,
        hidden_dim=256,
        num_attn_heads=4,
        expansion_ratio=4,
        dropout_rate=0.1,
        use_encoder_attn=True,
        bidirectional=False,
    ).eval()

    x = torch.rand((32, 128, 256))
    encoder_seq_mask = torch.randint(0, 2, (32, 128), dtype=torch.bool)
    encoder_output, _, _ = encoder(x, encoder_seq_mask)

    assert encoder_output.shape == (32, 128, 256)

    x = torch.rand((32, 64, 256))
    seq_mask = torch.randint(0, 2, (32, 64), dtype=torch.bool)
    x, self_attn_cache_set, enc_attn_cache_set = decoder(
        x, encoder_output, self_seq_mask=seq_mask, enc_seq_mask=encoder_seq_mask
    )

    assert x.shape == (32, 64, 256)
    for self_attn_cache in self_attn_cache_set:
        assert self_attn_cache[0].shape == (32, 4, 64, 64)
        assert self_attn_cache[1].shape == (32, 4, 64, 64)
    for enc_attn_cache in enc_attn_cache_set:
        assert enc_attn_cache[0].shape == (32, 4, 128, 64)
        assert enc_attn_cache[1].shape == (32, 4, 128, 64)


def test_Transformer_BART_generation():
    encoder = Transformer(
        num_layers=2,
        hidden_dim=256,
        num_attn_heads=4,
        expansion_ratio=4,
        dropout_rate=0.1,
        use_encoder_attn=False,
        bidirectional=True,
    ).eval()
    decoder = Transformer(
        num_layers=2,
        hidden_dim=256,
        num_attn_heads=4,
        expansion_ratio=4,
        dropout_rate=0.1,
        use_encoder_attn=True,
        bidirectional=False,
    ).eval()

    x = torch.rand((32, 128, 256))
    seq_mask = torch.randint(0, 2, (32, 128), dtype=torch.bool)
    encoder_output, _, _ = encoder(x, self_seq_mask=seq_mask)

    self_attn_cache_set, enc_attn_cache_set = None, None
    for i in range(16):
        x = torch.rand((32, 1, 256))
        x, self_attn_cache_set, enc_attn_cache_set = decoder(
            x,
            encoder_output if enc_attn_cache_set is None else None,
            enc_seq_mask=seq_mask,
            self_attn_cache_set=self_attn_cache_set,
            enc_attn_cache_set=enc_attn_cache_set,
        )

        assert x.shape == (32, 1, 256)
        for self_attn_cache in self_attn_cache_set:
            assert self_attn_cache[0].shape == (32, 4, i + 1, 64)
            assert self_attn_cache[1].shape == (32, 4, i + 1, 64)
        for enc_attn_cache in enc_attn_cache_set:
            assert enc_attn_cache[0].shape == (32, 4, 128, 64)
            assert enc_attn_cache[1].shape == (32, 4, 128, 64)
