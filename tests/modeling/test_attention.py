import torch

from modeling.attention import MultiHeadAttention


def test_MultiHeadAttention_BTC():
    model = MultiHeadAttention(hidden_dim=512, num_attn_heads=16, dropout_rate=0.1)

    q = torch.rand((4, 128, 512))
    k = torch.rand((4, 32, 512))
    v = torch.rand((4, 32, 512))

    x, attn_cache = model(q, k, v)
    assert x.shape == (4, 128, 512)
    assert attn_cache[0].shape == (4, 16, 32, 32)
    assert attn_cache[1].shape == (4, 16, 32, 32)


def test_MultiHeadAttention_BTC_attn_cache():
    model = MultiHeadAttention(hidden_dim=512, num_attn_heads=16, dropout_rate=0.1)

    q = torch.rand((4, 128, 512))
    k = torch.rand((4, 32, 512))
    v = torch.rand((4, 32, 512))

    x, attn_cache = model(q, k, v)
    x, attn_cache = model(q, k, v, attn_cache=attn_cache)
    assert x.shape == (4, 128, 512)
    assert attn_cache[0].shape == (4, 16, 64, 32)
    assert attn_cache[1].shape == (4, 16, 64, 32)


def test_MultiHeadAttention_BTC_BQK_mask():
    model = MultiHeadAttention(hidden_dim=512, num_attn_heads=16, dropout_rate=0.1)

    q = torch.rand((4, 128, 512))
    k = torch.rand((4, 32, 512))
    v = torch.rand((4, 32, 512))
    attn_mask = torch.randint(0, 2, (4, 128, 32))

    x, attn_cache = model(q, k, v, attn_mask=attn_mask)
    assert x.shape == (4, 128, 512)
    assert attn_cache[0].shape == (4, 16, 32, 32)
    assert attn_cache[1].shape == (4, 16, 32, 32)


def test_MultiHeadAttention_BTC_BQK_mask_attn_cache():
    model = MultiHeadAttention(hidden_dim=512, num_attn_heads=16, dropout_rate=0.1)

    q = torch.rand((4, 128, 512))
    k = torch.rand((4, 32, 512))
    v = torch.rand((4, 32, 512))
    attn_mask = torch.randint(0, 2, (4, 128, 64))

    x, attn_cache = model(q, k, v)
    x, attn_cache = model(q, k, v, attn_mask, attn_cache)
    assert x.shape == (4, 128, 512)
    assert attn_cache[0].shape == (4, 16, 64, 32)
    assert attn_cache[1].shape == (4, 16, 64, 32)


def test_MultiHeadAttention_BTC_1QK_mask():
    model = MultiHeadAttention(hidden_dim=512, num_attn_heads=16, dropout_rate=0.1)

    q = torch.rand((4, 128, 512))
    k = torch.rand((4, 32, 512))
    v = torch.rand((4, 32, 512))
    attn_mask = torch.randint(0, 2, (1, 128, 32))

    x, attn_cache = model(q, k, v, attn_mask=attn_mask)
    assert x.shape == (4, 128, 512)
    assert attn_cache[0].shape == (4, 16, 32, 32)
    assert attn_cache[1].shape == (4, 16, 32, 32)


def test_MultiHeadAttention_BTC_11K_mask():
    model = MultiHeadAttention(hidden_dim=512, num_attn_heads=16, dropout_rate=0.1)

    q = torch.rand((4, 128, 512))
    k = torch.rand((4, 32, 512))
    v = torch.rand((4, 32, 512))
    attn_mask = torch.randint(0, 2, (1, 1, 32))

    x, attn_cache = model(q, k, v, attn_mask=attn_mask)
    assert x.shape == (4, 128, 512)
    assert attn_cache[0].shape == (4, 16, 32, 32)
    assert attn_cache[1].shape == (4, 16, 32, 32)
