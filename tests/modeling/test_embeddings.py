import torch

from modeling.embeddings import GPT2Embedding, ViTEmbedding


def test_GPT2Embedding_BT():
    model = GPT2Embedding(vocab_size=8000, hidden_dim=128, max_seq_len=1024)

    assert model(torch.randint(0, 8000, (16, 128))).shape == (16, 128, 128)
    assert model(torch.randint(0, 8000, (4, 512))).shape == (4, 512, 128)
    assert model(torch.randint(0, 8000, (2, 1024))).shape == (2, 1024, 128)


def test_GPT2Embedding_BT_offset():
    model = GPT2Embedding(vocab_size=8000, hidden_dim=128, max_seq_len=1024)

    assert model(torch.randint(0, 8000, (16, 128)), 5).shape == (16, 128, 128)
    assert model(torch.randint(0, 8000, (4, 512)), 10).shape == (4, 512, 128)
    assert model(torch.randint(0, 8000, (2, 1000)), 24).shape == (2, 1000, 128)


def test_GPT2Embedding_offset_dependent():
    model = GPT2Embedding(vocab_size=8000, hidden_dim=128, max_seq_len=1024)
    model.eval()

    x = torch.randint(0, 8000, (16, 128))

    assert (model(x) == model(x)).all()
    assert (model(x, seq_offset=3) == model(x, seq_offset=3)).all()
    assert (model(x, seq_offset=3) != model(x, seq_offset=5)).any()


def test_ViTEmbedding_BCHW():
    model = ViTEmbedding(
        image_size=224,
        num_channels=3,
        patch_size=16,
        hidden_dim=128,
    )
    assert model(torch.rand((16, 3, 224, 224))).shape == (16, 196, 128)

    model = ViTEmbedding(
        image_size=384,
        num_channels=1,
        patch_size=32,
        hidden_dim=512,
    )
    assert model(torch.rand((8, 1, 384, 384))).shape == (8, 144, 512)
