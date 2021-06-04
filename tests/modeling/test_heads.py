import torch

from modeling.heads import ClassificationHead, LMHead


def test_ClassificationHead_BTC():
    model = ClassificationHead(hidden_dim=512, num_classes=1000, cls_token_pos=0)

    assert model(torch.rand((1, 2, 512))).shape == (1, 1000)
    assert model(torch.rand((4, 512, 512))).shape == (4, 1000)
    assert model(torch.rand((16, 128, 512))).shape == (16, 1000)
    assert model(torch.rand((32, 64, 512))).shape == (32, 1000)


def test_LMHead_BTC():
    model = LMHead(hidden_dim=512, vocab_size=8000)

    assert model(torch.rand((4, 64, 512))).shape == (4, 64, 8000)
    assert model(torch.rand((4, 128, 512))).shape == (4, 128, 8000)
    assert model(torch.rand((2, 512, 512))).shape == (2, 512, 8000)
    assert model(torch.rand((1, 32, 512))).shape == (1, 32, 8000)
