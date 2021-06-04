import torch

from modeling.feedforward import PositionwiseFeedForward


def test_PositionwiseFeedForward_BC():
    model = PositionwiseFeedForward(
        hidden_dim=1024, expansion_ratio=4, dropout_rate=0.1
    )

    assert model(torch.rand((4, 1024))).shape == (4, 1024)
    assert model(torch.rand((16, 1024))).shape == (16, 1024)
    assert model(torch.rand((32, 1024))).shape == (32, 1024)


def test_PositionwiseFeedForward_BTC():
    model = PositionwiseFeedForward(
        hidden_dim=1024, expansion_ratio=4, dropout_rate=0.1
    )

    assert model(torch.rand((4, 128, 1024))).shape == (4, 128, 1024)
    assert model(torch.rand((16, 32, 1024))).shape == (16, 32, 1024)
    assert model(torch.rand((32, 4, 1024))).shape == (32, 4, 1024)
