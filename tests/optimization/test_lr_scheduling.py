import numpy as np
import torch.nn as nn
from torch.optim import SGD

from optimization.lr_scheduling import LinearDecayLR


def test_LinearDecayLR_learning_rates():
    optimizer = SGD(nn.Linear(1, 1).parameters(), lr=1)
    scheduler = LinearDecayLR(optimizer, total_steps=100)

    learning_rates = []
    for _ in range(100):
        learning_rates.append(optimizer.param_groups[0]["lr"])
        scheduler.step()

    assert np.allclose(learning_rates, np.linspace(1, 0, 100, endpoint=False))
