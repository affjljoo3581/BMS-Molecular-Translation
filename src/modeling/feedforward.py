import math

import torch.nn as nn


class PositionwiseFeedForward(nn.Sequential):
    """
    A fully connected feed-forward layer, which is applied to each position separately
    and identically.

    Args:
        hidden_dim: The number of hidden units.
        expansion_ratio: The expansion ratio of the bottleneck layer. Default is `4`.
        dropout_rate: The probability of dropping elements randomly. Default is `0.1`.
    """

    def __init__(
        self, hidden_dim: int, expansion_ratio: float = 4, dropout_rate: float = 0.1
    ):
        super().__init__(
            nn.Linear(hidden_dim, math.floor(expansion_ratio * hidden_dim)),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(math.floor(expansion_ratio * hidden_dim), hidden_dim),
            nn.Dropout(dropout_rate),
        )
