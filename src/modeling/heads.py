import torch
import torch.nn as nn


class ClassificationHead(nn.Module):
    """
    Predict a category by using linear projection to the representation vector
    calculated from the classification token.

    Args:
        hidden_dim: The number of hidden units.
        num_classes: The number of categories to classify.
        cls_token_pos: The position of a classification token in each sequence. Default
            is `0`.
    """

    def __init__(self, hidden_dim: int, num_classes: int, cls_token_pos: int = 0):
        super().__init__()
        self.cls_token_pos = cls_token_pos
        self.linear_cls = nn.Linear(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear_cls(x[..., self.cls_token_pos, :])


class LMHead(nn.Module):
    """
    Project representations to the vocabulary dimensions. It can be used both
    language-modeling and masked language-modeling tasks.

    Args:
        hidden_dim: The number of hidden units.
        vocab_size: The number of tokens in the vocabulary.
    """

    def __init__(self, hidden_dim: int, vocab_size: int):
        super().__init__()
        self.linear_lm = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear_lm(x)
