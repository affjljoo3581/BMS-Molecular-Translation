from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class LabelSmoothedCrossEntropy(nn.Module):
    """Cross-entropy loss with label-smoothing.

    Args:
        epsilon: The label smoothing rate. Default is `0.1`.
        ignore_index: The label index to be ignored from calculating loss. This will be
            disabled if it is less than `0`. Default is `-100`.
        reduction: The reduction type. It must be one of {`None`, `mean`, `sum`}.
            Default is `mean`.
    """

    def __init__(
        self,
        epsilon: float = 0.1,
        ignore_index: int = -100,
        reduction: Optional[str] = "mean",
    ):
        super().__init__()
        self.epsilon = epsilon
        self.ignore_index = ignore_index
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        log_probs = logits.log_softmax(1)
        nll_loss = F.nll_loss(
            log_probs, labels, ignore_index=self.ignore_index, reduction=self.reduction
        )

        # Do not smooth the loss when label-smoothing is disabled.
        if self.epsilon == 0:
            return nll_loss

        smooth_loss = (-log_probs.mean(1)).masked_fill_(labels == self.ignore_index, 0)
        if self.reduction == "mean":
            smooth_loss = smooth_loss.sum() / ((smooth_loss > 0).float().sum() + 1e-10)
        elif self.reduction == "sum":
            smooth_loss = smooth_loss.sum()

        # Smooth the labels by interpolating the original NLL loss with the smoothing
        # loss.
        return (1 - self.epsilon) * nll_loss + self.epsilon * smooth_loss
