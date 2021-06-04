import torch
import torch.nn as nn

from optimization.cross_entropy import LabelSmoothedCrossEntropy


def test_label_smoothed_cross_entropy_without_smoothing():
    criterion1 = LabelSmoothedCrossEntropy(epsilon=0.0, ignore_index=0)
    criterion2 = nn.CrossEntropyLoss(ignore_index=0)

    logits = torch.rand((16, 256, 1000)).transpose(1, 2)
    labels = torch.randint(0, 1000, (16, 256))

    assert criterion1(logits, labels) == criterion2(logits, labels)


def test_label_smoothed_cross_entropy_loss():
    criterion = LabelSmoothedCrossEntropy(epsilon=0.1, ignore_index=0)

    logits = torch.rand((16, 256, 1000)).transpose(1, 2)
    labels = torch.randint(0, 1000, (16, 256))

    criterion(logits, labels)
