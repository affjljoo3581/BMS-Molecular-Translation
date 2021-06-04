from typing import Iterable

import torch.nn as nn

try:
    from apex.normalization import FusedLayerNorm as LayerNorm
except ModuleNotFoundError:
    from torch.nn import LayerNorm


def get_do_decay_params(model: nn.Module) -> Iterable[nn.Parameter]:
    """Find the model parameters which should be decayed.

    Args:
        model: The target model which contains decayable layers.
    """
    for module in model.modules():
        if isinstance(module, nn.Linear):
            yield module.weight
        elif isinstance(module, nn.Conv2d):
            yield module.weight
        elif isinstance(module, nn.Embedding):
            yield module.weight


def get_no_decay_params(model: nn.Module) -> Iterable[nn.Parameter]:
    """Find the model parameters which must not be decayed.

    Args:
        model: The target model which contains non-decayable layers.
    """
    for module in model.modules():
        if isinstance(module, nn.Linear):
            yield module.bias
        elif isinstance(module, nn.Conv2d):
            yield module.bias
        elif isinstance(module, LayerNorm):
            yield module.weight
            yield module.bias
