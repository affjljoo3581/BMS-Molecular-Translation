import torch.nn as nn

from modeling.mot import MoT
from modeling.transformer import Transformer
from optimization.weight_decay import get_do_decay_params, get_no_decay_params

try:
    from apex.normalization import FusedLayerNorm as LayerNorm
except ModuleNotFoundError:
    from torch.nn import LayerNorm


def test_get_do_decay_params():
    model = nn.Sequential(
        nn.Embedding(128, 10),
        nn.Linear(10, 20),
        nn.ReLU(),
        nn.Linear(20, 20),
        LayerNorm(20),
    )
    params = set(get_do_decay_params(model))

    assert model[0].weight in params
    assert model[1].weight in params
    assert model[3].weight in params

    assert model[1].bias not in params
    assert model[3].bias not in params
    assert model[4].weight not in params
    assert model[3].bias not in params


def test_get_no_decay_params():
    model = nn.Sequential(
        nn.Embedding(128, 10),
        nn.Linear(10, 20),
        nn.ReLU(),
        nn.Linear(20, 20),
        LayerNorm(20),
    )
    params = set(get_no_decay_params(model))

    assert model[1].bias in params
    assert model[3].bias in params
    assert model[4].weight in params
    assert model[3].bias in params

    assert model[0].weight not in params
    assert model[1].weight not in params
    assert model[3].weight not in params


def test_Transformer_parameters_are_covered():
    model = Transformer(
        num_layers=2,
        hidden_dim=16,
        num_attn_heads=2,
        expansion_ratio=4,
        dropout_rate=0.1,
        use_encoder_attn=True,
        bidirectional=False,
    )

    do_decay_params = set(get_do_decay_params(model))
    no_decay_params = set(get_no_decay_params(model))

    assert do_decay_params - no_decay_params == do_decay_params
    assert no_decay_params - do_decay_params == no_decay_params
    assert do_decay_params.union(no_decay_params) == set(model.parameters())


def test_MoT_parameters_are_covered():
    model = MoT(
        image_size=224,
        num_channels=3,
        patch_size=16,
        vocab_size=1000,
        max_seq_len=128,
        num_encoder_layers=2,
        num_decoder_layers=2,
        hidden_dim=64,
        num_attn_heads=4,
        expansion_ratio=4,
        encoder_dropout_rate=0.1,
        decoder_dropout_rate=0.1,
    )

    do_decay_params = set(get_do_decay_params(model))
    no_decay_params = set(get_no_decay_params(model))

    assert do_decay_params - no_decay_params == do_decay_params
    assert no_decay_params - do_decay_params == no_decay_params
    assert do_decay_params.union(no_decay_params) == set(model.parameters())
