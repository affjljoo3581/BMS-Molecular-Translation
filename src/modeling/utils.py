import torch.utils.checkpoint as checkpoint

from modeling.transformer import Transformer, TransformerLayer


def modify_transformer_layer_forward(layer: TransformerLayer):
    original_forward = layer.forward

    def modified_forward(*args, **kwargs):
        if layer.training:
            # Wrap the transformer layer to return the first tensor only.
            def wrapper(*args, **kwargs):
                return original_forward(*args, **kwargs)[0]

            return checkpoint.checkpoint(wrapper, *args, **kwargs), None, None
        else:
            return original_forward(*args, **kwargs)

    layer.forward = modified_forward


def apply_gradient_checkpointing(transformer: Transformer, ratio: float = 1.0):
    """Apply gradient checkpointing to the transformer layers.

    Args:
        transformer: The target transformer model.
        ratio: The ratio of gradient-checkpointed transformer layers. Default is `1.0`.
    """
    for i in range(int(ratio * len(transformer.transformer_layers))):
        modify_transformer_layer_forward(transformer.transformer_layers[i])
