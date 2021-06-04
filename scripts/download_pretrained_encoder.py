import argparse
import re
from typing import Dict, Iterator, Tuple

import torch

model_cfgs = {
    "deit-tiny": dict(
        type="deit",
        num_layers=12,
        hidden_dim=192,
        num_attn_heads=3,
        url=(
            "https://dl.fbaipublicfiles.com/deit/"
            "deit_tiny_distilled_patch16_224-b40b3cf7.pth"
        ),
    ),
    "deit-small": dict(
        type="deit",
        num_layers=12,
        hidden_dim=384,
        num_attn_heads=6,
        url=(
            "https://dl.fbaipublicfiles.com/deit/"
            "deit_small_distilled_patch16_224-649709d9.pth"
        ),
    ),
    "deit-base": dict(
        type="deit",
        num_layers=12,
        hidden_dim=768,
        num_attn_heads=12,
        url=(
            "https://dl.fbaipublicfiles.com/deit/"
            "deit_base_distilled_patch16_384-d0272ac0.pth"
        ),
    ),
    "vit-base": dict(
        type="vit",
        num_layers=12,
        hidden_dim=768,
        num_attn_heads=12,
        url=(
            "https://github.com/rwightman/pytorch-image-models/releases/download/"
            "v0.1-vitjx/jx_vit_base_patch16_224_in21k-e5005f0a.pth"
        ),
    ),
    "vit-large": dict(
        type="vit",
        num_layers=24,
        hidden_dim=1024,
        num_attn_heads=16,
        url=(
            "https://github.com/rwightman/pytorch-image-models/releases/download/"
            "v0.1-vitjx/jx_vit_large_patch16_224_in21k-606da67d.pth"
        ),
    ),
}


def parse_state_dict_to_mot(
    state_dict: Dict[str, torch.Tensor], include_embeddings: bool = False
) -> Iterator[Tuple[str, torch.Tensor]]:
    for name, param in state_dict.items():
        # Position embedding layer.
        match = re.match(r"pos_embed", name)
        if match and include_embeddings:
            name = "encoder_embedding.position_embedding.weight"
            yield name.format(*match.groups()), param[0, 1:]

        # Patch projection layer.
        match = re.match(r"patch_embed[.]proj[.](.*)", name)
        if match and include_embeddings:
            name = "encoder_embedding.projection.{}"
            if param.ndim == 4:
                param = param.mean(1, keepdim=True)

            yield name.format(*match.groups()), param

        # First LayerNorm layer.
        match = re.match(r"blocks[.](\d+)[.]norm1[.](.*)", name)
        if match:
            name = "transformer_encoder.transformer_layers.{}.ln_self_attn.{}"
            yield name.format(*match.groups()), param

        # Second LayerNorm layer.
        match = re.match(r"blocks[.](\d+)[.]norm2[.](.*)", name)
        if match:
            name = "transformer_encoder.transformer_layers.{}.ln_ff.{}"
            yield name.format(*match.groups()), param

        # Attention QKV layer.
        match = re.match(r"blocks[.](\d+)[.]attn[.]qkv[.](.*)", name)
        if match:
            name = "transformer_encoder.transformer_layers.{0}.self_attn.linear_{2}.{1}"
            for keyword, param in zip("qkv", param.chunk(3)):
                yield name.format(*match.groups(), keyword), param

        # Attention projection layer.
        match = re.match(r"blocks[.](\d+)[.]attn[.]proj[.](.*)", name)
        if match:
            name = "transformer_encoder.transformer_layers.{}.self_attn.linear_out.{}"
            yield name.format(*match.groups()), param

        # First position-wise FFN layer.
        match = re.match(r"blocks[.](\d+)[.]mlp[.]fc1[.](.*)", name)
        if match:
            name = "transformer_encoder.transformer_layers.{}.ff.0.{}"
            yield name.format(*match.groups()), param

        # Second position-wise FFN layer.
        match = re.match(r"blocks[.](\d+)[.]mlp[.]fc2[.](.*)", name)
        if match:
            name = "transformer_encoder.transformer_layers.{}.ff.3.{}"
            yield name.format(*match.groups()), param

        # Last LayerNorm layer.
        match = re.match(r"norm[.](.*)", name)
        if match:
            name = "transformer_encoder.ln_head.{}"
            yield name.format(*match.groups()), param


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("base_model", default="deit-base")
    parser.add_argument("--output", default="model.pth")
    parser.add_argument("--include_embeddings", default=False, action="store_true")
    args = parser.parse_args()

    print(f"[*] base model: {args.base_model}")
    print(f"[*] total layers: {model_cfgs[args.base_model]['num_layers']}")
    print(f"[*] dimensionality: {model_cfgs[args.base_model]['hidden_dim']}")
    print(f"[*] number of attn heads: {model_cfgs[args.base_model]['num_attn_heads']}")

    state_dict = torch.hub.load_state_dict_from_url(model_cfgs[args.base_model]["url"])
    if model_cfgs[args.base_model]["type"] == "deit":
        state_dict = state_dict["model"]

    state_dict = {
        name: param
        for name, param in parse_state_dict_to_mot(state_dict, args.include_embeddings)
    }
    torch.save(state_dict, args.output)
    print(f"[*] Finish extracting weights from [{args.base_model}].")
