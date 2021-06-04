import argparse
import math

import torch
import torch.nn.functional as F


def main(args: argparse.Namespace):
    state_dict = torch.load(args.input)

    patch_size = state_dict["encoder_embedding.projection.weight"].size(-1)
    position_embedding = state_dict["encoder_embedding.position_embedding.weight"]

    # Reshape the position embedding weights.
    emb_size = math.floor(math.sqrt(position_embedding.size(0)))
    position_embedding = position_embedding.transpose(0, 1)
    position_embedding = position_embedding.view(1, -1, emb_size, emb_size)

    # Resize the embedding weights and change to the original shape.
    image_size = args.image_size // patch_size
    position_embedding = F.interpolate(position_embedding, image_size, mode="bilinear")
    position_embedding = position_embedding.view(-1, image_size * image_size)
    position_embedding = position_embedding.transpose(0, 1)

    state_dict["encoder_embedding.position_embedding.weight"] = position_embedding
    torch.save(state_dict, args.output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input")
    parser.add_argument("--output", default="model.pth")
    parser.add_argument("--image_size", default=224, type=int)
    args = parser.parse_args()

    main(args)
