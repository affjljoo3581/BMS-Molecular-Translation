import argparse
import math

import matplotlib.pyplot as plt
import torch
from sklearn.decomposition import PCA


def main(args: argparse.Namespace):
    state_dict = torch.load(args.input, map_location="cpu")

    positions = state_dict["encoder_embedding.position_embedding.weight"]
    projection = state_dict["encoder_embedding.projection.weight"]

    max_seq_len = positions.size(0)
    image_size = math.floor(math.sqrt(max_seq_len))
    patch_size = projection.size(-1)

    # Visualize the position embedding cosine similarities.
    positions = positions / positions.norm(dim=-1, keepdim=True)
    similarity = positions @ positions.T

    for i in range(similarity.size(0)):
        plt.subplot(image_size, image_size, i + 1)
        plt.imshow(similarity[i].view(image_size, image_size), cmap="gray")
    plt.show()

    # Show the principal filters decomposed by PCA.
    pca = PCA()
    filters = pca.fit_transform(projection.flatten(1, -1).transpose(0, 1)).T
    for i in range(45):
        plt.subplot(5, 9, i + 1)
        plt.imshow(filters[i].reshape((patch_size, patch_size)))
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input")
    args = parser.parse_args()

    main(args)
