import torch
import torch.nn as nn


class GPT2Embedding(nn.Module):
    """
    Construct input representations by summing the corresponding token and position
    embeddings.

    Args:
        vocab_size: The number of tokens in the vocabulary.
        hidden_dim: The number of hidden units.
        max_seq_len: The maximum sequence length.
        dropout_rate: The probability of dropping elements randomly. Default is `0.1`.
    """

    def __init__(
        self,
        vocab_size: int,
        hidden_dim: int,
        max_seq_len: int,
        dropout_rate: float = 0.1,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim

        self.token_embedding = nn.Embedding(vocab_size, hidden_dim)
        self.position_embedding = nn.Embedding(max_seq_len, hidden_dim)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(
        self, x: torch.Tensor, seq_offset: int = 0, transposed: bool = False
    ) -> torch.Tensor:
        if transposed:
            return torch.matmul(x, self.token_embedding.weight.transpose(0, 1))
        else:
            # Get positional indices tensor for position embeddings.
            p = torch.arange(seq_offset, seq_offset + x.size(-1), device=x.device)
            p = p.view((1,) * (x.ndim - 1) + (-1,))

            return self.dropout(self.token_embedding(x) + self.position_embedding(p))


class ViTEmbedding(nn.Module):
    """
    Vision-Transformer reshapes an image into a sequence of flatten 2D patches, maps
    them with a trainable linear projection, and adds position embeddings to the patch
    embeddings.

    Args:
        image_size: The input image resolution.
        num_channels: The number of image channels.
        patch_size: The patch image resolution.
        hidden_dim: The number of hidden units.
        dropout_rate: The probability of dropping elements randomly. Default is `0.1`.
    """

    def __init__(
        self,
        image_size: int,
        num_channels: int,
        patch_size: int,
        hidden_dim: int,
        dropout_rate: float = 0.1,
    ):
        super().__init__()
        self.max_seq_len = (image_size // patch_size) ** 2

        self.projection = nn.Conv2d(num_channels, hidden_dim, patch_size, patch_size)
        self.position_embedding = nn.Embedding(self.max_seq_len, hidden_dim)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Get positional indices tensor for position embeddings.
        p = torch.arange(self.max_seq_len, device=x.device)
        p = p.view((1,) * (x.ndim - 3) + (-1,))

        # Slice the images into 2D patches and flatten them.
        x = x.type_as(self.projection.weight)
        x = self.projection(x).flatten(2).transpose(-1, -2)

        return self.dropout(x + self.position_embedding(p))
