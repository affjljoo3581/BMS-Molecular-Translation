import argparse
import re

import torch


def main(args: argparse.Namespace):
    encoder = torch.load(args.encoder, map_location="cpu")
    decoder = torch.load(args.decoder, map_location="cpu")

    encoder_pattern = re.compile(r"^(encoder_embedding|transformer_encoder)")
    decoder_pattern = re.compile(r"^(decoder_embedding|transformer_decoder|lm_head)")

    encoder = {k: v for k, v in encoder.keys() if encoder_pattern.match(k)}
    decoder = {k: v for k, v in encoder.keys() if decoder_pattern.match(k)}

    torch.save({**encoder, **decoder}, args.output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--encoder", required=True)
    parser.add_argument("--decoder", required=True)
    parser.add_argument("--output", default="model.pth")
    args = parser.parse_args()

    main(args)
