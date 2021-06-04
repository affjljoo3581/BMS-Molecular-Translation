import argparse
import re
import string
from typing import Iterator

import torch


"""
Modify Mode
===========

repeat-first
------------
ABC => ABCABCABC, ABCABCAB, ABCABCA
ABCDEFG => ABCDEF, ABCD

A, AB, ABC, ABCA, ABCAB, ABCABC, ...

repeat-last
-----------
ABC => ABCABCABC, BCABCABC, CABCABC
ABCDEFG => BCDEFG, DEFG

C, BC, ABC, CABC, BCABC, ...

repeat-interleave-first
-----------------------
ABC => AAAABBBCCC, AAABBCC,  AABBCC, AABBC, AABC
ABCDEFG => BCDEFG, DEFG

A, AB, ABC, AABC, AABBC, AABBCC, AAABBCC, AAABBBCC, ...

repeat-interleave-last
----------------------
ABC => AAABBBCCCC, AAABBBCCC, AABBCCC, ABBCC
ABCDEFG => ABCDEF, ABC

C, BC, ABC, ABCC, ABBCC, AABBCC, AABBCCC, AABBBCCC, ...
"""


def _repeat_indexing(
    index: int, current_length: int, desired_length: int
) -> Iterator[int]:
    yield from range(index, desired_length, current_length)


def _repeat_interleave_indexing(
    index: int, current_length: int, desired_length: int
) -> Iterator[int]:
    base_repeats = desired_length // current_length
    residuals = desired_length - base_repeats * current_length

    start_index = base_repeats * index + min(residuals, index)
    end_index = start_index + base_repeats + (1 if index < residuals else 0)
    yield from range(start_index, end_index)


def repeat_layer_indexing(
    index: int, current_length: int, desired_length: int, mode: str = "repeat-first"
) -> Iterator[int]:
    if mode == "repeat-first":
        yield from _repeat_indexing(index, current_length, desired_length)
    elif mode == "repeat-last":
        yield from map(
            lambda i: desired_length - i - 1,
            _repeat_indexing(
                current_length - index - 1, current_length, desired_length
            ),
        )
    elif mode == "repeat-interleave-first":
        yield from _repeat_interleave_indexing(index, current_length, desired_length)
    elif mode == "repeat-interleave-last":
        yield from map(
            lambda i: desired_length - i - 1,
            _repeat_interleave_indexing(
                current_length - index - 1, current_length, desired_length
            ),
        )
    else:
        raise ValueError(f"`{mode}` is not supported.")


def main(args: argparse.Namespace):
    new_state_dict = {}
    old_state_dict = torch.load(args.input, map_location="cpu")

    encoder_pattern = re.compile(
        r"(transformer_encoder[.]transformer_layers[.])(\d+)([.].+)"
    )
    decoder_pattern = re.compile(
        r"(transformer_decoder[.]transformer_layers[.])(\d+)([.].+)"
    )

    # Find the maximum index of encoder and decoder layers to get the entire number of
    # encoder and decoder layers respectively.
    encoder_layers, decoder_layers = [], []
    for name in old_state_dict.keys():
        # Check if the layer is in transformer-encoder.
        match = encoder_pattern.match(name)
        if match is not None:
            encoder_layers.append(int(match.group(2)))
            continue

        # Check if the layer is in transformer-decoder.
        match = decoder_pattern.match(name)
        if match is not None:
            decoder_layers.append(int(match.group(2)))
            continue

    num_old_encoder_layers = max(encoder_layers) + 1
    num_old_decoder_layers = max(decoder_layers) + 1

    # Print an information of the layer modification.
    if args.verbose:
        old_enc_sym = string.ascii_uppercase[:num_old_encoder_layers]
        old_dec_sym = string.ascii_uppercase[:num_old_decoder_layers]

        new_enc_sym = " " * args.num_encoder_layers
        for i, symbol in enumerate(old_enc_sym):
            for j in repeat_layer_indexing(
                i,
                num_old_encoder_layers,
                args.num_encoder_layers,
                mode=args.modify_mode,
            ):
                new_enc_sym = new_enc_sym[:j] + symbol + new_enc_sym[j + 1 :]

        new_dec_sym = " " * args.num_decoder_layers
        for i, symbol in enumerate(old_dec_sym):
            for j in repeat_layer_indexing(
                i,
                num_old_decoder_layers,
                args.num_decoder_layers,
                mode=args.modify_mode,
            ):
                new_dec_sym = new_dec_sym[:j] + symbol + new_dec_sym[j + 1 :]

        print(f"[*] current `num_encoder_layers`: {num_old_encoder_layers}")
        print(f"[*] current `num_decoder_layers`: {num_old_decoder_layers}")
        print()
        print(f"[*] desired `num_encoder_layers`: {args.num_encoder_layers}")
        print(f"[*] desired `num_decoder_layers`: {args.num_decoder_layers}")
        print()
        print(f"[*] modification mode: {args.modify_mode}")
        print(f"[*] encoder: {old_enc_sym} => {new_enc_sym}")
        print(f"[*] decoder: {old_dec_sym} => {new_dec_sym}")
        print()
        print(f"[*] ignore patterns: {args.ignore_patterns.split('|')}")

    # Copy the weights of each layer with the destination layer inices.
    for name, value in old_state_dict.items():
        # Check if the layer is in transformer-encoder.
        match = encoder_pattern.match(name)
        if match is not None:
            for new_index in repeat_layer_indexing(
                int(match.group(2)),
                num_old_encoder_layers,
                args.num_encoder_layers,
                mode=args.modify_mode,
            ):
                new_state_dict[
                    f"{match.group(1)}{new_index}{match.group(3)}"
                ] = value.clone()
            continue

        # Check if the layer is in transformer-decoder.
        match = decoder_pattern.match(name)
        if match is not None:
            for new_index in repeat_layer_indexing(
                int(match.group(2)),
                num_old_decoder_layers,
                args.num_decoder_layers,
                mode=args.modify_mode,
            ):
                new_state_dict[
                    f"{match.group(1)}{new_index}{match.group(3)}"
                ] = value.clone()
            continue

        # If the layer is neither encoder or decoder layer, then simply copy to the new
        # `state_dict`.
        if args.ignore_patterns:
            if any(pattern in name for pattern in args.ignore_patterns.split("|")):
                continue
        new_state_dict[name] = value

    torch.save(new_state_dict, args.output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input")
    parser.add_argument("--output", default="stacked-output.pth")
    parser.add_argument("--num_encoder_layers", default=6, type=int)
    parser.add_argument("--num_decoder_layers", default=6, type=int)
    parser.add_argument("--modify_mode", default="repeat-first")
    parser.add_argument("--ignore_patterns")
    parser.add_argument("--verbose", default=False, action="store_true")
    args = parser.parse_args()

    main(args)
