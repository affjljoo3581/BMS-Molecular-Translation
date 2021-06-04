import argparse
from collections import defaultdict

import torch


def main(args: argparse.Namespace):
    # Load all weights to the memory.
    state_dicts = defaultdict(list)
    for path in args.inputs:
        for k, v in torch.load(path, map_location="cpu").items():
            state_dicts[k].append(v)

    # Average the weights and save to the file.
    state_dict = {
        name: torch.mean(torch.stack(weights), dim=0).detach_()
        for name, weights in state_dicts.items()
    }
    torch.save(state_dict, args.output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="+")
    parser.add_argument("--output", default="averaged.pth")
    args = parser.parse_args()

    main(args)
