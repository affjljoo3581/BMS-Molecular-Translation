import argparse
import os
import warnings
from typing import Any, Dict

import pandas as pd
import torch
import tqdm
import yaml
from tokenizers import Tokenizer
from torch.utils.data import DataLoader

from data import BMSDataset, TestTransform
from modeling import MoT

# Disable warnings and error messages for parallelism.
warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"


@torch.no_grad()
def main(cfg: Dict[str, Any]):
    tokenizer = Tokenizer.from_file(cfg["data"]["tokenizer_path"])
    sample_submission = pd.read_csv(cfg["data"]["label_csv_path"])
    sample_submission["image_dir"] = cfg["data"]["image_dir"]

    # Create dataset and dataloader for test images through a sample submission file.
    dataset = BMSDataset(
        sample_submission,
        transform=TestTransform(cfg["model"]["image_size"]),
    )
    dataloader = DataLoader(
        dataset,
        cfg["predict"]["batch_size"],
        num_workers=os.cpu_count(),
        pin_memory=True,
    )

    # Create a MoT model with given configurations. Note that the parameters will be
    # moved to CUDA memory and converted to half precision if `use_fp16` is specified.
    model = MoT(
        image_size=cfg["model"]["image_size"],
        num_channels=1,
        patch_size=cfg["model"]["patch_size"],
        vocab_size=tokenizer.get_vocab_size(),
        max_seq_len=cfg["model"]["max_seq_len"],
        num_encoder_layers=cfg["model"]["num_encoder_layers"],
        num_decoder_layers=cfg["model"]["num_decoder_layers"],
        hidden_dim=cfg["model"]["hidden_dim"],
        num_attn_heads=cfg["model"]["num_attn_heads"],
        expansion_ratio=cfg["model"]["expansion_ratio"],
        encoder_dropout_rate=0.0,
        decoder_dropout_rate=0.0,
        use_torchscript=True,
    )
    model.load_state_dict(torch.load(cfg["predict"]["weight_path"]))
    model.eval().cuda()

    if cfg["environ"]["precision"] == 16:
        model.half()

    # Predict InChI strings from the test images.
    with open(cfg["environ"]["name"] + ".csv", "w") as fp:
        fp.write("image_id,InChI\n")

        for image_ids, images, _ in tqdm.tqdm(dataloader):
            images = images.cuda()
            if cfg["environ"]["precision"] == 16:
                images = images.half()

            # Generate the InChI strings and update to the submission file.
            inchis = model.generate(
                images,
                max_seq_len=cfg["model"]["max_seq_len"],
                tokenizer=tokenizer,
            )
            for image_id, inchi in zip(image_ids, inchis):
                fp.write(f'{image_id},"{inchi}"\n')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    args = parser.parse_args()

    with open(args.config, "r") as fp:
        cfg = yaml.load(fp, yaml.FullLoader)
    main(cfg)
