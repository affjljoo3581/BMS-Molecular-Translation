import argparse
import os
import warnings
from typing import Any, Dict, Optional

import pytorch_lightning as pl
import torch
import wandb
import yaml
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger

from lightning import BMSDataModule, BMSTrainingModule

# Disable warnings and error messages for parallelism.
warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def main(
    cfg: Dict[str, Any],
    resume: Optional[str] = None,
    checkpoint: Optional[str] = None,
    pretrained: Optional[str] = None,
):
    if cfg["environ"]["num_gpus"] > 1:
        pl.seed_everything(42, workers=True)

    # Train the model with the given training configuration.
    model = BMSTrainingModule.from_configuration(cfg, pretrained=pretrained)
    datamodule = BMSDataModule.from_configuration(cfg)

    pl.Trainer(
        # ==========================
        # Optimization Configuration
        # ==========================
        max_epochs=cfg["train"]["epochs"],
        accumulate_grad_batches=cfg["train"]["accumulate_grads"],
        gradient_clip_val=cfg["train"]["max_grad_norm"],
        # =========================
        # Environment Configuration
        # =========================
        gpus=cfg["environ"]["num_gpus"],
        precision=cfg["environ"]["precision"],
        amp_backend="apex" if args.use_apex_amp else "native",
        plugins="deepspeed_stage_2" if cfg["environ"]["num_gpus"] > 1 else None,
        # ===================
        # Other Configuration
        # ===================
        logger=WandbLogger(
            entity="bmskaggle",
            project="MoT",
            name=cfg["environ"]["name"],
            id=resume,
        ),
        callbacks=[LearningRateMonitor(logging_interval="step")],
        checkpoint_callback=False,
        resume_from_checkpoint=checkpoint,
        progress_bar_refresh_rate=1,
    ).fit(model, datamodule=datamodule)

    # Save the trained transformer weights.
    torch.save(model.model.state_dict(), cfg["environ"]["name"] + ".pth")
    wandb.save(cfg["environ"]["name"] + ".pth")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    parser.add_argument("--use_apex_amp", action="store_true", default=False)
    parser.add_argument("--resume")
    parser.add_argument("--checkpoint")
    parser.add_argument("--pretrained")
    args = parser.parse_args()

    with open(args.config, "r") as fp:
        cfg = yaml.load(fp, yaml.FullLoader)
    main(cfg, args.resume, args.checkpoint, args.pretrained)
