from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import Levenshtein
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
from tokenizers import Tokenizer
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler as LRScheduler

from modeling import MoT, apply_gradient_checkpointing
from optimization import (
    CosineDecayLR,
    LinearDecayLR,
    get_do_decay_params,
    get_no_decay_params,
)

try:
    from apex.optimizers import FusedAdam as Adam
except ModuleNotFoundError:
    from torch.optim import AdamW as Adam


class BMSTrainingModule(pl.LightningModule):
    def __init__(
        self,
        # =======================
        # Embedding Configuration
        # =======================
        tokenizer_path: str,
        image_size: int = 384,
        patch_size: int = 32,
        max_seq_len: int = 256,
        # ===================
        # Model Configuration
        # ===================
        num_encoder_layers: int = 12,
        num_decoder_layers: int = 12,
        hidden_dim: int = 768,
        num_attn_heads: int = 12,
        expansion_ratio: int = 4,
        encoder_dropout_rate: float = 0.1,
        decoder_dropout_rate: float = 0.1,
        # ==========================
        # Optimization Configuration
        # ==========================
        warmup_steps: int = 10000,
        learning_rate: float = 1e-4,
        learning_rate_decay: str = "linear",
        weight_decay: float = 1e-2,
        grad_ckpt_ratio: float = 0.0,
        # =========================
        # Environment Configuration
        # =========================
        checkpoint_path: str = "checkpoint.ckpt",
        num_gpus: int = 1,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.tokenizer = Tokenizer.from_file(tokenizer_path)

        self.model = MoT(
            image_size=image_size,
            num_channels=1,
            patch_size=patch_size,
            vocab_size=self.tokenizer.get_vocab_size(),
            max_seq_len=max_seq_len,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            hidden_dim=hidden_dim,
            num_attn_heads=num_attn_heads,
            expansion_ratio=expansion_ratio,
            encoder_dropout_rate=encoder_dropout_rate,
            decoder_dropout_rate=decoder_dropout_rate,
            use_torchscript=False,
        )
        self.criterion = nn.CrossEntropyLoss(
            ignore_index=self.tokenizer.token_to_id("[PAD]")
        )

        # Apply gradient checkpointing to the encoder and decoder transformers.
        apply_gradient_checkpointing(self.model.transformer_encoder, grad_ckpt_ratio)
        apply_gradient_checkpointing(self.model.transformer_decoder, grad_ckpt_ratio)

    def forward(
        self, images: torch.Tensor, inchis: List[str]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Tokenize the InChI strings and create sequence masks which indicating the
        # padding tokens.
        input_ids = torch.tensor(
            [x.ids for x in self.tokenizer.encode_batch(inchis)],
            dtype=torch.long,
            device=self.device,
        )
        input_mask = input_ids == self.tokenizer.token_to_id("[PAD]")

        # Predict the next-tokens with hidden representations from the encoder model.
        logits, _, _ = self.model.forward_decoder(
            input_ids=input_ids,
            input_mask=input_mask,
            encoder_output=self.model.forward_encoder(images),
        )
        return logits[:, :-1], input_ids[:, 1:], input_mask[:, 1:]

    def training_step(
        self, batch: Tuple[torch.Tensor, str], batch_idx: int
    ) -> torch.Tensor:
        _, images, inchis = batch
        logits, labels, mask = self(images, inchis)
        loss = self.criterion(logits.transpose(-2, -1), labels)

        self.log("train/loss", loss, sync_dist=self.hparams.num_gpus > 1)
        return loss

    def validation_step(self, batch: Tuple[torch.Tensor, str], batch_idx: int):
        _, images, inchis = batch
        preds = self.model.generate(images, self.hparams.max_seq_len, self.tokenizer)
        distance = np.mean([Levenshtein.distance(x, y) for x, y in zip(inchis, preds)])

        self.log("val/distance", distance, sync_dist=self.hparams.num_gpus > 1)

    def validation_epoch_end(self, outputs: Any):
        if not self.trainer.running_sanity_check:
            self.trainer.save_checkpoint(self.hparams.checkpoint_path)
            # self.logger.experiment.save(self.hparams.checkpoint_path)

    def configure_optimizers(self) -> Tuple[List[Optimizer], List[LRScheduler]]:
        param_groups = [
            {
                "params": get_no_decay_params(self),
                "weight_decay": 0.0,
            },
            {
                "params": get_do_decay_params(self),
                "weight_decay": self.hparams.weight_decay,
            },
        ]
        optimizer = Adam(
            param_groups, lr=self.hparams.learning_rate, betas=(0.9, 0.98), eps=1e-5
        )

        if self.hparams.learning_rate_decay == "linear":
            scheduler = LinearDecayLR(
                optimizer,
                total_steps=self.num_training_steps,
                warmup_steps=self.hparams.warmup_steps,
            )
        elif self.hparams.learning_rate_decay == "cosine":
            scheduler = CosineDecayLR(
                optimizer,
                total_steps=self.num_training_steps,
                warmup_steps=self.hparams.warmup_steps,
            )

        return [optimizer], [{"scheduler": scheduler, "interval": "step", "name": "lr"}]

    @property
    def num_training_steps(self) -> int:
        """Total training steps inferred from datamodule and devices."""
        if self.trainer.max_steps:
            return self.trainer.max_steps

        limit_batches = self.trainer.limit_train_batches
        batches = len(self.train_dataloader())
        batches = (
            min(batches, limit_batches)
            if isinstance(limit_batches, int)
            else int(limit_batches * batches)
        )

        num_devices = max(1, self.trainer.num_gpus, self.trainer.num_processes)
        if self.trainer.tpu_cores:
            num_devices = max(num_devices, self.trainer.tpu_cores)

        effective_accum = self.trainer.accumulate_grad_batches * num_devices
        return (batches // effective_accum) * self.trainer.max_epochs

    @staticmethod
    def from_configuration(
        cfg: Dict[str, Any], pretrained: Optional[str] = None
    ) -> BMSTrainingModule:
        module = BMSTrainingModule(
            tokenizer_path=cfg["data"]["tokenizer_path"],
            image_size=cfg["model"]["image_size"],
            patch_size=cfg["model"]["patch_size"],
            max_seq_len=cfg["model"]["max_seq_len"],
            num_encoder_layers=cfg["model"]["num_encoder_layers"],
            num_decoder_layers=cfg["model"]["num_decoder_layers"],
            hidden_dim=cfg["model"]["hidden_dim"],
            num_attn_heads=cfg["model"]["num_attn_heads"],
            expansion_ratio=cfg["model"]["expansion_ratio"],
            encoder_dropout_rate=cfg["model"]["encoder_dropout_rate"],
            decoder_dropout_rate=cfg["model"]["decoder_dropout_rate"],
            warmup_steps=cfg["train"]["warmup_steps"],
            learning_rate=cfg["train"]["learning_rate"],
            learning_rate_decay=cfg["train"]["learning_rate_decay"],
            weight_decay=cfg["train"]["weight_decay"],
            grad_ckpt_ratio=cfg["train"]["grad_ckpt_ratio"],
            checkpoint_path=f"{cfg['environ']['name']}.ckpt",
            num_gpus=cfg["environ"]["num_gpus"],
        )

        if pretrained is not None:
            module.model.load_state_dict(torch.load(pretrained), strict=False)
            torch.cuda.empty_cache()
        return module
