from __future__ import annotations

import os
from typing import Any, Dict, Optional

import pytorch_lightning as pl
from torch.utils.data import DataLoader

from data import BMSDataset, TrainTransform, ValidationTransform


class BMSDataModule(pl.LightningDataModule):
    def __init__(
        self,
        datasets: Dict[str, Dict[str, str]],
        val_ratio: float = 0.1,
        image_size: int = 384,
        train_batch_size: int = 32,
        val_batch_size: int = 32,
        num_gpus: int = 1,
    ):
        super().__init__()
        self.datasets = datasets
        self.val_ratio = val_ratio
        self.image_size = image_size
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.num_gpus = num_gpus

    def setup(self, stage: Optional[str] = None):
        self.train_dataset, self.val_dataset = BMSDataset.create_train_and_val_datasets(
            self.datasets,
            self.val_ratio,
            train_transform=TrainTransform(self.image_size),
            val_transform=ValidationTransform(self.image_size),
        )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.train_batch_size,
            shuffle=True,
            num_workers=max(os.cpu_count() // self.num_gpus, 4),
            pin_memory=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.val_batch_size,
            num_workers=max(os.cpu_count() // self.num_gpus, 4),
            pin_memory=True,
        )

    @staticmethod
    def from_configuration(cfg: Dict[str, Any]) -> BMSDataModule:
        return BMSDataModule(
            datasets=cfg["data"]["datasets"],
            val_ratio=cfg["data"]["val_ratio"],
            image_size=cfg["model"]["image_size"],
            train_batch_size=cfg["train"]["train_batch_size"],
            val_batch_size=cfg["train"]["val_batch_size"],
            num_gpus=cfg["environ"]["num_gpus"],
        )
