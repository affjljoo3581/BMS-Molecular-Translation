from __future__ import annotations

import math
import os
from typing import Callable, Dict, Tuple

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class BMSDataset(Dataset):
    """Dataset class for images and InChI strings in `BMS Molecular Translation`.

    Args:
        samples: The target samples which contains image ids, InChI strings and image
            directory.
        transform: The image transform which performs as an augmentation.
    """

    def __init__(self, samples: pd.DataFrame, transform: Callable):
        super().__init__()
        self.samples = samples
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, str]:
        image_id, inchi, image_dir = self.samples.iloc[index]

        img = cv2.imread(os.path.join(image_dir, *image_id[:3], image_id + ".png"))
        img = np.expand_dims(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), axis=-1)

        return image_id, self.transform(image=img)["image"], inchi

    @staticmethod
    def create_train_and_val_datasets(
        datasets: Dict[str, Dict[str, str]],
        val_ratio: float,
        train_transform: Callable,
        val_transform: Callable,
    ) -> Tuple[BMSDataset, BMSDataset]:
        """Create train and validation datasets by splitting the original datasets.

        Args:
            datasets: The dataset configurations. Each configuration contains the base
                image directory and the label csv file path. Note that `main` dataset
                should be given.
            val_ratio: The ratio of the validation samples to the full ones in the
                `main` dataset.
            train_transform: The image transform for train images which performs as an
                augmentation.
            val_transform The image transform for validation images which performs as an
                augmentation.

        Note:
            To compare the performance between **without-external-data** and
            **with-external-data** fairly, the validation samples will be splitted from
            the `main` dataset specified in `datasets` argument. Since the random seed
            is fixed, the validation samples of both **without-external-data** and
            **with-external-data** cases should be same.

        Returns:
            A tuple of `BMSDataset`s which handle training and validation samples
            respectively.
        """
        label_tables = {}
        for name, cfg in datasets.items():
            table = pd.read_csv(cfg["label_csv_path"]).sample(frac=1, random_state=0)
            table["image_dir"] = cfg["image_dir"]

            label_tables[name] = table

        # Note that the validation samples are splitted from the `main` dataset.
        val_size = math.floor(len(label_tables["main"]) * val_ratio)
        val_samples = label_tables["main"].iloc[:val_size]

        label_tables["main"] = label_tables["main"].iloc[val_size:]
        train_samples = pd.concat(label_tables.values(), ignore_index=True)

        return (
            BMSDataset(train_samples, train_transform),
            BMSDataset(val_samples, val_transform),
        )
