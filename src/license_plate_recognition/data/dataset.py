import os
from pathlib import Path
from typing import Union

import cv2
import torch
from torch.utils.data import Dataset
from torchvision.transforms.transforms import Compose, ToTensor

import license_plate_recognition.data.utils as utils


class LicencePlateDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(
        self,
        img_dir: Path,
        dictionary: utils.Dictionary,
        img_size: tuple[int, int] = (128, 32),
        train: bool = True,
        transform: Union[None, Compose, ToTensor] = None,
    ):
        super().__init__()

        self.img_dir = os.path.join(img_dir, "train" if train else "test")
        self.images = [
            os.path.join(self.img_dir, img_filename)
            for img_filename in os.listdir(self.img_dir)
        ]
        self.labels = [img_filename.split("-")[-1][:-4] for img_filename in self.images]
        self.dictionary = dictionary
        self.img_size = img_size  # (width, height)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        img = cv2.imread(self.images[idx])
        img = cv2.resize(img, self.img_size)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if self.transform:
            img = self.transform(img)
        label = self.labels[idx]
        label_encoded = torch.LongTensor(
            [self.dictionary.char2idx[char] for char in label]
        )
        return img, label_encoded
