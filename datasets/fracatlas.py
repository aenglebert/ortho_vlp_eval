from pathlib import Path

from PIL import ImageFile, Image
import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset, DataLoader
from pytorch_lightning import LightningDataModule

ImageFile.LOAD_TRUNCATED_IMAGES = True


class FracAtlasDataset(Dataset):
    def __init__(self,
                 csv_file,
                 image_root_dir,
                 transform=None):

        self.data = pd.read_csv(csv_file)

        image_root_dir = Path(image_root_dir)
        self.images_path_dict = {path.name: path for path in image_root_dir.glob("*/*.jpg")}

        self.pos_weight = torch.tensor((self.data.fractured == 0).sum() / (self.data.fractured == 1).sum())

        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image_name = self.data.iloc[idx].image_id
        image_path = self.images_path_dict[image_name]
        image = Image.open(image_path)
        image = image.convert('RGB')
        image = np.array(image)

        label = self.data.iloc[idx].fractured

        if self.transform:
            image = self.transform(image=image)['image']

        return {
            'images': image,
            'labels': label,
        }


class FracAtlasDataModule(LightningDataModule):
    def __init__(self,
                 data_dir,
                 batch_size=32,
                 num_workers=4,
                 train_ratio=1.0,
                 pin_memory=True,
                 train_transform=None,
                 test_transform=None,
                 ):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_ratio = train_ratio
        self.pin_memory = pin_memory
        self.train_transform = train_transform
        self.test_transform = test_transform

        self.n_classes = 1

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.pos_weight = None

    def setup(self, stage=None):
        self.train_dataset = FracAtlasDataset(
            csv_file=self.data_dir / 'train.csv',
            image_root_dir=self.data_dir / 'images',
            transform=self.train_transform,
        )
        self.pos_weight = self.train_dataset.pos_weight

        if self.train_ratio < 1.0:
            train_size = int(self.train_ratio * len(self.train_dataset))
            ignore_size = len(self.train_dataset) - train_size
            self.train_dataset, _ = torch.utils.data.random_split(
                self.train_dataset,
                [train_size, ignore_size],
                generator=torch.Generator().manual_seed(12345),
            )

        self.val_dataset = FracAtlasDataset(
            csv_file=self.data_dir / 'val.csv',
            image_root_dir=self.data_dir / 'images',
            transform=self.test_transform,
        )
        self.test_dataset = FracAtlasDataset(
            csv_file=self.data_dir / 'test.csv',
            image_root_dir=self.data_dir / 'images',
            transform=self.test_transform,
        )

    def get_pos_weight(self):
        return self.pos_weight

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )
