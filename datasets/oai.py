from pathlib import Path
import pandas as pd
import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset
from pytorch_lightning import LightningDataModule


class OAIAlignmentDataset(Dataset):
    """
    Dataset for the OAI alignment task
    """
    def __init__(self, csv_file, image_root_dir, transform=None):
        """
        :param csv_file: path to the csv file, the csv is constructed as follows:
            ID, RHKA, LHKA, Rfemlen, Lfemlen, Rtiblen, Ltiblen, filename
        :param image_root_dir: path to the root directory of the images (the filename column in the csv file),
        the images are extracted from the zip file, witout any subdirectories (i.e. all the images are in the same
        directory)
        :param transform: albumentations transform to apply to the images
        """
        self.data = pd.read_csv(csv_file)
        self.image_root_dir = image_root_dir
        self.transform = transform

        # Compute mean and std of the dataset (for the two HKA angles)
        self.mean = self.data[['RHKA', 'LHKA']].mean().values
        self.std = self.data[['RHKA', 'LHKA']].std().values
        #self.mean = self.data[['RHKA', 'LHKA', 'Rfemlen', 'Lfemlen', 'Rtiblen', 'Ltiblen']].mean().values
        #self.std = self.data[['RHKA', 'LHKA', 'Rfemlen', 'Lfemlen', 'Rtiblen', 'Ltiblen']].std().values

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]

        # Read the image
        image_path = Path(self.image_root_dir) / row.filename
        image = Image.open(image_path).convert('RGB')
        image = np.array(image)

        # Get the labels (HKA angles)
        #target = row[['RHKA', 'LHKA', 'Rfemlen', 'Lfemlen', 'Rtiblen', 'Ltiblen']].values.astype(np.float32)
        target = row[['RHKA', 'LHKA']].values.astype(np.float32)

        # Apply the transform
        if self.transform:
            image = self.transform(image=image)['image']

        return {
            'image': image,
            'target': target,
        }


class OAIAlignmentDataModule(LightningDataModule):
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

        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_ratio = train_ratio
        self.pin_memory = pin_memory
        self.train_transform = train_transform
        self.test_transform = test_transform
        self.train_csv = Path(data_dir) / "train.csv"
        self.val_csv = Path(data_dir) / "val.csv"
        self.test_csv = Path(data_dir) / "test.csv"
        self.image_root_dir = Path(data_dir) / "images"

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

        self.mean = None
        self.std = None
        self.target_dim = 2

    def prepare_data(self):
        pass

    def setup(self, stage=None):
        # Instantiate the datasets
        self.train_dataset = OAIAlignmentDataset(csv_file=self.train_csv,
                                                 image_root_dir=self.image_root_dir,
                                                 transform=self.train_transform)

        if self.train_ratio < 1:
            # split train_dataset to keep only train_ratio of the dataset
            train_size = int(self.train_ratio * len(self.train_dataset))
            ignore_size = len(self.train_dataset) - train_size
            self.train_dataset, _ = torch.utils.data.random_split(self.train_dataset, [train_size, ignore_size])

        self.val_dataset = OAIAlignmentDataset(csv_file=self.val_csv,
                                               image_root_dir=self.image_root_dir,
                                               transform=self.test_transform)

        self.test_dataset = OAIAlignmentDataset(csv_file=self.test_csv,
                                                image_root_dir=self.image_root_dir,
                                                transform=self.test_transform)

        # Compute mean and std
        self.mean = self.train_dataset.mean
        self.std = self.train_dataset.std

    def train_dataloader(self):
        return DataLoader(self.train_dataset,
                          batch_size=self.batch_size,
                          shuffle=True,
                          num_workers=self.num_workers,
                          drop_last=True,
                          pin_memory=self.pin_memory,
                          )

    def val_dataloader(self):
        return DataLoader(self.val_dataset,
                          batch_size=self.batch_size,
                          shuffle=False,
                          num_workers=self.num_workers,
                          drop_last=False,
                          pin_memory=self.pin_memory,
                          )

    def test_dataloader(self):
        return DataLoader(self.test_dataset,
                          batch_size=self.batch_size,
                          shuffle=False,
                          num_workers=self.num_workers,
                          drop_last=False,
                          pin_memory=self.pin_memory,
                          )

