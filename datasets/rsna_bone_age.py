from pathlib import Path
import pandas as pd
from PIL import Image
from tqdm import tqdm

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pytorch_lightning import LightningDataModule


class RSNABoneAge(Dataset):
    def __init__(self,
                 images_dir_list,
                 csv_path,
                 transform=None):

        # if images_path_list is a string, convert it to a list
        if isinstance(images_dir_list, str):
            images_dir_list = [images_dir_list]

        df = pd.read_csv(csv_path)

        # Rename the columns to use same names with both train and test datasets
        if 'Bone Age (months)' in df.columns:
            df = df.rename(columns={'Image ID': 'id',
                                    'Bone Age (months)': 'boneage',
                                    })

        self.data_list = []
        images_path = []
        for images_dir in images_dir_list:
            images_path.extend(list(Path(images_dir).glob('*.png')))

        for image_path in tqdm(images_path):
                image_id = int(image_path.stem)
                self.data_list.append({
                    'path': image_path,
                    'boneage': df[df.id == image_id].boneage.values[0],
                    'male': df[df.id == image_id].male.values[0],
                })

        self.transform = transform

        # Compute mean and std of the dataset
        self.mean = np.mean([data['boneage'] for data in self.data_list])
        self.std = np.std([data['boneage'] for data in self.data_list])

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        data = self.data_list[idx]

        image = Image.open(data['path']).convert('RGB')

        if self.transform:
            image = self.transform(image=np.array(image))['image']

        return {
            'image': image,
            'target': data['boneage'],
        }


class RSNABoneAgeDataModule(LightningDataModule):
    def __init__(self,
                 data_dir,
                 train_ratio=1.0,
                 train_transform=None,
                 test_transform=None,
                 batch_size=32,
                 num_workers=8,
                 ):
        super().__init__()

        data_dir = Path(data_dir)
        self.train_images_dir_list = [data_dir / 'boneage-training-dataset']
        self.test_images_dir_list = [data_dir / 'boneage-validation-dataset-1',
                                     data_dir / 'boneage-validation-dataset-2']
        self.train_csv_path = data_dir / 'train.csv'
        self.test_csv_path = data_dir / "Validation Dataset.csv"
        self.train_ratio = train_ratio
        self.train_transform = train_transform
        self.test_transform = test_transform
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

        self.mean = None
        self.std = None
        self.target_dim = 1

    def setup(self, stage=None):
        train_val_dataset = RSNABoneAge(images_dir_list=self.train_images_dir_list,
                                        csv_path=self.train_csv_path,
                                        transform=self.train_transform)

        self.mean = train_val_dataset.mean
        self.std = train_val_dataset.std

        # Split the train_val_dataset into train and val datasets
        train_size = int(0.9 * len(train_val_dataset))
        val_size = len(train_val_dataset) - train_size
        self.train_dataset, self.val_dataset = torch.utils.data.random_split(train_val_dataset,
                                                                             [train_size, val_size],
                                                                             )

        if self.train_ratio < 1:
            # split train_dataset to keep only train_ratio of the dataset
            train_size = int(self.train_ratio * len(self.train_dataset))
            ignore_size = len(self.train_dataset) - train_size
            self.train_dataset, _ = torch.utils.data.random_split(self.train_dataset, [train_size, ignore_size])

        self.test_dataset = RSNABoneAge(images_dir_list=self.test_images_dir_list,
                                        csv_path=self.test_csv_path,
                                        transform=self.test_transform)

    def train_dataloader(self):
        return DataLoader(self.train_dataset,
                          batch_size=self.batch_size,
                          shuffle=True,
                          num_workers=self.num_workers,
                          drop_last=True,
                          pin_memory=True,
                          )

    def val_dataloader(self):
        return DataLoader(self.val_dataset,
                          batch_size=self.batch_size,
                          shuffle=False,
                          num_workers=self.num_workers,
                          drop_last=False,
                          pin_memory=True,
                          )

    def test_dataloader(self):
        return DataLoader(self.test_dataset,
                          batch_size=self.batch_size,
                          shuffle=False,
                          num_workers=self.num_workers,
                          drop_last=False,
                          pin_memory=True,
                          )
