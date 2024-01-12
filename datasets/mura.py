import numpy as np

import albumentations as A
from torch.utils.data import Dataset, random_split
from pytorch_lightning import LightningDataModule
import pandas as pd
from tqdm import tqdm
import os
import torch
from PIL import Image


class MURAStudyCollator:
    def __init__(self, max_images):
        self.max_images = max_images

    def __call__(self, batch):

        seq_sizes = []
        images_list_of_list = []
        labels = []

        for image_list, label in batch:

            images_list_of_list.append(image_list[:self.max_images])
            seq_sizes.append(len(image_list))
            labels.append(label)

        # stack images
        images = torch.stack([image for image_list in images_list_of_list for image in image_list], 0)

        # Create a placeholder for the pooling mapping tensor
        pooling_matrix = torch.zeros((len(seq_sizes), images.shape[0]))

        # Fill the placeholder tensor with 1 where an image is present
        idx_x = 0
        for idx_y, size in enumerate(seq_sizes):
            pooling_matrix[idx_y, idx_x:idx_x + size] = 1
            idx_x += size

        # convert labels to tensor
        labels = torch.tensor(labels, dtype=torch.float)

        return {
            "images": images,
            "pooling_matrix": pooling_matrix,
            "labels": labels
        }


class MURAImageCollator:
    def __init__(self):
        pass

    def __call__(self, batch):
        images = torch.stack([image for image, _ in batch], 0)
        labels = torch.tensor([label for _, label in batch], dtype=torch.float)

        return {
            "images": images,
            "labels": labels
        }


class MURADataset(Dataset):
    def __init__(self, data_dir, csv_file, transform=None, study_level=False):
        """
        Args:
            data_dir: directory containing the dataset
            csv_file: csv file containing the dataset
            transform: transformation to apply
            study_level: if True, return a study level dataset, else return an image level dataset
            """
        if isinstance(csv_file, str):
            csv_path = os.path.join(data_dir, "MURA-v1.1/", csv_file)
            self.df = pd.read_csv(csv_path, header=None)

        elif isinstance(csv_file, pd.DataFrame):
            self.df = csv_file

        self.data_dir = data_dir

        # keep a list of list of images path, grouped by study with a label and location per study
        self.studies_path = []
        self.studies_label = []

        # Also keep a list of individuals images path with label and location of each image
        self.images_path = []
        self.images_label = []

        # loop over each study in df
        for index, row in tqdm(self.df.iterrows(), total=self.df.shape[0]):
            cur_study_images_path = []

            # study folder path
            cur_study_path = row.iloc[0]

            # study label to global list
            self.studies_label.append(row.iloc[1])

            # list images in the study folder
            cur_study_images_name = [file for file in os.listdir(os.path.join(data_dir, cur_study_path)) if
                                     not file.startswith('.')]

            # get full path for each image in the study
            for images_name in cur_study_images_name:
                images_path = os.path.join(data_dir, cur_study_path, images_name)
                # add image path to list of current study images path
                cur_study_images_path.append(images_path)
                # also add to global list of images paths
                self.images_path.append(images_path)
                # add image label to global list
                self.images_label.append(row.iloc[1])

            # add the list of current study images paths to global list of list
            self.studies_path.append(cur_study_images_path)

        # Compute pos_weight for BCEWithLogitsLoss
        if study_level:
            self.pos_weight = torch.tensor((len(self.studies_label) - sum(self.studies_label)) / sum(self.studies_label))
        else:
            self.pos_weight = torch.tensor((len(self.images_label) - sum(self.images_label)) / sum(self.images_label))

        self.studies_label = torch.tensor(self.studies_label)
        self.n_pos_studies = torch.sum(self.studies_label).item()
        self.images_label = torch.tensor(self.images_label)
        self.n_pos_images = torch.sum(self.images_label).item()

        # store parameters
        self.study_level = study_level
        self.transform = transform

    def __len__(self):
        # return number of studies or images depending of study_level
        if self.study_level:
            return len(self.studies_path)
        else:
            return len(self.images_path)

    def __pos__(self):
        # return number of positives studies or images depending of study_level
        if self.study_level:
            return self.n_pos_studies
        else:
            return self.n_pos_images

    def __getitem__(self, index):
        """
        Args:
            index: index of image or study
        Returns:
            tuple of ((list of images), label) if study_level
            tuple of (image, labels) if not study_level
        """
        if self.study_level:
            study_images = []
            # get every image in the study
            for image_path in self.studies_path[index]:
                image = Image.open(image_path).convert('RGB')

                # transform is needed
                if self.transform is not None:
                    # if Albumenations, convert to numpy array
                    if isinstance(self.transform, A.Compose):
                        image = np.array(image)
                        image = self.transform(image=image)['image']
                    else:
                        image = self.transform(image)

                # append to list of images
                study_images.append(image)

            # get label and return with list of images
            label = self.studies_label[index]
            return study_images, label

        else:
            # get a single image
            image_path = self.images_path[index]
            image = Image.open(image_path).convert('RGB')

            # transform if needed
            if self.transform is not None:
                # if Albumenations, convert to numpy array
                if isinstance(self.transform, A.Compose):
                    image = np.array(image)
                    image = self.transform(image=image)['image']
                else:
                    image = self.transform(image)

            # get image label and return with image
            label = self.images_label[index]
            return image, label


class MURADataModule(LightningDataModule):
    def __init__(self,
                 data_dir,
                 batch_size,
                 train_ratio=1.0,
                 num_workers=4,
                 study_level=False,
                 train_transform=None,
                 val_transform=None,
                 test_transform=None,
                 ):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.train_ratio = train_ratio
        self.study_level = study_level
        self.train_transform = train_transform
        self.test_transform = test_transform
        self.num_workers = num_workers
        self.n_classes = 1

        if study_level:
            #self.collate_fn = StudyCollator(8)
            self.collate_fn = MURAStudyCollator(8)
        else:
            self.collate_fn = MURAImageCollator()

    def prepare_data(self):
        pass

    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            train_val_dataset = MURADataset(self.data_dir,
                                            'train_labeled_studies.csv',
                                            study_level=self.study_level,
                                            transform=self.train_transform,
                                            )

            train_len = int(len(train_val_dataset)*0.9)
            self.train_dataset, self.val_dataset = random_split(train_val_dataset, [train_len, len(train_val_dataset)-train_len])

            if self.train_ratio < 1:
                # split train_dataset to keep only train_ratio of the dataset
                train_size = int(self.train_ratio * len(self.train_dataset))
                ignore_size = len(self.train_dataset) - train_size
                self.train_dataset, _ = random_split(self.train_dataset, [train_size, ignore_size])

        if stage == 'test' or stage is None:
            self.test_dataset = MURADataset(self.data_dir,
                                            'valid_labeled_studies.csv',
                                            study_level=True,
                                            transform=self.test_transform,
                                            )

    def get_pos_weight(self):
        return self.train_dataset.dataset.pos_weight

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_dataset,
                                           batch_size=self.batch_size,
                                           shuffle=True,
                                           num_workers=self.num_workers,
                                           pin_memory=True,
                                           collate_fn=self.collate_fn,
                                           )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.val_dataset,
                                           batch_size=self.batch_size,
                                           shuffle=False,
                                           num_workers=self.num_workers,
                                           pin_memory=True,
                                           collate_fn=self.collate_fn,
                                           )

    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.test_dataset,
                                           batch_size=self.batch_size,
                                           shuffle=False,
                                           num_workers=self.num_workers,
                                           pin_memory=True,
                                           collate_fn=MURAStudyCollator(12),
                                           )