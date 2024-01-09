import yaml
from tqdm import tqdm
from pathlib import Path
import cv2
import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset

from pytorch_lightning import LightningDataModule


class GRAZPEDWRICollate:
    def __init__(self,
                 classname_list,
                 tokenizer=None,
                 ):
        self.classname_list = classname_list
        # Append the no-class token
        # self.classname_list.append("")
        self.tokenizer = tokenizer

    def __call__(self,
                 batch,
                 ):
        batch_size = len(batch)
        images = torch.stack([batch_item["image"] for batch_item in batch])
        class_text = list(self.classname_list) * batch_size
        if self.tokenizer is not None:
            class_tokenized = self.tokenizer(class_text, padding=True, return_tensors='pt')

        labels = []
        for batch_item in batch:
            labels.append({"class_labels": torch.tensor(batch_item["class_labels"], dtype=torch.long),
                           "boxes": torch.tensor(batch_item["boxes"], dtype=torch.float),
                           })

        return {"pixel_values": images,
                "input_ids": class_tokenized["input_ids"] if self.tokenizer is not None else None,
                "attention_mask": class_tokenized["attention_mask"] if self.tokenizer is not None else None,
                "labels": labels,
                }


class GRAZPEDWRIDataset(Dataset):
    def __init__(self,
                 data_dir,
                 csv_name="dataset.csv",
                 transform=None,
                 classname_list=None,
                 ):
        super().__init__()

        data_dir = Path(data_dir)
        self.data_dir = data_dir

        # Get the class names from the yolo folder if not specified
        if classname_list is None:
            with open(data_dir / "yolov5" / "meta.yaml", 'r') as file:
                meta = yaml.safe_load(file)

            self.classname_list = meta["names"]
        else:
            self.classname_list = classname_list

        # Load the labels and boxes from the individual files
        labels = list((data_dir / "yolov5" / "labels").glob("*.txt"))
        self.labels_dict = {}
        for label in tqdm(labels, desc="Loading labels and boxes"):
            key = label.name[:-4]
            boxes = []
            class_labels = []
            with open(label) as label_file:
                lines = label_file.read()
                for line in lines.split("\n"):
                    split_line = line.split(" ")
                    if len(split_line) == 5:
                        class_labels.append(int(split_line[0]))
                        box = [float(item) for item in split_line[1:]]
                        # Fix rounding errors during transformations
                        box[2] = box[2] - 1e-6
                        box[3] = box[3] - 1e-6
                        boxes.append(box)
            self.labels_dict[key] = (class_labels, boxes)

        # Get a list of the images files
        self.files = [filestem + ".png" for filestem in list(pd.read_csv(data_dir / csv_name).filestem)]

        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        image_path = self.data_dir / "images" / self.files[idx]

        key = image_path.name[:-4]

        class_labels, boxes = self.labels_dict[key]

        image = cv2.imread(str(image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image.astype(np.float32)

        image = image / 255

        if self.transform:
            transformed = self.transform(image=image, bboxes=boxes, class_labels=class_labels)
            image = transformed['image']
            boxes = transformed['bboxes']
            class_labels = transformed['class_labels']

        # image = Image.open(image_path)

        return {"image": image,
                "class_labels": class_labels,
                "boxes": boxes,
                }


class GRAZPEDWRIDataModule(LightningDataModule):
    def __init__(self,
                 data_dir,
                 tokenizer=None,
                 batch_size=32,
                 num_workers=0,
                 train_transform=None,
                 val_transform=None,
                 test_transform=None,
                 classname_list=None,
                 ):
        super().__init__()

        self.data_dir = data_dir
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_transform = train_transform

        # If one of val or test transform is not specified,
        # but the other is, use the same transform for both
        # If both are not specified, use the train transform
        if val_transform is None and test_transform is not None:
            self.val_transform = test_transform
            self.test_transform = test_transform
        elif val_transform is not None and test_transform is None:
            self.val_transform = val_transform
            self.test_transform = val_transform
        elif val_transform is None and test_transform is None:
            self.val_transform = train_transform
            self.test_transform = train_transform
        else:
            self.val_transform = val_transform
            self.test_transform = test_transform

        self.classname_list = classname_list
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def setup(self, stage=None):
        self.train_dataset = GRAZPEDWRIDataset(self.data_dir,
                                               csv_name="train_dataset.csv",
                                               transform=self.train_transform,
                                               classname_list=self.classname_list,
                                               )
        self.val_dataset = GRAZPEDWRIDataset(self.data_dir,
                                             csv_name="val_dataset.csv",
                                             transform=self.val_transform,
                                             classname_list=self.classname_list,
                                             )
        self.test_dataset = GRAZPEDWRIDataset(self.data_dir,
                                              csv_name="test_dataset.csv",
                                              transform=self.test_transform,
                                              classname_list=self.classname_list,
                                              )

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_dataset,
                                           batch_size=self.batch_size,
                                           num_workers=self.num_workers,
                                           collate_fn=GRAZPEDWRICollate(tokenizer=self.tokenizer,
                                                                        classname_list=self.classname_list,
                                                                        ),
                                           )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.val_dataset,
                                           batch_size=self.batch_size,
                                           num_workers=self.num_workers,
                                           collate_fn=GRAZPEDWRICollate(tokenizer=self.tokenizer,
                                                                        classname_list=self.classname_list,
                                                                        ),
                                           )

    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.test_dataset,
                                           batch_size=self.batch_size,
                                           num_workers=self.num_workers,
                                           collate_fn=GRAZPEDWRICollate(tokenizer=self.tokenizer,
                                                                        classname_list=self.classname_list,
                                                                        ),
                                           )
