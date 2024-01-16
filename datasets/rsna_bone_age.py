from torch.utils.data import Dataset
from pathlib import Path
import pandas as pd
from PIL import Image
from tqdm import tqdm


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

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        data = self.data_list[idx]

        image = Image.open(data['path']).convert('RGB')

        if self.transform:
            image = self.transform(image=image)['image']

        return {
            'image': image,
            'boneage': data['boneage'],
            'male': data['male'],
        }
