from typing import Tuple
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import torchvision.transforms.functional as  TF
import medmnist
from medmnist import INFO
import pytorch_lightning as pl
import PIL
import torch
import random
import torch.utils.data as data

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils3D.utils import Transform3D


class PairedD4MedMNIST2D(Dataset):
    def __init__(self, data, split, test_all_rotations=False):
        """
        Args:
        data: MedMNIST dataset object. In particular, this class assumes that the dataset object has the following attributes:
            - imgs: List of PIL images
            - labels: List of labels
        transform: Transform to apply to the images
        Split: 'train', 'val', or 'test'
        x2_angle: Angle to rotate the second image by
        """
        self.data = data
        self.split = split
        self.test_all_rotations = test_all_rotations


    def __len__(self):
        """
        If self.test_all_rotations is True, for val and test we return the number of images in the dataset times the number of rotations
        Otherwise, we return the number of images
        """
        if self.test_all_rotations:
            if self.split == 'train':
                return len(self.data.imgs)
            else:
                return len(self.data.imgs) * len(self.rotation_indices)
        else:
            return len(self.data.imgs)
    

    def augment_image(self, img: torch.Tensor) -> Tuple[torch.Tensor, int]:
        choice = random.randint(0, 7)

        if choice == 0:  # Identity
            res = img
        elif choice == 1:  # r
            res = TF.rotate(img, 90)
        elif choice == 2:  # rr
            res = TF.rotate(img, 180)
        elif choice == 3:  # rrr
            res = TF.rotate(img, 270)
        elif choice == 4:  # srr
            res = TF.hflip(img)
        elif choice == 5:  # s
            res = TF.vflip(img)
        elif choice == 6:  # srrr
            res = TF.rotate(TF.hflip(img), 90)
        elif choice == 7:  #sr
            res = TF.rotate(TF.vflip(img), 90)

        return res, choice
    

    def __getitem__(self, idx):
        transformation_type = 0

        if self.split == 'train':
            x1, y1 = self.data.__getitem__(idx)
            augmented_X1, _ = self.augment_image(x1)
            transformed_X2, covariate = self.augment_image(augmented_X1)
        
        else:
            if self.test_all_rotations:
                img_idx = idx // 2
                x1, y1 = self.data.__getitem__(img_idx)
                flipping_idx = idx % 2

                flip_x1 = [True,False][flipping_idx]
                augmented_X1 = TF.hflip(x1) if flip_x1 else x1
                
                covariate = 1
                transformed_X2 = TF.hflip(augmented_X1)

            else:
                x1, y1 = self.data.__getitem__(idx)
                augmented_X1 = x1
                transformed_X2, covariate = self.augment_image(augmented_X1)

        return (augmented_X1, y1), (transformed_X2, y1), transformation_type, covariate



class D4MedMNISTDataModule(pl.LightningDataModule):
    def __init__(self, data_flag, batch_size, resize, as_rgb, size, download):
        super().__init__()
        self.data_flag = data_flag
        self.batch_size = batch_size
        self.resize = resize
        self.as_rgb = as_rgb
        self.size = size
        self.download = download
        self.info = INFO[data_flag]
        self.DataClass = getattr(medmnist, self.info['python_class'])

    def setup(self, stage=None):
        shape_transform = True if self.data_flag in ['adrenalmnist3d', 'vesselmnist3d'] else False
        train_transform = Transform3D(mul='random') if shape_transform else Transform3D()
        eval_transform = Transform3D(mul='0.5') if shape_transform else Transform3D()
        
        train_data = self.DataClass(split='train', transform=train_transform, download=self.download, as_rgb=self.as_rgb, size=self.size)
        val_data = self.DataClass(split='val', transform=eval_transform, download=self.download, as_rgb=self.as_rgb, size=self.size)
        test_data = self.DataClass(split='test', transform=eval_transform, download=self.download, as_rgb=self.as_rgb, size=self.size)

        self.train_dataset = PairedD4MedMNIST2D(train_data, split='train')
        self.val_dataset = PairedD4MedMNIST2D(val_data, split='val')
        self.test_dataset = PairedD4MedMNIST2D(test_data, split='test')


    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=15)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=15)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=15)
    

if __name__ == '__main__':
    
    
    data_module = D4MedMNISTDataModule('organmnist3d', 128, resize=False, as_rgb=True, size=28, download=False)
    data_module.setup()
    train_loader = data_module.train_dataloader()
    val_loader = data_module.val_dataloader()
    test_loader = data_module.test_dataloader()
    print(len(train_loader), len(val_loader), len(test_loader))
    for batch in train_loader:
        (x1, y1), (x2, y2), transformation_type, covariate = batch
        print(x1.shape, y1.shape, x2.shape, y2.shape)#, transformation_type, covariate)
        break
    for batch in val_loader:
        (x1, y1), (x2, y2), transformation_type, covariate = batch
        print(x1.shape, y1.shape, x2.shape, y2.shape)#, transformation_type, covariate)
        break
    for batch in test_loader:
        (x1, y1), (x2, y2), transformation_type, covariate = batch
        print(x1.shape, y1.shape, x2.shape, y2.shape)#, transformation_type, covariate)
        break
    