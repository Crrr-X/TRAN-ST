from __future__ import annotations
import torch
from torch.utils.data import Dataset
from src.data.utils import transform
from osgeo import gdal

gdal.DontUseExceptions()
# from libtiff import TIFF

class DemDataset(Dataset):
# A customized Dataset class
    def __init__(self, data_path: str, mode: str = "train", crop_size: int = 96, scale: int = 4):
    
        """Init dataset.

        Args:
            data_path (str): data path (end with '.txt').
            mode (str, optional): distinguish between train_dataset and val_dataset. Defaults to "train".
            crop_size (int, optional): crop size of input image. Defaults to 96.
            scale (int, optional): scale of reconstruction. Defaults to 2.
        """
        super(DemDataset, self).__init__()
        self.mode = mode
        with open(data_path, "r", errors='ignore') as lines: 
            self.samples = []
            for line in lines:
                hr_path = line.strip().split(" ")[0]
                lr_path = line.strip().split(" ")[1]
                feats = line.strip().split(" ")[2:] 
                self.samples.append([hr_path, lr_path, feats])
        self.samples.sort()
        if mode == "train":
            self.transform_scale = transform.Compose(transform.RandomScaleCrop(crop_size=crop_size, scale=scale), 
                                                     transform.RandomHorizontalFlip(),
                                                     transform.RandomVerticalFlip(),
                                                     transform.RandomRotation(),
                                                     transform.ToTensor(mode=mode))
        if mode == "val":
            self.transform = transform.Compose(transform.ToTensor(mode=mode))
            self.reverse = False
    

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Override function (__getitem__) of Dataset

        Args:
            index (int): _description_

        Returns:
            _type_: _description_
        """
        hr_path, lr_path, feats_path = self.samples[index]
        if not hr_path or not lr_path:
            raise ValueError
        hr = gdal.Open(hr_path).ReadAsArray()
        lr = gdal.Open(lr_path).ReadAsArray()
        feats = []
        if len(feats_path):
            for feat in feats_path:
                feats.append(gdal.Open(feat).ReadAsArray())
        if self.mode == "train":
            hr, lr, feats = self.transform_scale(hr, lr, feats)
        if self.mode == "val":
            hr, lr, feats = self.transform(hr, lr, feats)

        return hr, lr, feats
    
    def __len__(self) -> int:
        return len(self.samples)
