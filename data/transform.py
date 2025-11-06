import cv2
import torch
import numpy as np
from skimage import transform
import torchvision.transforms as transforms
import random
# from PIL import Image, ImageFilter, ImageEnhance
random.seed(1997)

class Compose(object):
    def __init__(self, *ops):
        self.ops = ops

    def __call__(self, hr, lr, feats):
        for op in self.ops:
            hr, lr, feats = op(hr, lr, feats)
        return hr, lr, feats

class ToTensor(object):
    def __init__(self, is_normalize: bool = False, mode: str = "train") -> None:
        self.is_normalize = is_normalize
        self.mode = mode

    def __call__(self, hr, lr, feats):
        hr = hr[:, :, np.newaxis].astype(np.float32)
        hr = torch.from_numpy(hr)
        hr = hr.permute(2, 0, 1)
        lr = lr[:, :, np.newaxis].astype(np.float32)
        lr = torch.from_numpy(lr)
        lr = lr.permute(2, 0, 1)

        feats_tensor = [self.__to_tensor__(feat) for feat in feats]

        if self.is_normalize:
            if self.mode == "train":
                hr = (hr - 1012) / (1819 - 1012)
            else:
                hr = (hr + 17) / (593 + 17)
            lr = (lr + 17) / (593 + 17)
            hr = transforms.Normalize((0.5,), (0.5,))(hr)
            lr = transforms.Normalize((0.5,), (0.5,))(lr)

        return hr, lr, feats_tensor 

    def __to_tensor__(self, input):
        input = input[:, :, np.newaxis].astype(np.float32)
        input = torch.from_numpy(input)
        input = input.permute(2, 0, 1)
        return input

class RandomScaleCrop(object):
    '''
    scale = [1,1,1,1,1.5,1.5,2,2.5]
    '''

    def __init__(self, crop_size, fill=0, scale=2):
        self.crop_size_h = crop_size
        self.crop_size_l = crop_size // scale
        self.fill = 0
        self.scale = scale

    def __call__(self, hr, lr):

        H, W = lr.shape
        if H == self.crop_size_l and W == self.crop_size_l:
            return hr, lr
        crop_h = random.randint(0, H-self.crop_size_l)
        crop_W = random.randint(0, W-self.crop_size_l)

        hr = hr[crop_h*self.scale:crop_h*self.scale+self.crop_size_h,
                crop_W*self.scale:crop_W*self.scale+self.crop_size_h]
        lr = lr[crop_h:crop_h+self.crop_size_l, crop_W:crop_W+self.crop_size_l]

        return hr, lr


class RandomHorizontalFlip(object):
    def __call__(self, hr, lr, feats):
        if random.random() < 0.5:
            hr = hr[:, ::-1].copy()
            lr = lr[:, ::-1].copy()
            feats = [feat[:, ::-1].copy() for feat in feats]

        return hr, lr, feats
    

class RandomVerticalFlip(object):
    def __call__(self, hr, lr, feats):
        if random.random() < 0.5:
            hr = hr[::-1, :].copy()
            lr = lr[::-1, :].copy()
            feats = [feat[::-1, :].copy() for feat in feats]
        return hr, lr, feats


class RandomRotation(object):
    def __init__(self, rotation_lists=[0, 90, 180, 270]):
        self.rotation_lists = rotation_lists

    def __call__(self, hr, lr):
        self.rotation = random.choice(self.rotation_lists)
        hr = (transform.rotate(hr, self.rotation)).astype(np.float32)
        lr = (transform.rotate(lr, self.rotation)).astype(np.float32)

        return hr, lr

if __name__ == '__main__':
    data_path = "The file address where your data is stored"
    hr = cv2.imread(data_path, cv2.IMREAD_UNCHANGED)
    lr = transform.rotate(hr, 90).astype(np.float32)
    a = 1
