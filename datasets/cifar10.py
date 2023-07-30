import numpy as np
import cv2
from torchvision import datasets
import albumentations as A

from .generic import MyDataSet


class AlbCIFAR10(datasets.CIFAR10):
    def __init__(self, root, alb_transform=None, **kwargs):
        super(AlbCIFAR10, self).__init__(root, **kwargs)
        self.alb_transform = alb_transform

    def __getitem__(self, index):
        image, label = super(AlbCIFAR10, self).__getitem__(index)
        if self.alb_transform is not None:
            image = self.alb_transform(image=np.array(image))['image']
        return image, label


class CIFAR10(MyDataSet):
    DataSet = AlbCIFAR10
    mean = (0.49139968, 0.48215827, 0.44653124)
    std = (0.24703233, 0.24348505, 0.26158768)
    default_alb_transforms = [
        A.ToGray(p=0.2),
        A.PadIfNeeded(40, 40, p=1),
        A.RandomCrop(32, 32, p=1),
        A.HorizontalFlip(p=0.5),
        # Since normalisation was the first step, mean is already 0, so cutout fill_value = 0
        A.CoarseDropout(max_holes=1, max_height=8, max_width=8, fill_value=0, p=1)
    ]
