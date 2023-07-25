import numpy as np
from torchvision import datasets
import albumentations as A

from .generic import MyDataSet


class AlbMNIST(datasets.MNIST):
    def __init__(self, root, alb_transform=None, **kwargs):
        super(AlbMNIST, self).__init__(root, **kwargs)
        self.alb_transform = alb_transform

    def __getitem__(self, index):
        image, label = super(AlbMNIST, self).__getitem__(index)
        if self.alb_transform is not None:
            image = self.alb_transform(image=np.array(image))['image']
        return image, label


class MNIST(MyDataSet):
    DataSet = AlbMNIST
    mean = (0.1307,)
    std = (0.3081,)
    default_alb_transforms = [
        A.Rotate(limit=7, p=1.),
        A.Perspective(scale=0.2, p=0.5, fit_output=False)
    ]
