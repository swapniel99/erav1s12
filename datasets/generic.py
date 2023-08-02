import os
from abc import ABC
from functools import cached_property

import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2

try:
    from epoch.utils import plot_examples
except ModuleNotFoundError:
    from utils import plot_examples


class MyDataSet(ABC):
    DataSet = None
    mean = None
    std = None
    classes = None
    default_alb_transforms = None

    def __init__(self, batch_size=1, normalize=True, shuffle=True, augment=True, alb_transforms=None):
        self.batch_size = batch_size
        self.normalize = normalize
        self.shuffle = shuffle
        self.augment = augment
        self.alb_transforms = alb_transforms or self.default_alb_transforms

        self.loader_kwargs = {'batch_size': batch_size, 'num_workers': os.cpu_count(), 'pin_memory': True}

    @classmethod
    def set_classes(cls, data):
        if cls.classes is None:
            cls.classes = {i: c for i, c in enumerate(data.classes)}

    @cached_property
    def train_data(self):
        res = self.DataSet('../data', train=True, download=True, alb_transform=self.get_train_transforms())
        self.set_classes(res)
        return res

    @cached_property
    def test_data(self):
        res = self.DataSet('../data', train=False, download=True, alb_transform=self.get_test_transforms())
        self.set_classes(res)
        return res

    @cached_property
    def train_loader(self):
        return torch.utils.data.DataLoader(self.train_data, shuffle=self.shuffle, **self.loader_kwargs)

    @cached_property
    def test_loader(self):
        return torch.utils.data.DataLoader(self.test_data, shuffle=False, **self.loader_kwargs)

    @cached_property
    def example_iter(self):
        return iter(self.train_loader)

    def get_train_transforms(self):
        all_transforms = list()
        if self.normalize:
            all_transforms.append(A.Normalize(self.mean, self.std))
        if self.augment and self.alb_transforms is not None:
            all_transforms.extend(self.alb_transforms)
        all_transforms.append(ToTensorV2())
        return A.Compose(all_transforms)

    def get_test_transforms(self):
        all_transforms = list()
        if self.normalize:
            all_transforms.append(A.Normalize(self.mean, self.std))
        all_transforms.append(ToTensorV2())
        return A.Compose(all_transforms)

    def download(self):
        self.DataSet('../data', train=True, download=True)
        self.DataSet('../data', train=False, download=True)

    def denormalise(self, tensor):
        result = tensor.clone().detach().requires_grad_(False)
        if self.normalize:
            for t, m, s in zip(result, self.mean, self.std):
                t.mul_(s).add_(m)
        return result

    def show_transform(self, img):
        if self.normalize:
            img = self.denormalise(img)
        if len(self.mean) == 3:
            return img.permute(1, 2, 0)
        else:
            return img.squeeze(0)

    def show_examples(self, figsize=(8, 6)):
        batch_data, batch_label = next(self.example_iter)
        images = list()
        labels = list()

        for i in range(len(batch_data)):
            image = batch_data[i]
            image = self.show_transform(image)

            label = batch_label[i].item()
            if self.classes is not None:
                label = f'{label}:{self.classes[label]}'

            images.append(image)
            labels.append(label)

        plot_examples(images, labels, figsize=figsize)
