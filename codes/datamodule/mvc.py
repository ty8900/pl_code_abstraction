import torch
from torchvision import transforms
from PIL import Image

from .base import AbstractDataModule


class MVCDataModule(AbstractDataModule):
    def __init__(self, args, ds):
        super().__init__(args, ds)

    @classmethod
    def code(cls):
        return 'mvc'

    def get_collate_fn(self):
        return collate


def collate(samples):
    out = {'label': torch.stack([torch.tensor(sample['label']) for sample in samples]),
           'image': torch.stack([get_img(sample['input']) for sample in samples])}
    return out


def get_img(path):
    img = Image.open(path)
    return transforms.ToTensor()(img)
