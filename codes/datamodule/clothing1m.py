import torch
from torchvision import transforms
from PIL import Image

from .base import AbstractDataModule


class Clothing1MDataModule(AbstractDataModule):
    def __init__(self, args, ds):
        super().__init__(args, ds)

    @classmethod
    def code(cls):
        return 'clothing1m'

    def get_collate_fn(self):
        return collate


def collate(samples):
    out = {'label': torch.stack([torch.LongTensor(sample['label']) for sample in samples]),
           'image': torch.stack([get_img(sample['input']) for sample in samples]),
           'index': torch.stack([torch.LongTensor(sample['index']) for sample in samples])}
    return out


def get_img(path):
    img = Image.open(path)
    ret = transforms.ToTensor()(img)
    if ret.shape[0] != 3:
        ret = torch.cat([ret, ret, ret])
        print("convert grayscale to RGB")
        img.convert("RGB").save(path)
    return ret
