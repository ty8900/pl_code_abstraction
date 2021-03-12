import pytorch_lightning as pl
from torch.utils.data import DataLoader

from abc import *


class AbstractDataModule(pl.LightningDataModule):
    def __init__(self, args, ds):
        super().__init__()
        self.args = args
        self.dataset = ds.load_dataset()
        self.train_batch_size = args.train_batch_size
        self.val_batch_size = args.val_batch_size
        self.test_batch_size = args.test_batch_size

    def train_dataloader(self):
        return DataLoader(self.dataset['train'], num_workers=8,
                          pin_memory=True, shuffle=True, collate_fn=self.get_collate_fn(),
                          batch_size=self.train_batch_size)

    def val_dataloader(self):
        return DataLoader(self.dataset['validation'], num_workers=8,
                          pin_memory=True, collate_fn=self.get_collate_fn(),
                          batch_size=self.val_batch_size)

    def test_dataloader(self):
        return DataLoader(self.dataset['test'], num_workers=8,
                          pin_memory=True, collate_fn=self.get_collate_fn(),
                          batch_size=self.test_batch_size)