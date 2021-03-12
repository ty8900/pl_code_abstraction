import pytorch_lightning as pl
import torch

from abc import *


class AbstractModel(pl.LightningModule, metaclass=ABCMeta):
    def __init__(self, args):
        super().__init__()
        self.args = args
        # self.model_init_seed = args.model_init_seed
        # self.model_init_range = args.model_init_range

    @classmethod
    @abstractmethod
    def code(cls):
        pass

    def configure_optimizers(self):
        if self.args.optimizer == 'adam':
            optimizer = torch.optim.Adam(self.parameters(), lr=self.args.lr)
        elif self.args.optimizer == 'sgd':
            optimizer = torch.optim.SGD(self.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay)

        scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                    step_size=self.args.step_size, gamma=self.args.gamma)
        return [optimizer], [scheduler]
