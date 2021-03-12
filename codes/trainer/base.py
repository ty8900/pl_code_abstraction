import pytorch_lightning as pl
from abc import *


class AbstractTrainer(metaclass=ABCMeta):
    def __init__(self, args, dm, model):
        self.dm = dm
        print(args.pretrained_model_path)
        if args.train_from_ckpt:
            self.model_path = args.pretrained_model_path
            self.model = model.load_from_checkpoint(args=args, checkpoint_path=self.model_path)
            self.trainer = pl.Trainer(gpus=args.gpus, max_epochs=args.max_epochs,
                                      progress_bar_refresh_rate=args.bar_refresh_rate,
                                      resume_from_checkpoint=self.model_path)
        else:
            self.model = model
            self.trainer = pl.Trainer(gpus=args.gpus, max_epochs=args.max_epochs,
                                      progress_bar_refresh_rate=args.bar_refresh_rate)

    @classmethod
    @abstractmethod
    def code(cls):
        pass

    def run(self):
        self.dm.setup('fit')
        self.trainer.fit(self.model, self.dm)
        self.trainer.test(self.model, self.dm.test_dataloader())
