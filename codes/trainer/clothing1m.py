from codes.trainer.base import AbstractTrainer


class Clothing1MTrainer(AbstractTrainer):
    def __init__(self, args, dm, model):
        super().__init__(args, dm, model)

    @classmethod
    def code(cls):
        return 'clothing1m'
