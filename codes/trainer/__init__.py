from .base import AbstractTrainer
from codes.datamodule import datamodule_factory
from codes.model import model_factory
from codes.utils.factory import all_subclasses, import_all_classes
import_all_classes(__file__, __name__, AbstractTrainer)

TRAINER = {c.code(): c
           for c in all_subclasses(AbstractTrainer)
           if c.code() is not None}


def trainer_factory(args):
    dm = datamodule_factory(args)
    model = model_factory(args)
    trainer = TRAINER[args.trainer_name]
    return trainer(args, dm, model)
