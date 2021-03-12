from codes.datamodule.base import AbstractDataModule
from codes.dataset import dataset_factory
from codes.utils.factory import all_subclasses, import_all_classes
import_all_classes(__file__, __name__, AbstractDataModule)

DATAMODULE = {c.code(): c
              for c in all_subclasses(AbstractDataModule)
              if c.code() is not None}


def datamodule_factory(args):
    ds = dataset_factory(args)
    datamodule = DATAMODULE[args.datamodule_name]
    return datamodule(args, ds)
